"""
utils/metrics.py
=================
Evaluation metrics for multi-class segmentation.

- Dice Score per class (binary: class i vs rest)
- HD95 per class via medpy
- test_single_volume: matches TransUNet evaluation exactly
  (slice-by-slice prediction on 3D volume, then compute per-class Dice/HD95)
"""

import numpy as np
import torch
from scipy.ndimage import zoom

try:
    from medpy import metric as medmetric
    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False
    print("[WARNING] medpy not found. HD95 will not be computed.")


def calculate_metric_percase(pred, gt):
    """
    Compute binary Dice + HD95 for one class.
    pred, gt : binary 3D numpy arrays (D, H, W)
    """
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        if MEDPY_AVAILABLE:
            dice = medmetric.binary.dc(pred, gt)
            hd95 = medmetric.binary.hd95(pred, gt)
        else:
            # Fallback dice calculation
            intersection = (pred & gt).sum()
            dice = 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)
            hd95 = float("nan")
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1.0, 0.0
    else:
        return 0.0, 0.0


@torch.no_grad()
def test_single_volume(image, label, net, classes, patch_size=(224, 224), device="cuda", is_2_5d=False):
    """
    Evaluate a single 3D volume slice-by-slice.
    Matches TransUNet test_single_volume implementation.

    Args:
        image  : [1, D, H, W] or [D, H, W] tensor
        label  : [1, D, H, W] or [D, H, W] tensor
        net    : model in eval mode
        classes: number of segmentation classes (including background)
        patch_size: (H, W) to resize each slice to before inference
        is_2_5d: if True, feeds 3 slices (z-1, z, z+1) instead of 1 slice.

    Returns:
        metric_list : list of (dice, hd95) for classes 1..classes-1
    """
    image = image.squeeze(0).cpu().numpy()  # (D, H, W)
    label = label.squeeze(0).cpu().numpy()  # (D, H, W)

    net.eval()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label, dtype=np.int64)
        for i in range(image.shape[0]):
            if is_2_5d:
                sl_prev = image[i-1] if i > 0 else image[i]
                sl_z = image[i]
                sl_next = image[i+1] if i < image.shape[0]-1 else image[i]
                
                # Stack them: (3, H, W)
                sl = np.stack([sl_prev, sl_z, sl_next], axis=0)
                x, y = sl_z.shape
            else:
                sl = image[i]             # (H, W)
                x, y = sl.shape

            ph, pw = patch_size
            if x != ph or y != pw:
                if is_2_5d:
                    sl = zoom(sl, (1.0, ph / x, pw / y), order=3)
                else:
                    sl = zoom(sl, (ph / x, pw / y), order=3)
                    
            if is_2_5d:
                inp = torch.from_numpy(sl.astype(np.float32)).unsqueeze(0).to(device) # [1, 3, H, W]
            else:
                inp = torch.from_numpy(sl.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, H, W]
                
            with torch.no_grad():
                out = torch.argmax(torch.softmax(net(inp), dim=1), dim=1).squeeze(0)
                out = out.cpu().numpy().astype(np.int64)
            if x != ph or y != pw:
                out = zoom(out, (x / ph, y / pw), order=0).astype(np.int64)
            prediction[i] = out
    else:
        inp = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(inp), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().numpy().astype(np.int64)

    metric_list = []
    for cls in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == cls, label == cls))
    return metric_list


def dice_coef_from_logits(logits, targets, num_classes, smooth=1e-5):
    """
    Compute per-class Dice from logits for monitoring during training.
    Returns mean dice over foreground classes (1..num_classes-1).

    Args:
        logits  : [B, C, H, W] tensor
        targets : [B, H, W] long tensor
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        preds = torch.argmax(probs, dim=1)    # [B, H, W]

        dice_per_class = []
        for cls in range(1, num_classes):
            pred_c = (preds == cls).float()
            gt_c   = (targets == cls).float()
            intersect = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            if union < 1:
                # No predictions and no GT for this class -> Dice = 1.0 (trivially correct)
                dice_per_class.append(1.0)
            else:
                dice_per_class.append(((2 * intersect + smooth) / (union + smooth)).item())

        return float(np.mean(dice_per_class)) if dice_per_class else 0.0
