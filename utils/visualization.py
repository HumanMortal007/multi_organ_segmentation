"""
utils/visualization.py
========================
Visualization utilities for multi-organ segmentation results.

Saves:
 - Training curves (loss, Dice)
 - Sample prediction panels (CT, GT, Pred, Overlay)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# 9-class colour map (index 0 = background = black)
ORGAN_COLORS = [
    (0.0,  0.0,  0.0,  1.0),   # 0 background  - black
    (1.0,  0.0,  0.0,  1.0),   # 1 aorta        - red
    (1.0,  1.0,  0.0,  1.0),   # 2 gallbladder  - yellow
    (0.0,  0.0,  1.0,  1.0),   # 3 spleen        - blue
    (0.0,  1.0,  0.0,  1.0),   # 4 left kidney  - green
    (0.0,  1.0,  1.0,  1.0),   # 5 right kidney - cyan
    (1.0,  0.5,  0.0,  1.0),   # 6 liver         - orange
    (0.5,  0.0,  0.5,  1.0),   # 7 stomach       - purple
    (1.0,  0.0,  1.0,  1.0),   # 8 pancreas      - magenta
]
CMAP = mcolors.ListedColormap([c[:3] for c in ORGAN_COLORS])
NORM = mcolors.BoundaryNorm(boundaries=range(10), ncolors=9)


def save_training_curves(history, save_dir):
    """Save loss and Dice training curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(epochs, history["train_dice"], "b-o", label="Train Dice", markersize=4)
    axes[1].plot(epochs, history["val_dice"],   "r-o", label="Val Dice",   markersize=4)
    axes[1].set_title("Dice Score"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[Visualization] Training curves saved to {save_dir}/training_curves.png")


def save_prediction_panel(image, label, pred, case_name, save_dir, slice_idx=None):
    """
    Save a 4-panel figure: CT | Ground Truth | Prediction | Overlay.

    Args:
        image     : (H, W) numpy float array
        label     : (H, W) numpy int array
        pred      : (H, W) numpy int array
        case_name : str, used for file name
        save_dir  : directory to save to
        slice_idx : optional slice index for naming
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create coloured segmentation images
    def to_rgb_seg(seg_map):
        h, w = seg_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for cls_idx, color in enumerate(ORGAN_COLORS):
            mask = (seg_map == cls_idx)
            rgb[mask] = color[:3]
        return rgb

    gt_rgb   = to_rgb_seg(label)
    pred_rgb = to_rgb_seg(pred)

    # Overlay: CT as gray + prediction as alpha
    ct_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fname = f"{case_name}_s{slice_idx}" if slice_idx is not None else case_name
    fig.suptitle(fname, fontsize=12, fontweight="bold")

    axes[0].imshow(ct_norm, cmap="gray"); axes[0].set_title("CT Slice"); axes[0].axis("off")
    axes[1].imshow(gt_rgb);  axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pred_rgb); axes[2].set_title("Prediction"); axes[2].axis("off")

    axes[3].imshow(ct_norm, cmap="gray")
    overlay = pred_rgb.copy()
    overlay_alpha = np.any(pred_rgb > 0, axis=2).astype(np.float32) * 0.5
    axes[3].imshow(pred_rgb, alpha=overlay_alpha); axes[3].set_title("Overlay"); axes[3].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_eval_results_table(per_class_dice, per_class_hd95, class_names, save_dir):
    """Print and save final evaluation results table."""
    os.makedirs(save_dir, exist_ok=True)
    lines = []
    lines.append("=" * 60)
    lines.append("FINAL EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"{'Class / Organ':<25} {'Dice':>10} {'HD95':>10}")
    lines.append("-" * 60)

    dice_vals = []
    hd95_vals = []
    for i, (name, d, h) in enumerate(zip(class_names[1:], per_class_dice, per_class_hd95)):
        hd_str = f"{h:10.4f}" if not (isinstance(h, float) and h != h) else "      N/A"
        lines.append(f"Class {i+1} / {name:<18} {d:10.4f} {hd_str}")
        dice_vals.append(d)
        if not (isinstance(h, float) and h != h):
            hd95_vals.append(h)

    lines.append("-" * 60)
    mean_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
    mean_hd   = float(np.mean(hd95_vals)) if hd95_vals else float("nan")
    hd_str = f"{mean_hd:10.4f}" if hd95_vals else "      N/A"
    lines.append(f"{'Mean Dice / HD95':<25} {mean_dice:10.4f} {hd_str}")
    lines.append("=" * 60)

    result_str = "\n".join(lines)
    print(result_str)

    log_path = os.path.join(save_dir, "eval_results.txt")
    with open(log_path, "w") as f:
        f.write(result_str + "\n")
    print(f"[Evaluation] Results saved to {log_path}")
    return mean_dice, mean_hd
