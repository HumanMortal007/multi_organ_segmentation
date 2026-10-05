"""
evaluate.py
===========
Standalone evaluation script. Load a trained checkpoint and evaluate
on the Synapse test volumes.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth --save_vis

    (With venv)
    venv/Scripts/python.exe evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import os
import time
import numpy as np
import torch
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet2d import build_unet2d
from datasets.synapse_dataset import get_test_loader
from utils.metrics import test_single_volume
from utils.visualization import save_prediction_panel, save_eval_results_table
from utils.gpu_monitor import get_gpu_info, print_gpu_info
from utils.timing import Timer
from scipy.ndimage import zoom as spzoom


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--config",     default="configs/unet2d_config.yaml")
    p.add_argument("--save_vis",   action="store_true", help="Save prediction visualizations")
    p.add_argument("--n_vis",      type=int, default=3,  help="Number of cases to visualise")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print_gpu_info()

    # Build model & load checkpoint
    model = build_unet2d(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {ckpt.get('epoch', '?')} | Best Dice: {ckpt.get('best_dice', '?'):.4f}")

    # Test loader
    test_loader, test_dataset = get_test_loader(cfg)
    num_classes = cfg["dataset"]["num_classes"]
    patch_size  = cfg["evaluation"]["patch_size"]
    class_names = cfg["dataset"]["class_names"]
    vis_dir     = cfg["results"]["vis_dir"]


    print(f"Evaluating {len(test_dataset)} test volumes...")

    print("=" * 60)

    metric_list = np.zeros((num_classes - 1, 2))
    timer = Timer(total_epochs=len(test_dataset))
    timer.start_experiment()

    for i_batch, batch in enumerate(test_loader):
        image     = batch["image"]          # [1, D, H, W]
        label     = batch["label"]          # [1, D, H, W]
        case_name = batch["case_name"][0]

        t0 = time.time()
        metrics = test_single_volume(image, label, model,
                                     classes=num_classes,
                                     patch_size=(patch_size, patch_size),
                                     device=str(device))
        elapsed = time.time() - t0

        metrics_arr = np.array(metrics)
        metric_list += metrics_arr
        per_dice = metrics_arr[:, 0].mean()
        print(f"  [{i_batch+1}/{len(test_dataset)}] {case_name}: dice={per_dice:.4f} ({elapsed:.1f}s)")

        # Visualize
        if args.save_vis and i_batch < args.n_vis:
            img_vol = image.squeeze(0).numpy()
            lbl_vol = label.squeeze(0).numpy()
            D = img_vol.shape[0]
            mid = D // 2
            sl_img = img_vol[mid]; sl_lbl = lbl_vol[mid].astype(np.int64)
            x, y = sl_img.shape
            if x != patch_size or y != patch_size:
                sl_r = spzoom(sl_img, (patch_size/x, patch_size/y), order=3)
            else: sl_r = sl_img
            inp = torch.from_numpy(sl_r.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = torch.argmax(torch.softmax(model(inp), dim=1), dim=1).squeeze().cpu().numpy()
            if x != patch_size or y != patch_size:
                out = spzoom(out, (x/patch_size, y/patch_size), order=0).astype(np.int64)
            save_prediction_panel(sl_img, sl_lbl, out, case_name, vis_dir, slice_idx=mid)

    metric_list /= len(test_dataset)
    per_class_dice = metric_list[:, 0].tolist()
    per_class_hd95 = metric_list[:, 1].tolist()

    mean_dice, mean_hd = save_eval_results_table(
        per_class_dice, per_class_hd95, class_names, cfg["results"]["logs_dir"]
    )
    print(f"Total eval time: {timer.format_time(timer.total_elapsed())}")


if __name__ == "__main__":
    main()
