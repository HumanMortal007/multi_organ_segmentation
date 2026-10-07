"""
scripts/evaluate.py
===================
Unified standalone evaluation script for all three trained models.
Loads a saved .pth checkpoint and runs inference — NO retraining.

Supports:
  --model unet        (2D Base U-Net)
  --model transunet   (2D TransUNet)
  --model unet_2_5d   (2.5D U-Net)

Calculates:
  1. Macro Mean Dice (existing, unchanged formula)
  2. Weighted Mean Dice (NEW — weighted by GT foreground volume per organ per case)
  3. Per-organ Dice
  4. HD95 per organ (existing, unchanged — units = pixels, no voxel spacing)

Saves:
  - experiments/<model>/results/evaluation/<timestamp>/metrics.json
  - experiments/<model>/results/evaluation/<timestamp>/metrics.csv
  - experiments/<model>/results/evaluation/<timestamp>/summary.txt

Usage (from project root with venv activated):
  venv/Scripts/python.exe scripts/evaluate.py --model unet       --checkpoint experiments/base_unet/checkpoints/best_model.pth
  venv/Scripts/python.exe scripts/evaluate.py --model transunet  --checkpoint experiments/transunet/checkpoints/best_model.pth
  venv/Scripts/python.exe scripts/evaluate.py --model unet_2_5d --checkpoint experiments/unet_2_5d/checkpoints/best_model.pth

Optional flags:
  --save_vis          Save CT / GT / Pred / Overlay panels for N cases
  --n_vis 3           How many cases to visualize (default 3)
  --foreground_slices Select the slice with the most foreground for visualization
                      (default: middle slice of volume)

============================================================
METRIC DEFINITIONS
============================================================

Macro Mean Dice (existing, preserved):
    For each test volume: dice_c = Dice(pred_c, gt_c)  for each class c
    Aggregate: mean across all cases, then mean across all classes.
    Formula: mean_dice = mean_c( mean_case( dice_c ) )
    Every organ has equal weight regardless of physical size.

Weighted Mean Dice (NEW, additive):
    For each test case:
        weight_c = GT_foreground_pixels_c / total_GT_foreground_pixels_in_case
        weighted_dice_case = sum_c( weight_c * dice_c )
    Aggregate: weighted_mean_dice = mean_case( weighted_dice_case )
    Larger organs contribute more. Absent organs contribute weight=0.
    Cases with zero foreground are excluded from aggregation.

HD95 (existing, preserved):
    medpy.metric.binary.hd95(pred, gt) with default voxelspacing=1.
    Units: image-grid pixels/voxels (NOT millimeters).
    Empty masks -> HD95 = 0.0  (existing behavior, preserved).

============================================================
KNOWN METRIC ISSUES (NOT CHANGED — REPORTED ONLY)
============================================================

Issue 1: GT absent + pred absent -> Dice=0.0
    Logically should be 1.0 (model correctly predicted nothing).
    Kept as-is for backward compatibility with existing results.

Issue 2: GT absent + pred present -> Dice=1.0
    This is a false-positive hallucination but returns Dice=1.0.
    Logically inconsistent. Preserved for backward compatibility.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from scipy.ndimage import zoom as spzoom
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.synapse_dataset import get_test_loader
from models.transunet import build_transunet
from models.unet2d import build_unet2d
from models.unet_2_5d import build_unet_2_5d
from utils.metrics import test_single_volume
from utils.visualization import save_prediction_panel


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained segmentation model without retraining."
    )
    p.add_argument("--model", type=str, required=True,
                   choices=["unet", "transunet", "unet_2_5d"],
                   help="Model architecture to evaluate.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the saved .pth checkpoint file.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config (auto-selected if omitted).")
    p.add_argument("--save_vis", action="store_true",
                   help="Save CT / GT / Pred / Overlay visualization panels.")
    p.add_argument("--n_vis", type=int, default=3,
                   help="Number of test cases to visualize (default 3).")
    p.add_argument("--foreground_slices", action="store_true",
                   help="Pick the slice with the most foreground for visualization.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Config resolution
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIGS = {
    "unet":      "configs/unet2d_config.yaml",
    "transunet": "configs/transunet.yaml",
    "unet_2_5d": "configs/unet_2_5d_config.yaml",
}


def load_config(args):
    config_path = args.config if args.config else _DEFAULT_CONFIGS[args.model]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"[Config] Loaded: {config_path}")
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Model construction + checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_name, cfg, device):
    if model_name == "unet":
        return build_unet2d(cfg).to(device)
    elif model_name == "transunet":
        return build_transunet(cfg).to(device)
    elif model_name == "unet_2_5d":
        return build_unet_2_5d(cfg).to(device)
    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch     = ckpt.get("epoch",     "?")
    best_dice = ckpt.get("best_dice", float("nan"))
    print(f"[Checkpoint] Loaded : {checkpoint_path}")
    print(f"             Epoch={epoch}  |  Stored best_dice={best_dice:.4f}")
    return model, epoch, best_dice


# ──────────────────────────────────────────────────────────────────────────────
# Weighted Mean Dice
# ──────────────────────────────────────────────────────────────────────────────

def compute_weighted_dice_case(dice_list, label_3d, num_classes):
    """
    GT-volume-weighted Dice for a single test case.

    Formula:
        weight_c = GT_pixels_c / sum_c(GT_pixels_c)
        weighted_dice = sum_c( weight_c * dice_c )

    Returns None if no foreground GT exists in this volume.
    """
    fg_counts = [float((label_3d == cls).sum()) for cls in range(1, num_classes)]
    total_fg = sum(fg_counts)
    if total_fg == 0:
        return None
    weighted = sum((cnt / total_fg) * d for cnt, d in zip(fg_counts, dice_list))
    return float(weighted)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helper
# ──────────────────────────────────────────────────────────────────────────────

def pick_vis_slice(img_vol, lbl_vol, foreground_slices=False):
    D = img_vol.shape[0]
    if foreground_slices:
        fg_counts = [(lbl_vol[z] > 0).sum() for z in range(D)]
        return int(np.argmax(fg_counts))
    return D // 2


def run_single_slice_inference(model, img_vol, slice_idx, patch_size, device, is_2_5d=False):
    D, H, W = img_vol.shape
    if is_2_5d:
        z = slice_idx
        sl_prev = img_vol[z - 1] if z > 0 else img_vol[z]
        sl_z    = img_vol[z]
        sl_next = img_vol[z + 1] if z < D - 1 else img_vol[z]
        sl = np.stack([sl_prev, sl_z, sl_next], axis=0)  # (3, H, W)
        sl_res = spzoom(sl, (1.0, patch_size / H, patch_size / W), order=3) if (H != patch_size or W != patch_size) else sl
        inp = torch.from_numpy(sl_res.astype(np.float32)).unsqueeze(0).to(device)
        vis_img = sl_z
    else:
        sl = img_vol[slice_idx]
        sl_res = spzoom(sl, (patch_size / H, patch_size / W), order=3) if (H != patch_size or W != patch_size) else sl
        inp = torch.from_numpy(sl_res.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        vis_img = sl

    with torch.no_grad():
        out = torch.argmax(torch.softmax(model(inp), dim=1), dim=1).squeeze(0).cpu().numpy()

    if H != patch_size or W != patch_size:
        out = spzoom(out, (H / patch_size, W / patch_size), order=0).astype(np.int64)
    return vis_img, out


# ──────────────────────────────────────────────────────────────────────────────
# Result saving
# ──────────────────────────────────────────────────────────────────────────────

def make_output_dir(cfg):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = cfg["results"]["dir"]
    out_dir = os.path.join(base, "evaluation", ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ts


def save_results(out_dir, model_name, checkpoint_path, ckpt_epoch, ckpt_best_dice,
                 class_names, per_class_dice, per_class_hd95,
                 macro_mean_dice, weighted_mean_dice, mean_hd95,
                 n_cases, eval_cfg, ts):

    # ── summary.txt ─────────────────────────────────────────────────────────────
    W = 66
    lines = []
    lines.append("=" * W)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * W)
    lines.append(f"Model             : {model_name}")
    lines.append(f"Checkpoint        : {checkpoint_path}")
    lines.append(f"Checkpoint epoch  : {ckpt_epoch}")
    lines.append(f"Checkpoint Dice   : {ckpt_best_dice:.4f}")
    lines.append(f"Evaluated volumes : {n_cases}")
    lines.append(f"Timestamp         : {ts}")
    lines.append(f"Patch size        : {eval_cfg.get('patch_size', 224)} px")
    lines.append(f"HD95 units        : pixels (voxelspacing=1, NOT mm)")
    lines.append("-" * W)
    lines.append(f"  {'Organ':<22} {'Dice':>8} {'HD95 (px)':>12}")
    lines.append("-" * W)

    for i, name in enumerate(class_names[1:]):
        d = per_class_dice[i]
        h = per_class_hd95[i]
        h_str = f"{h:12.4f}" if not (isinstance(h, float) and h != h) else "         N/A"
        lines.append(f"  {name:<22} {d:8.4f} {h_str}")

    lines.append("-" * W)
    wdice_str = f"{weighted_mean_dice:.4f}" if weighted_mean_dice is not None else "N/A"
    hd_str = f"{mean_hd95:.4f}" if not (isinstance(mean_hd95, float) and mean_hd95 != mean_hd95) else "N/A"
    lines.append(f"")
    lines.append(f"  {'Macro Mean Dice':<28} {macro_mean_dice:.4f}  (unweighted, organs equal)")
    lines.append(f"  {'Weighted Mean Dice':<28} {wdice_str:>6}  (weighted by GT foreground px)")
    lines.append(f"  {'Mean HD95':<28} {hd_str:>6} pixels")
    lines.append("=" * W)
    lines.append("")
    lines.append("METRIC NOTES")
    lines.append(f"Macro Dice    = mean_c( mean_case( dice_c ) )")
    lines.append(f"Weighted Dice = mean_case( sum_c( (GT_px_c / total_GT_px) * dice_c ) )")
    lines.append(f"HD95          = medpy hd95, voxelspacing=1 (pixels, lower is better)")
    lines.append("")
    lines.append("KNOWN ISSUES (reported, not changed)")
    lines.append("  Issue 1: GT absent + pred absent -> Dice=0.0  (logically should be 1.0)")
    lines.append("  Issue 2: GT absent + pred present -> Dice=1.0 (false positive, inconsistent)")
    lines.append("=" * W)

    summary_str = "\n".join(lines)
    print(summary_str)
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_str + "\n")

    # ── metrics.json ────────────────────────────────────────────────────────────
    json_data = {
        "model": model_name,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_best_dice": float(ckpt_best_dice) if ckpt_best_dice == ckpt_best_dice else None,
        "timestamp": ts,
        "n_evaluated_cases": n_cases,
        "patch_size_px": eval_cfg.get("patch_size", 224),
        "hd95_units": "pixels (voxelspacing=1, not mm)",
        "macro_mean_dice": round(float(macro_mean_dice), 6),
        "weighted_mean_dice": round(float(weighted_mean_dice), 6) if weighted_mean_dice is not None else None,
        "mean_hd95_pixels": round(float(mean_hd95), 6) if not (isinstance(mean_hd95, float) and mean_hd95 != mean_hd95) else None,
        "per_organ": {
            name: {
                "dice": round(float(per_class_dice[i]), 6),
                "hd95_pixels": round(float(per_class_hd95[i]), 6)
                               if not (isinstance(per_class_hd95[i], float) and per_class_hd95[i] != per_class_hd95[i])
                               else None,
            }
            for i, name in enumerate(class_names[1:])
        },
        "known_issues": [
            "GT absent + pred absent -> Dice=0.0 (preserved from original; logically should be 1.0)",
            "GT absent + pred present -> Dice=1.0 (false-positive hallucination, preserved for backward compat)"
        ]
    }
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # ── metrics.csv ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["organ", "dice", "hd95_pixels"])
        for i, name in enumerate(class_names[1:]):
            h = per_class_hd95[i]
            h_val = round(float(h), 6) if not (isinstance(h, float) and h != h) else ""
            writer.writerow([name, round(float(per_class_dice[i]), 6), h_val])
        writer.writerow([])
        writer.writerow(["macro_mean_dice", round(float(macro_mean_dice), 6), ""])
        wdice_val = round(float(weighted_mean_dice), 6) if weighted_mean_dice is not None else ""
        writer.writerow(["weighted_mean_dice", wdice_val, ""])
        hd_val = round(float(mean_hd95), 6) if not (isinstance(mean_hd95, float) and mean_hd95 != mean_hd95) else ""
        writer.writerow(["mean_hd95_pixels", "", hd_val])

    print(f"\n[Saved] Summary → {summary_path}")
    print(f"[Saved] JSON    → {json_path}")
    print(f"[Saved] CSV     → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Build model and load checkpoint (NO optimizer, NO gradients, NO training)
    model = build_model(args.model, cfg, device)
    model, ckpt_epoch, ckpt_best_dice = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    test_loader, test_dataset = get_test_loader(cfg)
    num_classes = cfg["dataset"]["num_classes"]   # 9
    class_names = cfg["dataset"]["class_names"]
    patch_size  = cfg["evaluation"]["patch_size"]
    eval_cfg    = cfg["evaluation"]
    vis_dir     = cfg["results"]["vis_dir"]
    is_2_5d     = (args.model == "unet_2_5d")

    print(f"\n[Evaluation] {args.model}  |  {len(test_dataset)} test volumes")
    print("=" * 66)

    macro_metric_list  = np.zeros((num_classes - 1, 2))
    weighted_dice_cases = []
    t_start = time.time()

    for i_batch, batch in enumerate(tqdm(test_loader, desc="Evaluating", unit="vol")):
        image     = batch["image"]
        label     = batch["label"]
        case_name = batch["case_name"][0]
        label_np  = label.squeeze(0).numpy()

        # ── Volumetric inference (existing logic, unchanged) ─────────────────────
        with torch.no_grad():
            metrics = test_single_volume(
                image, label, model,
                classes=num_classes,
                patch_size=(patch_size, patch_size),
                device=str(device),
                is_2_5d=is_2_5d,
            )

        metrics_arr = np.array(metrics)
        macro_metric_list += metrics_arr
        per_case_dice = metrics_arr[:, 0]

        # ── Weighted Dice (new) ──────────────────────────────────────────────────
        wdice = compute_weighted_dice_case(per_case_dice.tolist(), label_np, num_classes)
        if wdice is not None:
            weighted_dice_cases.append(wdice)

        wdice_str = f"{wdice:.4f}" if wdice is not None else "N/A"
        tqdm.write(f"  [{i_batch+1:>2}/{len(test_dataset)}] {case_name}: "
                   f"macro={per_case_dice.mean():.4f}  weighted={wdice_str}")

        # ── Optional visualization ───────────────────────────────────────────────
        if args.save_vis and i_batch < args.n_vis:
            img_vol = image.squeeze(0).numpy()
            lbl_vol = label_np.astype(np.int64)
            sl_idx  = pick_vis_slice(img_vol, lbl_vol, foreground_slices=args.foreground_slices)
            vis_img, pred_2d = run_single_slice_inference(
                model, img_vol, sl_idx, patch_size, device, is_2_5d=is_2_5d
            )
            save_prediction_panel(vis_img, lbl_vol[sl_idx], pred_2d, case_name, vis_dir, slice_idx=sl_idx)

    t_total = time.time() - t_start
    n_cases = len(test_dataset)

    # ── Aggregate ────────────────────────────────────────────────────────────────
    macro_metric_list /= max(n_cases, 1)
    per_class_dice = macro_metric_list[:, 0].tolist()
    per_class_hd95 = macro_metric_list[:, 1].tolist()

    macro_mean_dice    = float(np.mean(per_class_dice))
    weighted_mean_dice = float(np.mean(weighted_dice_cases)) if weighted_dice_cases else None
    valid_hd = [h for h in per_class_hd95 if not (isinstance(h, float) and h != h)]
    mean_hd95 = float(np.mean(valid_hd)) if valid_hd else float("nan")

    print(f"\n[Done] {t_total:.1f}s  |  {n_cases} volumes evaluated")

    out_dir, ts = make_output_dir(cfg)
    save_results(
        out_dir=out_dir, model_name=args.model,
        checkpoint_path=args.checkpoint, ckpt_epoch=ckpt_epoch,
        ckpt_best_dice=ckpt_best_dice, class_names=class_names,
        per_class_dice=per_class_dice, per_class_hd95=per_class_hd95,
        macro_mean_dice=macro_mean_dice, weighted_mean_dice=weighted_mean_dice,
        mean_hd95=mean_hd95, n_cases=n_cases, eval_cfg=eval_cfg, ts=ts,
    )


if __name__ == "__main__":
    main()
