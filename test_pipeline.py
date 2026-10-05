"""
test_pipeline.py
=================
Quick sanity check to verify the entire pipeline without full training.
Tests: dataset loading, model forward pass, loss, backward pass, metrics, volumetric eval.

Usage:
    python test_pipeline.py
    venv/Scripts/python.exe test_pipeline.py
"""

import os
import sys
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader


def run_pipeline_test():
    print("=" * 60)
    print("Pipeline Sanity Test")
    print("=" * 60)

    # ---- Config ----
    with open("configs/unet2d_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override for fast test
    cfg["training"]["batch_size"] = 4
    cfg["training"]["num_workers"] = 0            # num_workers=0 avoids Windows multiproc issues in script
    cfg["training"]["persistent_workers"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        from utils.gpu_monitor import get_gpu_info
        info = get_gpu_info()
        print(f"GPU          : {info['name']}")
        print(f"Total VRAM   : {info['total_gb']:.2f} GB")
    print()

    # ---- [1] Dataset Loading ----
    print("[1] Loading dataset...")
    t0 = time.time()
    from datasets.synapse_dataset import get_train_loader, SynapseTestDataset, RandomGenerator

    train_loader, train_dataset = get_train_loader(cfg, augment=False)

    # Build test loader with num_workers=0 for this script
    test_dataset = SynapseTestDataset(
        data_dir=cfg["dataset"]["test_data_dir"],
        list_dir=cfg["dataset"]["list_dir"],
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"    Train slices: {len(train_dataset)}")
    print(f"    Test volumes: {len(test_dataset)}")
    print(f"    Loading time: {time.time()-t0:.2f}s")
    print()

    # ---- [2] Model ----
    print("[2] Building model...")
    from models.unet2d import build_unet2d, count_parameters
    model = build_unet2d(cfg).to(device)
    train_p, total_p = count_parameters(model)
    print(f"    Parameters: {train_p:.2f}M trainable / {total_p:.2f}M total")
    print()

    # ---- [3] Forward Pass ----
    print("[3] Testing forward pass...")
    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    print(f"    Input shape : {images.shape} dtype={images.dtype}")
    print(f"    Label shape : {labels.shape} dtype={labels.dtype}")
    print(f"    Label range : [{labels.min().item()}, {labels.max().item()}]")

    t0 = time.time()
    with torch.no_grad():
        out = model(images)
    fwd_time = time.time() - t0
    print(f"    Output shape: {out.shape}")
    print(f"    Forward time: {fwd_time*1000:.1f}ms")

    if torch.cuda.is_available():
        from utils.gpu_monitor import get_gpu_info
        info = get_gpu_info()
        print(f"    VRAM used   : {info['alloc_gb']:.2f} GB / {info['total_gb']:.2f} GB")
    print()

    # ---- [4] Loss ----
    print("[4] Testing loss computation...")
    from losses.segmentation_losses import build_criterion
    criterion = build_criterion(cfg)
    with torch.no_grad():
        loss, ce_l, dice_l = criterion(out, labels)
    print(f"    Total loss: {loss.item():.4f}")
    print(f"    CE loss   : {ce_l.item():.4f}")
    print(f"    Dice loss : {dice_l.item():.4f}")
    print()

    # ---- [5] Backward Pass ----
    print("[5] Testing backward pass + AMP...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    model.train()
    optimizer.zero_grad()
    if scaler:
        with torch.amp.autocast(device_type="cuda"):
            out2 = model(images)
            loss2, _, _ = criterion(out2, labels)
        scaler.scale(loss2).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        out2 = model(images)
        loss2, _, _ = criterion(out2, labels)
        loss2.backward()
        optimizer.step()

    if torch.cuda.is_available():
        from utils.gpu_monitor import get_gpu_info
        info = get_gpu_info()
        print(f"    VRAM after backward: {info['alloc_gb']:.2f} GB")
    print(f"    Backward pass: OK (loss={loss2.item():.4f})")
    print()

    # ---- [6] Metrics ----
    print("[6] Testing metrics...")
    from utils.metrics import dice_coef_from_logits
    model.eval()
    with torch.no_grad():
        out3 = model(images)
        dice = dice_coef_from_logits(out3, labels, cfg["dataset"]["num_classes"])
    print(f"    Batch Dice Score: {dice:.4f}")
    print()

    # ---- [7] Volumetric Evaluation ----
    print("[7] Testing volumetric evaluation (1 volume)...")
    from utils.metrics import test_single_volume
    batch = next(iter(test_loader))
    vol_image = batch["image"]
    vol_label = batch["label"]
    case_name = batch["case_name"][0]
    print(f"    Volume: {case_name}, shape: {tuple(vol_image.squeeze().shape)}")

    t0 = time.time()
    metrics = test_single_volume(
        vol_image, vol_label, model,
        classes=cfg["dataset"]["num_classes"],
        patch_size=(cfg["dataset"]["img_size"], cfg["dataset"]["img_size"]),
        device=str(device),
    )
    eval_time = time.time() - t0
    mean_dice = np.mean([m[0] for m in metrics])
    print(f"    Eval time   : {eval_time:.1f}s")
    print(f"    Mean Dice   : {mean_dice:.4f}  (random init — expected ~0.0)")
    print()

    print("=" * 60)
    print("ALL 7 TESTS PASSED! Pipeline is working correctly.")
    print("=" * 60)
    print()
    print("GPU Summary:")
    if torch.cuda.is_available():
        from utils.gpu_monitor import get_gpu_info
        i = get_gpu_info()
        print(f"  GPU         : {i['name']}")
        print(f"  Max VRAM    : {i['max_alloc_gb']:.2f} GB / {i['total_gb']:.2f} GB")
    print()
    print("Ready to train! Recommended commands:")
    print()
    print("  Quick test (5 epochs):")
    print("    venv/Scripts/python.exe train.py --epochs 5")
    print()
    print("  Main experiment (20 epochs):")
    print("    venv/Scripts/python.exe train.py --epochs 20")
    print()
    print("  Extended run (50 epochs):")
    print("    venv/Scripts/python.exe train.py --epochs 50")


if __name__ == "__main__":
    run_pipeline_test()
