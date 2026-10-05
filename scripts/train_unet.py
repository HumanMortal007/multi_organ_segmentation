"""
train.py
=========
Main training script for 2D U-Net multi-organ segmentation on Synapse/BTCV.

Usage:
    # Quick pipeline test (5 epochs):
    python train_unet.py --epochs 5

    # Main experiment (20 epochs):
    python train_unet.py --epochs 20

    # Extended run (50 epochs):
    python train_unet.py --epochs 50

    # Resume from checkpoint:
    python train_unet.py --resume checkpoints/last_checkpoint.pth --epochs 20

    # Disable augmentation:
    python train_unet.py --no_aug --epochs 5

    # Use venv (Windows):
    # venv/Scripts/python.exe train.py --epochs 20

"""

import argparse
import os
import sys
import time
import random
import logging
import json
import math

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import yaml
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project imports
from datasets.synapse_dataset import get_train_loader, get_test_loader, SynapseTrainDataset, RandomGenerator
from models.unet2d import build_unet2d, count_parameters
from losses.segmentation_losses import build_criterion
from utils.metrics import dice_coef_from_logits, test_single_volume
from utils.timing import Timer
from utils.checkpoint import save_checkpoint, load_checkpoint, make_checkpoint
from utils.visualization import save_training_curves, save_prediction_panel, save_eval_results_table
from utils.gpu_monitor import get_gpu_info, print_gpu_info, reset_peak_memory
from torch.utils.data import DataLoader, random_split


# ============================================================
# Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="2D U-Net Multi-Organ Segmentation - Synapse/BTCV")
    parser.add_argument("--config",     default="configs/unet2d_config.yaml", help="Config file path")
    parser.add_argument("--epochs",     type=int,   default=None,  help="Override num_epochs from config")
    parser.add_argument("--batch_size", type=int,   default=None,  help="Override batch_size from config")
    parser.add_argument("--lr",         type=float, default=None,  help="Override learning rate")
    parser.add_argument("--resume",     type=str,   default=None,  help="Path to checkpoint to resume from")
    parser.add_argument("--no_aug",     action="store_true",       help="Disable augmentation")
    parser.add_argument("--no_amp",     action="store_true",       help="Disable AMP (for debugging)")
    parser.add_argument("--seed",       type=int,   default=1234,  help="Random seed")
    parser.add_argument("--val_freq",   type=int,   default=1,     help="Validate every N epochs")
    parser.add_argument("--skip_eval",  action="store_true",       help="Skip final test evaluation")
    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Logging
# ============================================================

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    fmt = "%(asctime)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("unet2d")


# ============================================================
# Training epoch
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, num_classes):
    model.train()
    total_loss = total_dice = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)  # [B, 1, H, W]
        labels = batch["label"].to(device, non_blocking=True)  # [B, H, W]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss, _, _ = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, _, _ = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        dice = dice_coef_from_logits(logits.detach(), labels.detach(), num_classes)
        total_loss += loss.item()
        total_dice += dice
        n_batches  += 1

    return total_loss / n_batches, total_dice / n_batches


# ============================================================
# Validation epoch
# ============================================================

@torch.no_grad()
def validate_epoch(model, loader, criterion, device, use_amp, num_classes):
    model.eval()
    total_loss = total_dice = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Validation", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss, _, _ = criterion(logits, labels)
        else:
            logits = model(images)
            loss, _, _ = criterion(logits, labels)

        dice = dice_coef_from_logits(logits, labels, num_classes)
        total_loss += loss.item()
        total_dice += dice
        n_batches  += 1

    return total_loss / n_batches, total_dice / n_batches


# ============================================================
# Test set volumetric evaluation
# ============================================================

def evaluate_test_set(model, test_loader, cfg, device, logger, vis_dir, n_vis=3):
    """Slice-by-slice 3D evaluation matching TransUNet evaluation pipeline."""
    from scipy.ndimage import zoom as spzoom

    num_classes = cfg["dataset"]["num_classes"]
    patch_size  = cfg["evaluation"]["patch_size"]

    model.eval()
    metric_list = np.zeros((num_classes - 1, 2))
    n_cases = 0

    for i_batch, batch in enumerate(test_loader):
        image     = batch["image"]          # [1, D, H, W]
        label     = batch["label"]          # [1, D, H, W]
        case_name = batch["case_name"][0]

        t_start = time.time()
        metrics = test_single_volume(
            image, label, model,
            classes=num_classes,
            patch_size=(patch_size, patch_size),
            device=str(device),
        )
        elapsed = time.time() - t_start

        metrics_arr = np.array(metrics)
        metric_list += metrics_arr
        n_cases += 1
        per_case_dice = metrics_arr[:, 0].mean()
        logger.info(f"  [{i_batch+1}/{len(test_loader)}] {case_name}: dice={per_case_dice:.4f} ({elapsed:.1f}s)")

        # Visualize a middle slice
        if i_batch < n_vis:
            img_vol = image.squeeze(0).numpy()
            lbl_vol = label.squeeze(0).numpy()
            mid     = img_vol.shape[0] // 2
            sl_img  = img_vol[mid]
            sl_lbl  = lbl_vol[mid].astype(np.int64)
            x, y    = sl_img.shape
            sl_r    = spzoom(sl_img, (patch_size / x, patch_size / y), order=3) if x != patch_size else sl_img
            inp     = torch.from_numpy(sl_r.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = torch.argmax(torch.softmax(model(inp), dim=1), dim=1).squeeze().cpu().numpy()
            if x != patch_size:
                out = spzoom(out, (x / patch_size, y / patch_size), order=0).astype(np.int64)
            save_prediction_panel(sl_img, sl_lbl, out, case_name, vis_dir, slice_idx=mid)

    metric_list /= max(n_cases, 1)
    return metric_list[:, 0].tolist(), metric_list[:, 1].tolist()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Load config
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs     is not None: cfg["training"]["num_epochs"]   = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"]   = args.batch_size
    if args.lr         is not None: cfg["optimizer"]["lr"]          = args.lr
    if args.no_aug:                  cfg["augmentation"]["enabled"]  = False
    if args.no_amp:                  cfg["amp"]["enabled"]           = False

    # Seed
    set_seed(args.seed)

    # Logging
    logger = setup_logging(cfg["results"]["logs_dir"])
    logger.info("=" * 70)
    logger.info("  2D U-Net Multi-Organ Segmentation  |  Synapse/BTCV")
    logger.info("=" * 70)
    logger.info(f"Config      : {args.config}")
    logger.info(f"Epochs      : {cfg['training']['num_epochs']}")
    logger.info(f"Batch size  : {cfg['training']['batch_size']}")
    logger.info(f"Learning rate: {cfg['optimizer']['lr']}")
    logger.info(f"AMP         : {cfg['amp']['enabled']}")
    logger.info(f"Augmentation: {cfg['augmentation']['enabled']}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device      : {device}")
    if device.type == "cuda":
        gpu = get_gpu_info()
        logger.info(f"GPU         : {gpu['name']}")
        logger.info(f"Total VRAM  : {gpu['total_gb']:.2f} GB")

    # Create directories
    for d in [cfg["results"]["plots_dir"], cfg["results"]["vis_dir"],
              cfg["results"]["logs_dir"], cfg["checkpointing"]["dir"]]:
        os.makedirs(d, exist_ok=True)

    # ---- Dataset ----
    logger.info("\n[1] Loading dataset...")
    t_ds = time.time()

    # Build full train dataset
    img_size  = cfg["dataset"]["img_size"]
    augment   = not args.no_aug and cfg["augmentation"]["enabled"]
    transform_train = RandomGenerator(
        output_size=(img_size, img_size),
        augment=augment,
        do_rot_flip=cfg["augmentation"]["random_rot_flip"],
        do_rotate=cfg["augmentation"]["random_rotate"],
    )
    transform_val = RandomGenerator(
        output_size=(img_size, img_size),
        augment=False,
    )

    full_dataset = SynapseTrainDataset(
        data_dir=cfg["dataset"]["train_data_dir"],
        list_dir=cfg["dataset"]["list_dir"],
        transform=None,  # Will apply per-loader below
    )
    n_total = len(full_dataset)
    n_val   = max(1, int(0.1 * n_total))   # 10% for validation
    n_train = n_total - n_val
    logger.info(f"  Total slices : {n_total}")
    logger.info(f"  Train slices : {n_train}  |  Val slices: {n_val}")

    # Use index-based split
    indices = list(range(n_total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # Build loaders with correct transforms
    full_dataset.transform = transform_train
    train_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, train_idx),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        persistent_workers=(cfg["training"]["num_workers"] > 0 and cfg["training"]["persistent_workers"]),
        drop_last=True,
    )

    # Val loader (same dataset object but with val transform applied via wrapper)
    val_dataset = SynapseTrainDataset(
        data_dir=cfg["dataset"]["train_data_dir"],
        list_dir=cfg["dataset"]["list_dir"],
        transform=transform_val,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, val_idx),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )

    test_loader, test_dataset = get_test_loader(cfg)
    ds_time = time.time() - t_ds
    logger.info(f"  Test volumes : {len(test_dataset)}")
    logger.info(f"  Dataset load time: {ds_time:.1f}s")

    # ---- Model ----
    logger.info("\n[2] Building model...")
    model = build_unet2d(cfg).to(device)
    train_p, total_p = count_parameters(model)
    logger.info(f"  UNet2D — {train_p:.2f}M trainable params ({total_p:.2f}M total)")

    # ---- Loss ----
    criterion = build_criterion(cfg)
    num_classes = cfg["dataset"]["num_classes"]

    # ---- Optimizer & Scheduler ----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    num_epochs = cfg["training"]["num_epochs"]
    sch_name   = cfg["scheduler"]["name"]
    if sch_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=cfg["scheduler"]["eta_min"])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5, verbose=True)

    # ---- AMP ----
    use_amp = cfg["amp"]["enabled"] and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("  AMP: enabled (torch.amp.GradScaler)")

    # ---- Resume ----
    start_epoch = 0
    best_dice   = 0.0
    if args.resume:
        start_epoch, best_dice, _ = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    # ---- Training history ----
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    # ---- Timer ----
    timer = Timer(total_epochs=num_epochs)
    timer.start_experiment()
    reset_peak_memory()

    logger.info(f"\n[3] Training — {num_epochs} epochs | {len(train_loader)} batches/epoch")
    logger.info("=" * 70)

    for epoch in range(start_epoch, num_epochs):
        timer.start_epoch()

        # --- Train ---
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, num_classes
        )

        # --- Validate ---
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_dice = validate_epoch(
                model, val_loader, criterion, device, use_amp, num_classes
            )
        else:
            val_loss = val_dice = float("nan")

        # --- Scheduler step ---
        if isinstance(scheduler, ReduceLROnPlateau):
            if not math.isnan(val_dice):
                scheduler.step(val_dice)
        else:
            scheduler.step()

        epoch_time = timer.end_epoch()
        cur_lr     = optimizer.param_groups[0]["lr"]

        # GPU memory
        gpu_info = get_gpu_info() if device.type == "cuda" else {}
        vram_str = f"{gpu_info.get('alloc_gb', 0):.2f} / {gpu_info.get('total_gb', 8):.2f} GB"

        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss if not math.isnan(val_loss) else None)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice if not math.isnan(val_dice) else None)

        # ---- Console output ----
        logger.info(f"\nEpoch [{epoch+1:>3}/{num_epochs}]")
        logger.info(f"  Train Loss  : {train_loss:.4f}  |  Val Loss  : {val_loss:.4f}")
        logger.info(f"  Train Dice  : {train_dice:.4f}  |  Val Dice  : {val_dice:.4f}")
        logger.info(f"  LR          : {cur_lr:.3e}")
        logger.info(f"  Epoch Time  : {timer.format_time(epoch_time)}")
        if len(timer.epoch_times) >= 2:
            avg_t = timer.avg_epoch_time()
            rem_t = timer.estimated_remaining(epoch + 1)
            logger.info(f"  Avg/Epoch   : {timer.format_time(avg_t)}")
            logger.info(f"  Est. Remain : {timer.format_time(rem_t)}")
        logger.info(f"  GPU VRAM    : {vram_str}")

        # After epoch 3, print full time estimates
        if epoch == 2:
            logger.info("\n  === TIME ESTIMATES ===")
            avg = timer.avg_epoch_time()
            logger.info(f"  Avg epoch time  : {timer.format_time(avg)}")
            logger.info(f"  Est. 20 epochs  : {timer.format_time(avg * 20)}")
            logger.info(f"  Est. 50 epochs  : {timer.format_time(avg * 50)}")
            logger.info("  ========================")

        # ---- Checkpointing ----
        ckpt = make_checkpoint(model, optimizer, scheduler, epoch, best_dice, cfg)
        save_checkpoint(ckpt, os.path.join(cfg["checkpointing"]["dir"], "last_checkpoint.pth"))

        if not math.isnan(val_dice) and val_dice > best_dice:
            best_dice = val_dice
            ckpt["best_dice"] = best_dice
            save_checkpoint(ckpt, os.path.join(cfg["checkpointing"]["dir"], "best_model.pth"))
            logger.info(f"  *** New best model! Val Dice = {best_dice:.4f} ***")

    # ---- Post-training ----
    # Clean history of None values for JSON
    for k in history:
        history[k] = [v if v is not None else 0.0 for v in history[k]]

    save_training_curves(history, cfg["results"]["plots_dir"])
    with open(os.path.join(cfg["results"]["logs_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    total_train_time = timer.total_elapsed()
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total Training Time  : {timer.format_time(total_train_time)}")
    logger.info(f"Best Validation Dice : {best_dice:.4f}")
    if device.type == "cuda":
        gpu = get_gpu_info()
        logger.info(f"Max VRAM Used        : {gpu['max_alloc_gb']:.2f} GB / {gpu['total_gb']:.2f} GB")

    if args.skip_eval:
        logger.info("Skipping final evaluation (--skip_eval flag set).")
        return

    # ---- Final evaluation on test volumes ----
    logger.info("\n[4] Final Evaluation on Test Volumes...")
    best_ckpt = os.path.join(cfg["checkpointing"]["dir"], "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"  Loaded best checkpoint (epoch {ckpt.get('epoch','?')+1})")

    t_eval = time.time()
    per_class_dice, per_class_hd95 = evaluate_test_set(
        model, test_loader, cfg, device, logger,
        vis_dir=cfg["results"]["vis_dir"]
    )
    eval_time = time.time() - t_eval

    class_names = cfg["dataset"]["class_names"]
    mean_dice, mean_hd = save_eval_results_table(
        per_class_dice, per_class_hd95, class_names, cfg["results"]["logs_dir"]
    )

    logger.info(f"\nEvaluation Time     : {timer.format_time(eval_time)}")
    logger.info(f"Test Mean Dice      : {mean_dice:.4f}")
    logger.info(f"Total Experiment    : {timer.format_time(timer.total_elapsed())}")

    # Save experiment summary
    summary = {
        "best_val_dice"       : best_dice,
        "test_mean_dice"      : mean_dice,
        "test_per_class_dice" : {c: d for c, d in zip(class_names[1:], per_class_dice)},
        "total_train_time_s"  : total_train_time,
        "eval_time_s"         : eval_time,
        "total_experiment_s"  : timer.total_elapsed(),
    }
    if device.type == "cuda":
        summary["max_vram_gb"] = get_gpu_info()["max_alloc_gb"]

    summary_path = os.path.join(cfg["results"]["logs_dir"], "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
