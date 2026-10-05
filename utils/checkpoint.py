"""
utils/checkpoint.py
====================
Save and load training checkpoints.

Checkpoints contain:
  - model state_dict
  - optimizer state_dict
  - scheduler state_dict
  - current epoch
  - best validation Dice
  - training configuration
"""

import os
import torch


def save_checkpoint(state, filepath):
    """Save checkpoint to filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cuda"):
    """
    Load checkpoint.
    Returns: (start_epoch, best_dice, cfg)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_dice   = ckpt.get("best_dice", 0.0)
    cfg         = ckpt.get("config", {})

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print(f"[Checkpoint] Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")
    return start_epoch, best_dice, cfg


def make_checkpoint(model, optimizer, scheduler, epoch, best_dice, cfg):
    """Build the checkpoint state dict."""
    return {
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else {},
        "best_dice"           : best_dice,
        "config"              : cfg,
    }
