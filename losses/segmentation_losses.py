"""
losses/segmentation_losses.py
==============================
Segmentation loss functions matching TransUNet:
  - CrossEntropyLoss (no softmax before CE)
  - DiceLoss (softmax applied internally)
  - CombinedLoss = alpha * CE + beta * Dice

Usage:
    from losses.segmentation_losses import CombinedLoss
    criterion = CombinedLoss(num_classes=9, ce_weight=0.5, dice_weight=0.5)
    loss = criterion(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dice Loss  (matches TransUNet DiceLoss exactly)
# ============================================================

class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss.
    Applies softmax internally.
    Computes per-class Dice and averages.
    """

    def __init__(self, n_classes, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def _one_hot(self, target):
        """Convert integer label map [B, H, W] to one-hot [B, C, H, W]."""
        parts = [(target == i).unsqueeze(1) for i in range(self.n_classes)]
        return torch.cat(parts, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1.0 - (2.0 * intersect + self.smooth) / (y_sum + z_sum + self.smooth)

    def forward(self, inputs, target, softmax=True, weight=None):
        """
        Args:
            inputs : [B, C, H, W] raw logits
            target : [B, H, W] integer class labels
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target_oh = self._one_hot(target)

        if weight is None:
            weight = [1.0] * self.n_classes
        assert inputs.shape == target_oh.shape,             f"Shape mismatch: {inputs.shape} vs {target_oh.shape}"

        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target_oh[:, i]) * weight[i]
        return loss / self.n_classes


# ============================================================
# Combined Loss
# ============================================================

class CombinedLoss(nn.Module):
    """
    Total Loss = ce_weight * CrossEntropy + dice_weight * DiceLoss

    CrossEntropy : expects raw logits (no softmax).
    DiceLoss     : applies softmax internally.
    """

    def __init__(self, num_classes=9, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_loss   = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes=num_classes)
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        """
        Args:
            logits : [B, C, H, W]
            target : [B, H, W] integer labels
        Returns:
            (total_loss, ce_loss_value, dice_loss_value)
        """
        loss_ce   = self.ce_loss(logits, target)
        loss_dice = self.dice_loss(logits, target, softmax=True)
        total     = self.ce_weight * loss_ce + self.dice_weight * loss_dice
        return total, loss_ce, loss_dice


def build_criterion(cfg):
    """Build loss from config."""
    return CombinedLoss(
        num_classes=cfg["dataset"]["num_classes"],
        ce_weight=cfg["loss"]["ce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
    )
