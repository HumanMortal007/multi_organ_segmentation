"""
overfit_test.py
===============
Sanity check: overfit a tiny dataset (16 samples) to ensure the model, loss, and optimizer
are functioning correctly and capable of memorization.
"""

import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from datasets.synapse_dataset import get_train_loader
from models.unet2d import build_unet2d
from losses.segmentation_losses import build_criterion
from utils.metrics import dice_coef_from_logits

def main():
    print("========================================")
    print("TINY OVERFITTING TEST (16 SAMPLES)")
    print("========================================")

    with open("configs/unet2d_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get train dataset and take exactly 16 samples
    loader, full_dataset = get_train_loader(cfg, augment=False)
    tiny_dataset = Subset(full_dataset, range(16))
    tiny_loader = DataLoader(tiny_dataset, batch_size=4, shuffle=True, num_workers=0)

    # 2. Build model, loss, optimizer
    model = build_unet2d(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # 3. Train for 50 epochs
    model.train()
    for epoch in range(1, 51):
        total_loss = 0
        total_dice = 0
        
        for batch in tiny_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss, _, _ = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss, _, _ = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            dice = dice_coef_from_logits(logits.detach(), labels.detach(), cfg["dataset"]["num_classes"])
            total_loss += loss.item()
            total_dice += dice

        avg_loss = total_loss / len(tiny_loader)
        avg_dice = total_dice / len(tiny_loader)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")

    print("========================================")
    if avg_dice > 0.90:
        print("OVERFIT SUCCESSFUL: Model successfully memorized the tiny dataset.")
    else:
        print("OVERFIT FAILED: Model could not memorize the data.")

if __name__ == "__main__":
    main()
