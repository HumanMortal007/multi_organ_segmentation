"""
test_overfit.py
===============
Tiny overfitting test for TransUNet.
Takes 4 samples from the dataset and trains for 50 epochs to see if loss goes down and Dice goes up.
"""

import os
import sys
import torch
import torch.optim as optim
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transunet import build_transunet
from datasets.synapse_dataset import SynapseTrainDataset, RandomGenerator
from losses.segmentation_losses import build_criterion
from utils.metrics import dice_coef_from_logits


def main():
    print("=" * 50)
    print("TINY OVERFIT TEST")
    print("=" * 50)
    
    with open("configs/transunet.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset (just first 4 samples)
    ds = SynapseTrainDataset(
        data_dir=cfg["dataset"]["train_data_dir"],
        list_dir=cfg["dataset"]["list_dir"],
        transform=RandomGenerator((224, 224), augment=False)
    )
    
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, list(range(4))), 
        batch_size=4, 
        shuffle=False
    )
    
    model = build_transunet(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    scaler = torch.amp.GradScaler("cuda")
    
    model.train()
    
    for epoch in range(50):
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss, _, _ = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            dice = dice_coef_from_logits(logits.detach(), labels.detach(), cfg["dataset"]["num_classes"])
            
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/50 | Loss: {loss.item():.4f} | Dice: {dice:.4f}")

if __name__ == "__main__":
    main()
