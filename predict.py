"""
predict.py
==========
Quick script to run inference on random slices from the test set and visualize the results.

Usage:
    venv/Scripts/python.exe predict.py --num_samples 5
"""

import os
import argparse
import random
import yaml
import torch
import numpy as np
import h5py
from matplotlib import pyplot as plt

from models.unet2d import build_unet2d
from utils.visualization import save_prediction_panel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet2d_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random slices to test")
    parser.add_argument("--out_dir", default="results/random_predictions")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load config and device
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load model
    print(f"Loading model from {args.checkpoint}...")
    model = build_unet2d(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded! (Best Val Dice: {ckpt.get('best_dice', 'N/A'):.4f})")

    # 3. Find test volumes
    test_dir = cfg["dataset"]["test_data_dir"]
    test_list = os.path.join(cfg["dataset"]["list_dir"], "test_vol.txt")
    
    with open(test_list, "r") as f:
        case_names = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(case_names)} test volumes.")

    # 4. Generate random samples
    patch_size = cfg["evaluation"]["patch_size"]
    
    for i in range(args.num_samples):
        # Pick a random patient and a random slice
        case_name = random.choice(case_names)
        vol_path = os.path.join(test_dir, f"{case_name}.npy.h5")
        
        with h5py.File(vol_path, 'r') as f:
            image_vol = f['image'][:]
            label_vol = f['label'][:]
            
        num_slices = image_vol.shape[0]
        # Pick a slice from the middle 50% to ensure it actually contains organs (ends are usually empty)
        slice_idx = random.randint(int(num_slices * 0.25), int(num_slices * 0.75))
        
        img_slice = image_vol[slice_idx]
        lbl_slice = label_vol[slice_idx]
        
        # Resize if necessary (test volumes are 512x512, model expects 224x224)
        from scipy.ndimage import zoom
        x, y = img_slice.shape
        if x != patch_size or y != patch_size:
            img_resized = zoom(img_slice, (patch_size / x, patch_size / y), order=3)
        else:
            img_resized = img_slice
            
        # Run inference
        inp = torch.from_numpy(img_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze().cpu().numpy()
            
        # Resize prediction back to original size for comparison
        if x != patch_size or y != patch_size:
            pred_resized = zoom(pred, (x / patch_size, y / patch_size), order=0).astype(np.int64)
        else:
            pred_resized = pred
            
        # Save visualization
        save_prediction_panel(img_slice, lbl_slice, pred_resized, case_name, args.out_dir, slice_idx=slice_idx)
        print(f"Saved prediction for {case_name} (Slice {slice_idx}) to {args.out_dir}")

    print(f"\nDone! Check the '{args.out_dir}' folder for your images.")

if __name__ == "__main__":
    main()
