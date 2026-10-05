"""
inference.py
============
Standalone script to perform inference on a few random test slices.
Useful for quickly checking if the model can segment random data.

Usage:
    venv/Scripts/python.exe scripts/inference.py --model transunet --checkpoint experiments/transunet/checkpoints/best_model.pth --num_slices 5
"""

import argparse
import os
import random
import yaml
import torch
import numpy as np
import sys
from scipy.ndimage import zoom as spzoom

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.synapse_dataset import SynapseTrainDataset, get_test_loader
from models.unet2d import build_unet2d
from models.unet_2_5d import build_unet_2_5d
from models.transunet import build_transunet
from utils.visualization import save_prediction_panel


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on random slices")
    parser.add_argument("--model", type=str, choices=["unet", "transunet", "unet_2_5d"], required=True, help="Model architecture to use")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--num_slices", type=int, default=5, help="Number of random slices to visualize")
    parser.add_argument("--save_dir", type=str, default="inference_results", help="Directory to save the visualizations")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Select config based on model type
    if args.model == "transunet":
        config_file = "configs/transunet.yaml"
    elif args.model == "unet_2_5d":
        config_file = "configs/unet_2_5d_config.yaml"
    else:
        config_file = "configs/unet2d_config.yaml"
        
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Build model
    if args.model == "transunet":
        model = build_transunet(cfg).to(device)
    elif args.model == "unet_2_5d":
        model = build_unet_2_5d(cfg).to(device)
    else:
        model = build_unet2d(cfg).to(device)
        
    # 2. Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
        
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (Best Dice: {ckpt.get('best_dice', 0):.4f})")
    
    # 3. Load dataset
    # We will just load random slices directly from the training npz directory for a quick 2D test,
    # or you could extract slices from the test volumes. Here we use train_npz for pure 2D slice inference.
    data_dir = cfg["dataset"]["train_data_dir"]
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    random.shuffle(all_files)
    selected_files = all_files[:args.num_slices]
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nRunning inference on {len(selected_files)} random slices...")
    
    with torch.no_grad():
        for i, file_name in enumerate(selected_files):
            file_path = os.path.join(data_dir, file_name)
            
            if args.model == "unet_2_5d":
                case_name_full, slice_str = file_name.replace('.npz', '').split('_slice')
                z = int(slice_str)
                prev_sn = f"{case_name_full}_slice{z-1:03d}.npz"
                next_sn = f"{case_name_full}_slice{z+1:03d}.npz"
                
                path_prev = os.path.join(data_dir, prev_sn)
                path_z = file_path
                path_next = os.path.join(data_dir, next_sn)
                
                if not os.path.exists(path_prev): path_prev = path_z
                if not os.path.exists(path_next): path_next = path_z
                
                d_prev = np.load(path_prev)
                d_z = np.load(path_z)
                d_next = np.load(path_next)
                
                img_prev, img_z, img_next = d_prev["image"], d_z["image"], d_next["image"]
                label = d_z["label"]
                
                img_h, img_w = img_z.shape
                target_size = cfg["dataset"]["img_size"]
                
                if img_h != target_size or img_w != target_size:
                    img_stack = np.stack([img_prev, img_z, img_next], axis=0)
                    image_resized = spzoom(img_stack, (1.0, target_size / img_h, target_size / img_w), order=3)
                    label_resized = spzoom(label, (target_size / img_h, target_size / img_w), order=0).astype(np.int64)
                    image_vis = image_resized[1] # middle slice for visualization
                else:
                    image_resized = np.stack([img_prev, img_z, img_next], axis=0)
                    label_resized = label.astype(np.int64)
                    image_vis = image_resized[1]
                    
                img_tensor = torch.from_numpy(image_resized).float().unsqueeze(0).to(device)
            else:
                data = np.load(file_path)
                image, label = data["image"], data["label"]
                
                img_h, img_w = image.shape
                target_size = cfg["dataset"]["img_size"]  # 224
                
                # Resize to model input size (same as training preprocessing)
                if img_h != target_size or img_w != target_size:
                    image_resized = spzoom(image, (target_size / img_h, target_size / img_w), order=3)
                    label_resized = spzoom(label, (target_size / img_h, target_size / img_w), order=0).astype(np.int64)
                else:
                    image_resized, label_resized = image, label.astype(np.int64)
                
                # Convert to tensor: [B, C, H, W]
                img_tensor = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0).to(device)
                image_vis = image_resized
            
            # Predict
            with torch.amp.autocast(device_type="cuda"):
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
            
            # Visualization
            case_name = file_name.replace('.npz', '')
            save_path = save_prediction_panel(
                image=image_vis, 
                label=label_resized, 
                pred=pred, 
                case_name=f"inference_{args.model}_{case_name}", 
                save_dir=args.save_dir
            )
            print(f"[{i+1}/{len(selected_files)}] Saved visualization to {save_path}")
            
    print(f"\nAll Done! Visualizations are available in the '{args.save_dir}' folder.")


if __name__ == "__main__":
    main()
