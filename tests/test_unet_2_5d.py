"""
test_unet_2_5d.py
=================
Tests for the 2.5D U-Net implementation.
"""

import os
import sys
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_2_5d import build_unet_2_5d


def test_model_sanity():
    print("=" * 40)
    print("2.5D U-NET MODEL TEST")
    print("=" * 40)
    
    with open("configs/unet_2_5d_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    model = build_unet_2_5d(cfg)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Parameters: {total/1e6:.2f}M")
    print(f"Trainable Parameters: {trainable/1e6:.2f}M")
    
    b, c, h, w = 2, 3, 224, 224
    print(f"Input Shape: [{b}, {c}, {h}, {w}]")
    x = torch.randn(b, c, h, w)
    
    try:
        out = model(x)
        print(f"Output Shape: {list(out.shape)}")
        
        assert out.shape == (b, 9, 224, 224), f"Wrong output shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        print("NaN Check: PASS")
        
        loss = out.mean()
        loss.backward()
        print("Gradient Check: PASS")
        
        print("Forward Pass: PASS")
    except Exception as e:
        print(f"Forward Pass: FAIL - {e}")


def test_gpu_memory():
    print("=" * 40)
    print("GPU MEMORY TEST")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU test.")
        return
        
    with open("configs/unet_2_5d_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    model = build_unet_2_5d(cfg).cuda()
    model.train()
    
    for bs in [2, 4, 8]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(bs, 3, 224, 224).cuda()
        with torch.amp.autocast(device_type="cuda"):
            out = model(x)
            loss = out.mean()
        loss.backward()
        
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Batch Size {bs} - Peak VRAM: {peak_mb:.2f} MB")


if __name__ == "__main__":
    test_model_sanity()
    test_gpu_memory()
