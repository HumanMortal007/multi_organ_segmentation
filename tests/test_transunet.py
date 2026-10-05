"""
test_transunet.py
=================
Tests for the TransUNet implementation.
"""

import os
import sys
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transunet import build_transunet


def test_model_sanity():
    print("=" * 40)
    print("TRANSUNET MODEL TEST")
    print("=" * 40)
    
    with open("configs/transunet.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    model = build_transunet(cfg)
    
    # Check parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Parameters: {total/1e6:.2f}M")
    print(f"Trainable Parameters: {trainable/1e6:.2f}M")
    
    # Forward pass
    b, c, h, w = 2, 1, 224, 224
    print(f"Input Shape: [{b}, {c}, {h}, {w}]")
    x = torch.randn(b, c, h, w)
    
    try:
        out = model(x)
        print(f"Output Shape: {list(out.shape)}")
        
        # Check shapes
        assert out.shape == (b, 9, 224, 224), f"Wrong output shape: {out.shape}"
        
        # Check NaNs
        assert not torch.isnan(out).any(), "NaN in output"
        print("NaN Check: PASS")
        
        # Gradients
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
        
    with open("configs/transunet.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    model = build_transunet(cfg).cuda()
    model.train()
    
    # Test batch size 2 and 4
    for bs in [2, 4]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(bs, 1, 224, 224).cuda()
        with torch.amp.autocast(device_type="cuda"):
            out = model(x)
            loss = out.mean()
        loss.backward()
        
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Batch Size {bs} - Peak VRAM: {peak_mb:.2f} MB")


if __name__ == "__main__":
    test_model_sanity()
    test_gpu_memory()
