"""
tests/test_transunet_2_5d.py
==============================
Pre-training GPU memory and correctness test for 2.5D TransUNet.

Checks:
  1. Model builds correctly with in_channels=3.
  2. Forward pass: [B, 3, 224, 224] -> [B, 9, 224, 224].
  3. Loss computes without NaN.
  4. Backward pass completes.
  5. Optimizer step runs.
  6. GPU peak VRAM is within safe limits for RTX 4060 (8GB).
  7. Checkpoint can be saved and reloaded.

Usage:
    venv/Scripts/python.exe tests/test_transunet_2_5d.py
"""

import os
import sys
import tempfile
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transunet_2_5d import build_transunet_2_5d
from losses.segmentation_losses import build_criterion
from utils.checkpoint import save_checkpoint, load_checkpoint, make_checkpoint

CONFIG_PATH = "configs/transunet_2_5d_config.yaml"
PASS = "  [PASS]"
FAIL = "  [FAIL]"


def sep(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


def run_tests():
    sep("2.5D TransUNet — Pre-Training Test")

    # ----------------------------------------------------------------
    # Load config
    # ----------------------------------------------------------------
    print(f"\n[1] Loading config: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    B = cfg["training"]["batch_size"]
    H = W = cfg["dataset"]["img_size"]
    C_in = cfg["model"]["in_channels"]
    C_out = cfg["model"]["num_classes"]
    print(f"    Batch size  : {B}")
    print(f"    Input shape : [{B}, {C_in}, {H}, {W}]")
    print(f"    Output shape: [{B}, {C_out}, {H}, {W}]")
    assert C_in == 3, f"Expected in_channels=3, got {C_in}"
    print(f"{PASS} Config loaded. in_channels=3 confirmed.")

    # ----------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2] Device: {device}")
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    GPU     : {torch.cuda.get_device_name(0)}")
        print(f"    Total VRAM: {total_vram:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    else:
        print("    WARNING: CUDA not available. Running on CPU (slow).")

    # ----------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------
    print(f"\n[3] Building 2.5D TransUNet...")
    model = build_transunet_2_5d(cfg).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"    Trainable params : {trainable/1e6:.3f}M")
    print(f"    Total params     : {total_p/1e6:.3f}M")
    print(f"{PASS} Model built successfully.")

    # ----------------------------------------------------------------
    # Synthetic batch (mimics 2.5D loader output)
    # ----------------------------------------------------------------
    print(f"\n[4] Creating synthetic 2.5D batch [{B}, 3, {H}, {W}]...")
    images = torch.randn(B, 3, H, W).to(device)   # [B, 3, H, W]
    labels = torch.randint(0, C_out, (B, H, W)).to(device)  # [B, H, W]
    print(f"    images shape: {tuple(images.shape)}")
    print(f"    labels shape: {tuple(labels.shape)}")

    # ----------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------
    print(f"\n[5] Forward pass (AMP)...")
    criterion = build_criterion(cfg)
    use_amp = cfg["amp"]["enabled"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"])
    optimizer.zero_grad(set_to_none=True)

    try:
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss, ce_loss, dice_loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss, ce_loss, dice_loss = criterion(logits, labels)

        assert logits.shape == (B, C_out, H, W), \
            f"Expected [{B},{C_out},{H},{W}], got {tuple(logits.shape)}"
        assert not torch.isnan(loss), "Loss is NaN!"
        print(f"    Output shape : {tuple(logits.shape)}")
        print(f"    Total Loss   : {loss.item():.4f}  (CE={ce_loss.item():.4f} | Dice={dice_loss.item():.4f})")
        print(f"{PASS} Forward pass OK.")
    except Exception as e:
        print(f"{FAIL} Forward pass FAILED: {e}")
        return False

    # ----------------------------------------------------------------
    # Backward pass
    # ----------------------------------------------------------------
    print(f"\n[6] Backward pass...")
    try:
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"{PASS} Backward pass + optimizer step OK.")
    except Exception as e:
        print(f"{FAIL} Backward pass FAILED: {e}")
        return False

    # ----------------------------------------------------------------
    # VRAM report
    # ----------------------------------------------------------------
    if device.type == "cuda":
        alloc  = torch.cuda.memory_allocated(0) / 1e9
        peak   = torch.cuda.max_memory_allocated(0) / 1e9
        total  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[7] GPU Memory Report:")
        print(f"    Current allocated : {alloc:.3f} GB")
        print(f"    Peak allocated    : {peak:.3f} GB")
        print(f"    Total available   : {total:.2f} GB")
        print(f"    Headroom          : {total - peak:.2f} GB")
        if peak < total * 0.90:
            print(f"{PASS} VRAM usage is safe ({peak:.2f}/{total:.2f} GB = {100*peak/total:.0f}%)")
        else:
            print(f"  [WARN] VRAM usage is high ({peak:.2f}/{total:.2f} GB). Consider reducing batch_size.")

    # ----------------------------------------------------------------
    # Checkpoint save + reload
    # ----------------------------------------------------------------
    print(f"\n[8] Checkpoint save & reload test...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
            ckpt = make_checkpoint(model, optimizer, None, epoch=0, best_dice=0.0, cfg=cfg)
            save_checkpoint(ckpt, ckpt_path)

            # Fresh model, reload
            model2 = build_transunet_2_5d(cfg).to(device)
            epoch_r, dice_r, _ = load_checkpoint(ckpt_path, model2, None, None, device)
            print(f"    Reloaded epoch={epoch_r}, best_dice={dice_r:.4f}")
        print(f"{PASS} Checkpoint save/reload OK.")
    except Exception as e:
        print(f"{FAIL} Checkpoint test FAILED: {e}")
        return False

    # ----------------------------------------------------------------
    # Mini training loop (5 steps)
    # ----------------------------------------------------------------
    print(f"\n[9] Mini training loop (5 batches)...")
    model.train()
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"])
    scaler2 = torch.amp.GradScaler("cuda") if use_amp else None
    for step in range(5):
        imgs = torch.randn(B, 3, H, W).to(device)
        lbls = torch.randint(0, C_out, (B, H, W)).to(device)
        optimizer2.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                out = model(imgs)
                l, _, _ = criterion(out, lbls)
            scaler2.scale(l).backward()
            scaler2.unscale_(optimizer2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler2.step(optimizer2)
            scaler2.update()
        else:
            out = model(imgs)
            l, _, _ = criterion(out, lbls)
            l.backward()
            optimizer2.step()
        print(f"    Step {step+1}/5 — loss={l.item():.4f}")
    print(f"{PASS} Mini training loop OK.")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    sep("ALL TESTS PASSED")
    print("  2.5D TransUNet is ready for full training.")
    print(f"  Run: venv/Scripts/python.exe scripts/train_transunet_2_5d.py")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

