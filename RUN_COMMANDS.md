# Project Execution Commands

This document contains all the correct commands for running the Multi-Organ Segmentation experiments on your laptop.

> **Important**: You must be inside the project folder and have your virtual environment activated before running these commands.

**Step 1: Open your terminal and navigate to the project directory:**
```bash
cd E:\pp2\project_TransUNet
```

**Step 2: Activate your virtual environment:**
```bash
# If using Command Prompt:
venv\Scripts\activate.bat

# If using PowerShell:
.\venv\Scripts\Activate.ps1
```

Once activated, you will see `(venv)` at the start of your terminal prompt. You can then use `python` instead of the full path.

---

## 📋 Quick Reference — All 4 Models

| Model | Train | Resume | Test | Evaluate | Demo |
|---|---|---|---|---|---|
| 2D Base U-Net | `train_unet.py` | `--resume ...base_unet/.../last_checkpoint.pth` | — | `evaluate.py --model unet` | `inference.py --model unet` |
| 2D TransUNet | `train_transunet.py` | `--resume ...transunet/.../last_checkpoint.pth` | `test_transunet.py` | `evaluate.py --model transunet` | `inference.py --model transunet` |
| 2.5D U-Net | `train_unet_2_5d.py` | `--resume ...unet_2_5d/.../last_checkpoint.pth` | `test_unet_2_5d.py` | `evaluate.py --model unet_2_5d` | `inference.py --model unet_2_5d` |
| 2.5D TransUNet | `train_transunet_2_5d.py` | `--resume ...transunet_2_5d/.../last_checkpoint.pth` | `test_transunet_2_5d.py` | `evaluate.py --model transunet_2_5d` | *(see section 4)* |

---

## 1. 2D Base U-Net

### Train
```bash
python scripts/train_unet.py --epochs 20
```

### Resume (if interrupted)
```bash
python scripts/train_unet.py --epochs 20 --resume experiments/base_unet/checkpoints/last_checkpoint.pth
```

### Evaluate
```bash
venv/Scripts/python.exe scripts/evaluate.py --model unet --checkpoint experiments/base_unet/checkpoints/best_model.pth
```

### Random Inference Demo (5 random slices)
```bash
python scripts/inference.py --model unet --checkpoint experiments/base_unet/checkpoints/best_model.pth --num_slices 5
```

---

## 2. 2D TransUNet

### Pre-Training GPU Tests
```bash
# Sanity and GPU memory test
python tests/test_transunet.py

# Tiny overfitting test (runs 50 epochs on 4 samples very quickly)
python tests/test_overfit.py
```

### Train
```bash
python scripts/train_transunet.py --epochs 20
```

### Resume (if interrupted)
```bash
python scripts/train_transunet.py --epochs 20 --resume experiments/transunet/checkpoints/last_checkpoint.pth
```

### Evaluate
```bash
# Basic evaluation
venv/Scripts/python.exe scripts/evaluate.py --model transunet --checkpoint experiments/transunet/checkpoints/best_model.pth

# With visualizations (saves CT / GT / Pred / Overlay panels)
# --save_vis       : enable visualization output
# --n_vis 5        : how many test volumes to visualize
# --foreground_slices : pick the slice with the most organ pixels (better visuals)
venv/Scripts/python.exe scripts/evaluate.py --model transunet --checkpoint experiments/transunet/checkpoints/best_model.pth --save_vis --n_vis 5 --foreground_slices
```

### Random Inference Demo (5 random slices)
```bash
python scripts/inference.py --model transunet --checkpoint experiments/transunet/checkpoints/best_model.pth --num_slices 5
```

---

## 3. 2.5D U-Net

### Pre-Training GPU Tests
```bash
python tests/test_unet_2_5d.py
python tests/test_overfit_2_5d.py
```

### Train
```bash
python scripts/train_unet_2_5d.py --epochs 20
```

### Resume (if interrupted)
```bash
python scripts/train_unet_2_5d.py --epochs 20 --resume experiments/unet_2_5d/checkpoints/last_checkpoint.pth
```

### Evaluate
```bash
venv/Scripts/python.exe scripts/evaluate.py --model unet_2_5d --checkpoint experiments/unet_2_5d/checkpoints/best_model.pth
```

### Random Inference Demo (5 random slices)
```bash
python scripts/inference.py --model unet_2_5d --checkpoint experiments/unet_2_5d/checkpoints/best_model.pth --num_slices 5
```

---

## 4. 2.5D TransUNet *(New — Fourth Model)*

This model combines the Transformer architecture with 2.5D adjacent-slice input (`[z-1, z, z+1]`).

### Pre-Training GPU Test
```bash
# Runs 9 checks: forward pass, backward pass, VRAM report, checkpoint save/reload,
# and a 5-step mini training loop.
# Expected peak VRAM: ~0.80 GB at batch=8 (confirmed safe on RTX 4060).
venv/Scripts/python.exe tests/test_transunet_2_5d.py
```

### Train
```bash
venv/Scripts/python.exe scripts/train_transunet_2_5d.py
```

### Resume (if interrupted)
```bash
venv/Scripts/python.exe scripts/train_transunet_2_5d.py --resume experiments/transunet_2_5d/checkpoints/last_checkpoint.pth
```

### Evaluate
```bash
# Basic evaluation (Macro Dice + Weighted Dice + HD95 + per-organ table)
venv/Scripts/python.exe scripts/evaluate.py --model transunet_2_5d --checkpoint experiments/transunet_2_5d/checkpoints/best_model.pth

# With visualizations
venv/Scripts/python.exe scripts/evaluate.py --model transunet_2_5d --checkpoint experiments/transunet_2_5d/checkpoints/best_model.pth --save_vis --n_vis 5 --foreground_slices
```

> **How many times does it run?**
> Each command runs the evaluation **exactly once** on all **12 test volumes**.
> You can run the same command as many times as you want — it **never retrains** the model.
> Each run saves results to a **new timestamped folder** so previous results are never overwritten.

> Results saved to: `experiments/transunet_2_5d/results/evaluation/<timestamp>/`
> Checkpoints saved to: `experiments/transunet_2_5d/checkpoints/`

---

## 5. 🖼️ Dataset Inspection Demo

Loads any `.npz` training file and opens a popup window showing the raw CT image, ground truth mask, and overlay. Useful for demonstrating data to your guide or panel.

```bash
# Visualize a specific training slice
venv/Scripts/python.exe scripts/demo_npz.py --file data/Synapse/train_npz/case0005_slice050.npz

# Try another patient / slice
venv/Scripts/python.exe scripts/demo_npz.py --file data/Synapse/train_npz/case0001_slice030.npz
```

---

## 💡 Notes on Hardware & Execution
* **Progress Bar:** `tqdm` progress bars have been added, so when you run the training commands, you will now see real-time progress for both the training and validation epochs.
* **AMP (Mixed Precision):** Is enabled by default to save VRAM and increase speed.
* **VRAM Usage:** The batch size for all models is set to `8`, and peak VRAM is tightly managed (~0.8–1.5 GB across all models), so you should never hit CUDA OOM on your RTX 4060 laptop GPU.
