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

## 1. Train TransUNet (Main 20-Epoch Experiment)

```bash
# This is the main training command for the TransUNet model.
# It uses the configs/transunet.yaml configuration.
python scripts/train_transunet.py --epochs 20
```

## 2. Train Base U-Net (Baseline 20-Epoch Experiment)

```bash
# This is the training command for the original 2D U-Net model.
# It uses the configs/unet2d_config.yaml configuration.
python scripts/train_unet.py --epochs 20
```

## 3. Train 2.5D U-Net (20-Epoch Experiment)

```bash
# This is the training command for the 2.5D U-Net model.
# It uses the configs/unet_2_5d_config.yaml configuration.
python scripts/train_unet_2_5d.py --epochs 20
```

## 4. Resume Training

If your training is interrupted (e.g., PC restarts, process crashes), you can resume from the last saved checkpoint:

```bash
# Resume TransUNet training
python scripts/train_transunet.py --epochs 20 --resume experiments/transunet/checkpoints/last_checkpoint.pth

# Resume Base U-Net training
python scripts/train_unet.py --epochs 20 --resume experiments/base_unet/checkpoints/last_checkpoint.pth

# Resume 2.5D U-Net training
python scripts/train_unet_2_5d.py --epochs 20 --resume experiments/unet_2_5d/checkpoints/last_checkpoint.pth
```

## 5. Run Tests

To verify that the model architecture and pipeline are working without launching a full training job, run these tests:

```bash
# Run TransUNet sanity and GPU memory test
python tests/test_transunet.py

# Run TransUNet tiny overfitting test (runs 50 epochs on 4 samples very quickly)
python tests/test_overfit.py

# Run 2.5D U-Net tests
python tests/test_unet_2_5d.py
python tests/test_overfit_2_5d.py
```

## 6. Standalone Evaluation

If you want to re-evaluate a trained model (the evaluation automatically runs at the end of training, but you can run it manually):

```bash
# Evaluate TransUNet
python scripts/evaluate_transunet.py --checkpoint experiments/transunet/checkpoints/best_model.pth

# Evaluate Base U-Net
python scripts/evaluate_unet.py --checkpoint experiments/base_unet/checkpoints/best_model.pth
```

## 6. Random Inference & Segmentation (Demo)

If you want to quickly test the model on random slices and immediately see the segmented visualization panels (saved to a new folder):

```bash
# Test TransUNet on 5 random slices
python scripts/inference.py --model transunet --checkpoint experiments/transunet/checkpoints/best_model.pth --num_slices 5

# Test Base U-Net on 5 random slices
python scripts/inference.py --model unet --checkpoint experiments/base_unet/checkpoints/best_model.pth --num_slices 5

# Test 2.5D U-Net on 5 random slices
python scripts/inference.py --model unet_2_5d --checkpoint experiments/unet_2_5d/checkpoints/best_model.pth --num_slices 5
```

---

## 💡 Notes on Hardware & Execution
* **Progress Bar:** `tqdm` progress bars have been added, so when you run the training commands, you will now see real-time progress for both the training and validation epochs.
* **AMP (Mixed Precision):** Is enabled by default to save VRAM and increase speed.
* **VRAM Usage:** The batch size for TransUNet is set to `4`, and peak VRAM is tightly managed (~500 MB), so you should never hit CUDA OOM on your RTX 4060 laptop GPU.
