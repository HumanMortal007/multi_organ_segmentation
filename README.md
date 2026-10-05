# Multi-Organ Segmentation using 2D, 2.5D U-Net, and TransUNet

This repository contains a lightweight, PyTorch-based implementation for Multi-Organ CT Image Segmentation. The project investigates whether incorporating limited through-plane context into 2D segmentation networks (2.5D slice-stacking) and utilizing Vision Transformers (TransUNet) can improve abdominal multi-organ segmentation while retaining the practical advantages of 2D processing on consumer-grade hardware.

## 🎯 Project Objectives
1. **Base 2D U-Net**: Establish a baseline multi-organ segmentation using standard 2D convolutions.
2. **2.5D U-Net**: Stack 3 adjacent depth slices (`z-1`, `z`, `z+1`) as input channels to inject spatial Z-axis continuity into a highly efficient 2D architecture.
3. **2D TransUNet**: Utilize a Vision Transformer (ViT) bottleneck to capture global long-range self-attention for complex organs.

## 📊 Dataset
We use the **Synapse Multi-Organ CT Dataset** (BTCV). The scans are preprocessed into highly optimized 2D `.npz` arrays.
* **Classes**: 9 (Background, Aorta, Gallbladder, Spleen, Left/Right Kidney, Liver, Stomach, Pancreas)
* **Training Data**: 18 Volumes (extracted to individual `.npz` slices)
* **Test Data**: 12 Volumes (Evaluated slice-by-slice as full 3D volumes)

## 🏗️ Architectures

### 1. Base 2D U-Net
A lightweight Convolutional Neural Network with 4 Down/Up stages. Extremely fast and memory-efficient.

### 2. 2.5D U-Net
Modifies the data loader to dynamically fetch `[z-1, z, z+1]` for any given slice `z`. The input shape becomes `[B, 3, H, W]`. Boundary slices (e.g., when `z-1` doesn't exist) duplicate `z` to prevent data leakage. This provides volumetric awareness without the $O(N^3)$ computational explosion of true 3D models.

### 3. TransUNet
A hybrid CNN-ViT architecture. It uses a CNN to extract high-resolution spatial feature maps, a Vision Transformer to encode global contextual relationships, and a cascaded upsampler for precise mask generation.

## 🚀 Performance & Results (20 Epochs)

| Metric / Organ       | Base 2D U-Net | 2.5D U-Net | 2D TransUNet |
|----------------------|---------------|------------|--------------|
| **Test Mean Dice**   | 0.7234        | 0.7245     | **0.7464**   |
| **Best Val Dice**    | 0.8405        | 0.8735     | **0.8707**   |
| **Max VRAM Used**    | ~0.90 GB      | ~0.91 GB   | ~0.87 GB     |

**Inference:**
* TransUNet dominated on complex scattered organs (e.g., Stomach and Pancreas).
* 2.5D U-Net provided targeted improvements on continuous/tubular structures over the baseline 2D U-Net.
* All models trained easily on an 8GB RTX 4060, peaking at under 1GB VRAM by utilizing Automatic Mixed Precision (AMP) and Bilinear Upsampling.

## 🛠️ Usage & Commands

See the comprehensive [RUN_COMMANDS.md](RUN_COMMANDS.md) for full execution details. 

**Quick Start Inference:**
```bash
# Test 2.5D U-Net on 5 random slices
python scripts/inference.py --model unet_2_5d --checkpoint experiments/unet_2_5d/checkpoints/best_model.pth --num_slices 5
```

**Training:**
```bash
# Train TransUNet
python scripts/train_transunet.py --epochs 20

# Train 2.5D U-Net
python scripts/train_unet_2_5d.py --epochs 20
```

## 📝 Dependencies
* Python 3.8+
* PyTorch (with CUDA)
* NumPy, SciPy, Matplotlib
* `medpy` (for HD95 metric evaluation)
