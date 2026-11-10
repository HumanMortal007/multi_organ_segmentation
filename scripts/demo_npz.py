import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def show_npz(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Could not find file {filepath}")
        return

    print(f"Loading {filepath}...")
    data = np.load(filepath)
    image = data["image"]
    label = data["label"]

    ORGAN_COLORS = [
        (0.0,  0.0,  0.0,  1.0),   # 0 background  - black
        (1.0,  0.0,  0.0,  1.0),   # 1 aorta        - red
        (1.0,  1.0,  0.0,  1.0),   # 2 gallbladder  - yellow
        (0.0,  0.0,  1.0,  1.0),   # 3 spleen        - blue
        (0.0,  1.0,  0.0,  1.0),   # 4 left kidney  - green
        (0.0,  1.0,  1.0,  1.0),   # 5 right kidney - cyan
        (1.0,  0.5,  0.0,  1.0),   # 6 liver         - orange
        (0.5,  0.0,  0.5,  1.0),   # 7 stomach       - purple
        (1.0,  0.0,  1.0,  1.0),   # 8 pancreas      - magenta
    ]

    h, w = label.shape
    rgb_label = np.zeros((h, w, 3), dtype=np.float32)
    for cls_idx, color in enumerate(ORGAN_COLORS):
        mask = (label == cls_idx)
        rgb_label[mask] = color[:3]

    # Create the popup window
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    filename = os.path.basename(filepath)
    fig.suptitle(f"Data Extracted from: {filename}", fontsize=16, fontweight="bold")

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input (Raw CT Image)")
    axes[0].axis("off")

    axes[1].imshow(rgb_label)
    axes[1].set_title("Target (Ground Truth Mask)")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    overlay_alpha = np.any(rgb_label > 0, axis=2).astype(np.float32) * 0.5
    axes[2].imshow(rgb_label, alpha=overlay_alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    print("Opening visualization window... (Close the window to end the script)")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/Synapse/train_npz/case0005_slice050.npz", help="Path to .npz file")
    args = parser.parse_args()
    show_npz(args.file)
