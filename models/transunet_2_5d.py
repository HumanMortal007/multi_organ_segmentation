"""
models/transunet_2_5d.py
=========================
2.5D TransUNet — a thin wrapper around the existing 2D TransUNet.

The 2.5D formulation uses adjacent CT slices as input channels:
  [z-1, z, z+1]  →  [3, H, W]

After batching:
  [B, 3, H, W]

The model predicts only the segmentation mask for the middle slice z:
  output: [B, 9, H, W]  (logits for 9 classes)

Architecture is IDENTICAL to the 2D TransUNet (models/transunet.py).
The ONLY change is in_channels = 3 instead of 1.

This means:
  - CNN Encoder: first DoubleConv accepts 3 channels instead of 1.
    Parameter delta: 2 × 32 × 3 × 3 = 576 extra weights (< 0.001M).
  - Patch Embedding: unchanged.
  - Vision Transformer: unchanged (6 layers, 8 heads, hidden_dim=256).
  - Cascaded Decoder: unchanged.

No pretrained weights are used (same as 2D TransUNet).
"""

from models.transunet import TransUNet


def build_transunet_2_5d(cfg):
    """
    Build 2.5D TransUNet from config dict.
    Strictly forces in_channels=3 regardless of config to prevent accidents.
    All other hyperparameters come from the config.
    """
    m_cfg = cfg["model"]

    # Safety check — force 3 channels for 2.5D input
    in_channels = m_cfg.get("in_channels", 3)
    if in_channels != 3:
        raise ValueError(
            f"[transunet_2_5d] Expected in_channels=3 for 2.5D input, "
            f"but got in_channels={in_channels}. "
            f"Check configs/transunet_2_5d_config.yaml."
        )

    return TransUNet(
        in_channels=3,
        num_classes=m_cfg["num_classes"],
        img_size=m_cfg["img_size"],
        patch_size=m_cfg["patch_size"],
        hidden_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        num_heads=m_cfg["num_heads"],
        mlp_dim=m_cfg["mlp_dim"],
        dropout=m_cfg.get("dropout", 0.1),
        cnn_channels=m_cfg["cnn_channels"],
        bilinear=m_cfg["bilinear"],
    )

