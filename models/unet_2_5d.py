"""
models/unet_2_5d.py
===================
2.5D U-Net Model.
Reuses the 2D U-Net architecture but enforces 3 input channels 
to process 3 neighboring CT slices [z-1, z, z+1].
"""

from models.unet2d import UNet2D

def build_unet_2_5d(cfg):
    """
    Build 2.5D UNet from config dict.
    Strictly forces in_channels=3.
    """
    return UNet2D(
        in_channels=3,
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        bilinear=cfg["model"]["bilinear"],
    )
