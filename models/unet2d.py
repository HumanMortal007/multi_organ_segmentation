"""
models/unet2d.py
=================
Standard 2D U-Net for multi-class segmentation.

Architecture:
  Encoder:  32 -> 64 -> 128 -> 256 channels (4 stages)
  Bottleneck: 512 channels
  Decoder:  256 -> 128 -> 64 -> 32 (skip connections)
  Output: 1x1 conv -> [B, num_classes, H, W]

Lightweight enough for NVIDIA RTX 4060 Laptop GPU (8 GB VRAM).
Uses bilinear upsampling + conv (saves VRAM vs transposed conv).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BN -> ReLU blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool then DoubleConv (encoder step)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample, concatenate skip, then DoubleConv (decoder step)."""

    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    Lightweight 2D U-Net.

    Args:
        in_channels  (int): number of input channels (1 for grayscale CT).
        num_classes  (int): number of output segmentation classes.
        base_channels(int): base channel count (default 32).
        bilinear     (bool): use bilinear upsampling (True saves VRAM).
    """

    def __init__(self, in_channels=1, num_classes=9, base_channels=32, bilinear=True):
        super().__init__()
        b = base_channels
        self.inc   = DoubleConv(in_channels, b)       # 1  -> 32
        self.down1 = Down(b,     b * 2)               # 32 -> 64
        self.down2 = Down(b * 2, b * 4)               # 64 -> 128
        self.down3 = Down(b * 4, b * 8)               # 128 -> 256
        self.down4 = Down(b * 8, b * 16)              # 256 -> 512  (bottleneck)

        self.up1   = Up(b * 16, b * 8,  b * 8,  bilinear)   # 512 + 256 -> 256
        self.up2   = Up(b * 8,  b * 4,  b * 4,  bilinear)   # 256 + 128 -> 128
        self.up3   = Up(b * 4,  b * 2,  b * 2,  bilinear)   # 128 + 64  -> 64
        self.up4   = Up(b * 2,  b,      b,       bilinear)   # 64  + 32  -> 32

        self.outc  = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)    # [B, 32,  H,    W   ]
        x2 = self.down1(x1) # [B, 64,  H/2,  W/2 ]
        x3 = self.down2(x2) # [B, 128, H/4,  W/4 ]
        x4 = self.down3(x3) # [B, 256, H/8,  W/8 ]
        x5 = self.down4(x4) # [B, 512, H/16, W/16]  bottleneck

        # Decoder with skip connections
        x = self.up1(x5, x4) # [B, 256, H/8,  W/8 ]
        x = self.up2(x,  x3) # [B, 128, H/4,  W/4 ]
        x = self.up3(x,  x2) # [B, 64,  H/2,  W/2 ]
        x = self.up4(x,  x1) # [B, 32,  H,    W   ]

        logits = self.outc(x) # [B, num_classes, H, W]
        return logits


def build_unet2d(cfg):
    """Build UNet2D from config dict."""
    return UNet2D(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        bilinear=cfg["model"]["bilinear"],
    )


def count_parameters(model):
    """Return (trainable_params, total_params) as millions."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable / 1e6, total / 1e6
