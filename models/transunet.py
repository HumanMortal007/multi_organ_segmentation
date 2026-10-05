"""
models/transunet.py
===================
2D TransUNet implementation.
Follows the paper "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"
but implemented from scratch in a lightweight form to fit in 8GB VRAM.

Architecture:
1. CNN Feature Extractor (Encoder) -> features at multiple resolutions for skip connections.
2. Patch Embedding -> 1D sequence of tokens.
3. Transformer Encoder -> multi-head self-attention on tokens.
4. Reshape tokens -> 2D feature map.
5. Cascaded Upsampler (Decoder) -> U-Net style decoder with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# CNN Encoder (Stem)
# ==============================================================================

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


class CNNEncoder(nn.Module):
    """
    Standard CNN encoder extracting features at 4 resolutions.
    Output spatial dimensions: (H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8), (H/16, W/16).
    """
    def __init__(self, in_channels, channels=[32, 64, 128, 256]):
        super().__init__()
        self.inc = DoubleConv(in_channels, channels[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(channels[0], channels[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(channels[1], channels[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(channels[2], channels[3]))
        # We stop at down3 for CNN, since Transformer will process the bottleneck.

    def forward(self, x):
        x1 = self.inc(x)      # [B, C1, H,   W  ]
        x2 = self.down1(x1)   # [B, C2, H/2, W/2]
        x3 = self.down2(x2)   # [B, C3, H/4, W/4]
        x4 = self.down3(x3)   # [B, C4, H/8, W/8]
        # x4 will be patched and sent to the Transformer
        return x1, x2, x3, x4


# ==============================================================================
# Patch Embedding
# ==============================================================================

class PatchEmbedding(nn.Module):
    """
    Converts 2D feature map [B, C, H/8, W/8] into a sequence of tokens.
    Uses a 2D convolution to extract patches.
    """
    def __init__(self, in_channels, hidden_dim, patch_size=2):
        super().__init__()
        # patch_size=2 relative to H/8 means overall patch size of 16 in original image.
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H/8, W/8]
        x = self.proj(x)                 # [B, hidden_dim, H/(8*patch_size), W/(8*patch_size)]
        x = x.flatten(2)                 # [B, hidden_dim, N_tokens]
        x = x.transpose(1, 2)            # [B, N_tokens, hidden_dim]
        return x


# ==============================================================================
# Vision Transformer Encoder
# ==============================================================================

class TransformerEncoder(nn.Module):
    """
    Standard Vision Transformer Encoder.
    """
    def __init__(self, num_tokens, hidden_dim, num_layers, num_heads, mlp_dim, dropout):
        super().__init__()
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=mlp_dim, 
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, N_tokens, hidden_dim]
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x) # [B, N_tokens, hidden_dim]
        return x


# ==============================================================================
# Cascaded Upsampler (Decoder)
# ==============================================================================

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
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class CascadedUpsampler(nn.Module):
    """
    U-Net style decoder combining transformer features and CNN skip connections.
    """
    def __init__(self, hidden_dim, cnn_channels, num_classes, bilinear=True):
        super().__init__()
        # cnn_channels = [C1, C2, C3, C4]
        # Transformer output has hidden_dim and spatial size H/16.
        
        # up1: H/16 -> H/8, concat with x4 (C4)
        self.up1 = Up(hidden_dim, cnn_channels[3], cnn_channels[2], bilinear)
        # up2: H/8 -> H/4, concat with x3 (C3)
        self.up2 = Up(cnn_channels[2], cnn_channels[2], cnn_channels[1], bilinear)
        # up3: H/4 -> H/2, concat with x2 (C2)
        self.up3 = Up(cnn_channels[1], cnn_channels[1], cnn_channels[0], bilinear)
        # up4: H/2 -> H, concat with x1 (C1)
        self.up4 = Up(cnn_channels[0], cnn_channels[0], cnn_channels[0], bilinear)
        
        self.outc = nn.Conv2d(cnn_channels[0], num_classes, kernel_size=1)

    def forward(self, x, skips):
        # skips: x1, x2, x3, x4
        x1, x2, x3, x4 = skips
        
        x = self.up1(x, x4)   # H/8, W/8
        x = self.up2(x, x3)   # H/4, W/4
        x = self.up3(x, x2)   # H/2, W/2
        x = self.up4(x, x1)   # H, W
        
        logits = self.outc(x)
        return logits


# ==============================================================================
# Full TransUNet Model
# ==============================================================================

class TransUNet(nn.Module):
    """
    2D TransUNet for Multi-Organ Segmentation.
    """
    def __init__(self, in_channels=1, num_classes=9, img_size=224, patch_size=16, 
                 hidden_dim=256, num_layers=6, num_heads=8, mlp_dim=1024, dropout=0.1, 
                 cnn_channels=[32, 64, 128, 256], bilinear=True):
        super().__init__()
        
        # 1. CNN Feature Extractor
        self.cnn = CNNEncoder(in_channels, cnn_channels)
        
        # The CNN produces 4 feature maps. The deepest one (x4) has spatial size H/8.
        # We need to extract patches from it. A patch_size of 16 in the original image
        # corresponds to a patch_size of 16/8 = 2 in the x4 feature map.
        cnn_patch_size = patch_size // 8
        num_patches = (img_size // patch_size) ** 2  # 224/16 = 14 -> 196 tokens
        self.num_patches_side = img_size // patch_size
        
        # 2. Patch Embedding
        self.patch_embed = PatchEmbedding(cnn_channels[-1], hidden_dim, patch_size=cnn_patch_size)
        
        # 3. Vision Transformer
        self.transformer = TransformerEncoder(num_patches, hidden_dim, num_layers, num_heads, mlp_dim, dropout)
        
        # 4. Decoder
        self.decoder = CascadedUpsampler(hidden_dim, cnn_channels, num_classes, bilinear)

    def forward(self, x):
        # 1. CNN Extraction
        x1, x2, x3, x4 = self.cnn(x)
        # x1: [B, C1, H, W]
        # x2: [B, C2, H/2, W/2]
        # x3: [B, C3, H/4, W/4]
        # x4: [B, C4, H/8, W/8]
        
        # 2. Patch Embedding
        tokens = self.patch_embed(x4) # [B, 196, hidden_dim]
        
        # 3. Transformer
        encoded_tokens = self.transformer(tokens) # [B, 196, hidden_dim]
        
        # 4. Reshape Tokens -> 2D Feature Map
        B, N, D = encoded_tokens.shape
        H_t = W_t = self.num_patches_side
        encoded_features = encoded_tokens.transpose(1, 2).contiguous().view(B, D, H_t, W_t) # [B, hidden_dim, H/16, W/16]
        
        # 5. Cascaded Upsampler
        logits = self.decoder(encoded_features, skips=(x1, x2, x3, x4))
        
        return logits


def build_transunet(cfg):
    """Build TransUNet from config dict."""
    m_cfg = cfg["model"]
    return TransUNet(
        in_channels=m_cfg["in_channels"],
        num_classes=m_cfg["num_classes"],
        img_size=m_cfg["img_size"],
        patch_size=m_cfg["patch_size"],
        hidden_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        num_heads=m_cfg["num_heads"],
        mlp_dim=m_cfg["mlp_dim"],
        dropout=m_cfg.get("dropout", 0.1),
        cnn_channels=m_cfg["cnn_channels"],
        bilinear=m_cfg["bilinear"]
    )
