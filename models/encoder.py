"""CNN-Transformer Encoder: ResNet-50 backbone + Transformer encoder.

Extracts spatial features from wound images and models global context
via self-attention, producing a rich feature representation for decoding.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import math


class PositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding for spatial feature maps."""

    def __init__(self, d_model: int, h: int = 16, w: int = 16):
        super().__init__()
        self.row_embed = nn.Embedding(h, d_model // 2)
        self.col_embed = nn.Embedding(w, d_model // 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, HW, D) — adds positional encoding."""
        h = w = int(math.sqrt(x.shape[1]))
        rows = torch.arange(h, device=x.device)
        cols = torch.arange(w, device=x.device)
        row_emb = self.row_embed(rows).unsqueeze(1).expand(-1, w, -1)
        col_emb = self.col_embed(cols).unsqueeze(0).expand(h, -1, -1)
        pos = torch.cat([row_emb, col_emb], dim=-1).reshape(h * w, -1)
        return x + pos.unsqueeze(0)


class CNNTransformerEncoder(nn.Module):
    """ResNet-50 feature extraction + Transformer encoder for global context.

    Input:  (B, 3, 256, 256) RGB image
    Output: (B, 256, 256) feature sequence ready for decoder heads
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # ResNet-50 backbone (remove avgpool + fc)
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Output: (B, 2048, 8, 8) for 256x256 input

        # 1x1 projection: 2048 -> d_model
        self.projection = nn.Conv2d(2048, d_model, kernel_size=1)

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model, h=8, w=8)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 256, 256) input image

        Returns:
            (B, 64, 256) encoded feature sequence
        """
        # CNN feature extraction
        features = self.backbone(x)  # (B, 2048, 8, 8)
        features = self.projection(features)  # (B, 256, 8, 8)

        # Reshape to sequence
        B, C, H, W = features.shape
        tokens = features.flatten(2).permute(0, 2, 1)  # (B, 64, 256)

        # Add positional encoding
        tokens = self.pos_encoding(tokens)

        # Transformer encoder
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        return tokens
