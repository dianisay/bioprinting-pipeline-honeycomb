"""Autoregressive Cartesian Decoder — Ablation baseline.

Predicts boundary points one at a time, each conditioned on all
previously predicted points. Uses teacher forcing during training.
"""

import torch
import torch.nn as nn


class AutoregressiveDecoder(nn.Module):
    """Sequential point prediction via autoregressive Transformer decoder.

    Generates boundary points one by one, where each prediction is
    conditioned on encoder features and all previously generated points.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        num_points: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model

        # Input embedding for point coordinates
        self.point_embed = nn.Linear(2, d_model)

        # Start token (learned)
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding for sequence position
        self.pos_embed = nn.Embedding(num_points + 1, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),
        )

    def forward(
        self, encoder_output: torch.Tensor, target_points: torch.Tensor = None
    ) -> dict:
        """
        Args:
            encoder_output: (B, seq_len, d_model)
            target_points: (B, N, 2) — GT points for teacher forcing (training only)

        Returns:
            dict with 'points': (B, N, 2)
        """
        if target_points is not None:
            return self._forward_teacher_forcing(encoder_output, target_points)
        return self._forward_autoregressive(encoder_output)

    def _forward_teacher_forcing(
        self, encoder_output: torch.Tensor, target_points: torch.Tensor
    ) -> dict:
        B = encoder_output.shape[0]

        # Embed target points and prepend start token
        embedded = self.point_embed(target_points)  # (B, N, d_model)
        start = self.start_token.expand(B, -1, -1)
        tgt = torch.cat([start, embedded[:, :-1]], dim=1)  # (B, N, d_model)

        # Add positional encoding
        positions = torch.arange(self.num_points, device=tgt.device)
        tgt = tgt + self.pos_embed(positions).unsqueeze(0)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.num_points, device=tgt.device
        )

        decoded = self.transformer_decoder(
            tgt, encoder_output, tgt_mask=causal_mask
        )
        points = self.output_head(decoded)  # (B, N, 2)

        return {"points": points}

    @torch.no_grad()
    def _forward_autoregressive(self, encoder_output: torch.Tensor) -> dict:
        B = encoder_output.shape[0]
        device = encoder_output.device

        generated = []
        current_input = self.start_token.expand(B, -1, -1)

        for i in range(self.num_points):
            pos = self.pos_embed(torch.tensor([i], device=device))
            tgt = current_input + pos

            decoded = self.transformer_decoder(tgt, encoder_output)
            point = self.output_head(decoded[:, -1:])  # (B, 1, 2)
            generated.append(point)

            # Next input
            next_embed = self.point_embed(point)
            current_input = torch.cat([current_input, next_embed], dim=1)

        points = torch.cat(generated, dim=1)  # (B, N, 2)
        return {"points": points}


class AutoregressiveLoss(nn.Module):
    """Simple MSE loss for autoregressive predictions (already ordered)."""

    def forward(self, pred: dict, target: dict) -> dict:
        loss = nn.functional.mse_loss(pred["points"], target["points"])
        return {"total": loss}
