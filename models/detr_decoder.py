"""DETR-style Parallel Cartesian Decoder — Ablation baseline.

Predicts N boundary points simultaneously using learned queries and
cross-attention. Requires Hungarian matching for loss assignment.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class DETRDecoder(nn.Module):
    """Parallel set prediction decoder (DETR-style).

    Uses N learned query embeddings that attend to encoder features
    via cross-attention to predict all boundary points simultaneously.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        num_queries: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries

        # Learned query embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer decoder with cross-attention
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

        # Point prediction head
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, encoder_output: torch.Tensor) -> dict:
        """
        Args:
            encoder_output: (B, seq_len, d_model)

        Returns:
            dict with 'points': (B, N, 2)
        """
        B = encoder_output.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        decoded = self.transformer_decoder(queries, encoder_output)
        points = self.point_head(decoded)  # (B, N, 2)

        return {"points": points}


class HungarianLoss(nn.Module):
    """Hungarian matching loss for set prediction.

    Finds optimal assignment between predicted and GT points,
    then computes L2 loss on matched pairs.
    """

    def forward(self, pred: dict, target: dict) -> dict:
        pred_points = pred["points"]  # (B, N, 2)
        target_points = target["points"]  # (B, N, 2)

        B, N, _ = pred_points.shape
        total_loss = torch.tensor(0.0, device=pred_points.device)

        for b in range(B):
            # Cost matrix: pairwise L2 distances
            cost = torch.cdist(pred_points[b], target_points[b], p=2)
            cost_np = cost.detach().cpu().numpy()

            row_idx, col_idx = linear_sum_assignment(cost_np)

            matched_pred = pred_points[b, row_idx]
            matched_target = target_points[b, col_idx]
            total_loss += nn.functional.mse_loss(matched_pred, matched_target)

        return {"total": total_loss / B}
