"""Quick sanity check for all models."""
import sys
sys.path.insert(0, ".")

import torch
from models.encoder import CNNTransformerEncoder
from models.polar_decoder import PolarDecoder, PolarBoundaryLoss
from models.detr_decoder import DETRDecoder, HungarianLoss
from models.autoregressive_decoder import AutoregressiveDecoder, AutoregressiveLoss
from utils.metrics import chamfer_distance, hausdorff_distance, boundary_iou, closure_error
from configs.default import Config
import numpy as np

print("=" * 50)
print("SANITY CHECK: Models + Metrics")
print("=" * 50)

# Config
cfg = Config()
print(f"\nConfig loaded: {cfg.encoder.backbone}, d={cfg.encoder.projection_dim}")

# Encoder
print("\n[1] Encoder forward pass...")
enc = CNNTransformerEncoder(pretrained=False)
x = torch.randn(2, 3, 256, 256)
features = enc(x)
print(f"    Input:  {x.shape}")
print(f"    Output: {features.shape}")
assert features.shape == (2, 64, 256), f"Unexpected shape: {features.shape}"
print("    PASS")

# Polar Decoder
print("\n[2] Polar Decoder...")
dec = PolarDecoder()
out = dec(features)
print(f"    Centroid: {out['centroid'].shape}")
print(f"    Radii:    {out['radii'].shape}")
print(f"    Points:   {out['points'].shape}")
assert out["points"].shape == (2, 64, 2)
print("    PASS")

# Polar Loss
print("\n[3] Polar Loss...")
target = {
    "centroid": torch.rand(2, 2),
    "radii": torch.rand(2, 64) * 0.3,
    "points": torch.rand(2, 64, 2),
}
loss_fn = PolarBoundaryLoss()
losses = loss_fn(out, target)
print(f"    Total: {losses['total']:.4f} (centroid: {losses['centroid']:.4f}, radii: {losses['radii']:.4f}, points: {losses['points']:.4f})")
print("    PASS")

# DETR Decoder
print("\n[4] DETR Decoder...")
detr = DETRDecoder()
detr_out = detr(features)
print(f"    Points: {detr_out['points'].shape}")
assert detr_out["points"].shape == (2, 64, 2)
print("    PASS")

# Autoregressive Decoder (teacher forcing)
print("\n[5] Autoregressive Decoder (teacher forcing)...")
ar = AutoregressiveDecoder()
ar_out = ar(features, target_points=torch.rand(2, 64, 2))
print(f"    Points: {ar_out['points'].shape}")
assert ar_out["points"].shape == (2, 64, 2)
print("    PASS")

# Metrics
print("\n[6] Metrics...")
pts_a = np.random.rand(64, 2)
pts_b = pts_a + np.random.randn(64, 2) * 0.02
chamfer = chamfer_distance(pts_a, pts_b)
haus = hausdorff_distance(pts_a, pts_b)
iou = boundary_iou(pts_a, pts_b)
closure = closure_error(pts_a)
print(f"    Chamfer:  {chamfer:.4f}")
print(f"    Hausdorff: {haus:.4f}")
print(f"    IoU:      {iou:.4f}")
print(f"    Closure:  {closure:.4f}")
print("    PASS")

print("\n" + "=" * 50)
print("ALL CHECKS PASSED")
print("=" * 50)
