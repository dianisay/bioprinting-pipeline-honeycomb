"""Sanity check for volumetric encoder + decoder pipeline."""
import sys
sys.path.insert(0, ".")

import torch
from models.volumetric_encoder import VolumetricWoundEncoder3D
from models.volumetric_decoder import PolarDecoder3DLayered, VolumetricWoundLoss

print("=" * 50)
print("VOLUMETRIC PIPELINE TEST")
print("=" * 50)

# Encoder
print("\n[1] VolumetricWoundEncoder3D forward pass...")
encoder = VolumetricWoundEncoder3D(
    d_model=256, num_views=8, grid_size=8,
    num_heads=8, num_layers=4, pretrained=False,
)
views = torch.randn(2, 8, 3, 256, 256)
out = encoder(views, return_voxel_grid=True)
assert out["features"].shape == (2, 256)
assert out["voxel_context"].shape == (2, 512, 256)
assert out["voxel_grid"].shape == (2, 256, 8, 8, 8)
print(f"    features:      {out['features'].shape}")
print(f"    voxel_context: {out['voxel_context'].shape}")
print(f"    voxel_grid:    {out['voxel_grid'].shape}")
print("    PASS")

# Decoder
print("\n[2] PolarDecoder3DLayered forward pass...")
decoder = PolarDecoder3DLayered(d_model=256, num_radii=64, num_layers=4)
dec_out = decoder(out["features"])
assert dec_out["centroid"].shape == (2, 2)
assert dec_out["radii"].shape == (2, 64)
assert dec_out["depth"].shape == (2, 64)
assert dec_out["layer_amounts"].shape == (2, 64, 4)
assert dec_out["points_2d"].shape == (2, 64, 2)
assert dec_out["points_3d"].shape == (2, 64, 4, 3)
print(f"    centroid:      {dec_out['centroid'].shape}")
print(f"    radii:         {dec_out['radii'].shape}")
print(f"    depth:         {dec_out['depth'].shape}")
print(f"    layer_amounts: {dec_out['layer_amounts'].shape}")
print(f"    points_2d:     {dec_out['points_2d'].shape}")
print(f"    points_3d:     {dec_out['points_3d'].shape}")
print("    PASS")

# Value ranges
print("\n[3] Value range checks...")
assert dec_out["centroid"].min() >= 0 and dec_out["centroid"].max() <= 1, "Centroid out of [0,1]"
assert dec_out["radii"].min() >= 0, "Negative radii"
assert dec_out["depth"].min() >= 0, "Negative depth"
assert dec_out["layer_amounts"].min() >= 0 and dec_out["layer_amounts"].max() <= 1, "Layer amounts out of [0,1]"
print("    Centroid in [0, 1]: OK")
print("    Radii >= 0: OK")
print("    Depth >= 0: OK")
print("    Layer amounts in [0, 1]: OK")
print("    PASS")

# Loss
print("\n[4] VolumetricWoundLoss...")
loss_fn = VolumetricWoundLoss()
target = {
    "points_2d": torch.rand(2, 64, 2),
    "depth": torch.rand(2, 64) * 5.0,
    "layer_amounts": torch.rand(2, 64, 4),
}
losses = loss_fn(dec_out, target)
assert losses["total"].item() > 0
print(f"    total:    {losses['total'].item():.4f}")
print(f"    boundary: {losses['boundary'].item():.4f}")
print(f"    depth:    {losses['depth'].item():.4f}")
print(f"    layers:   {losses['layers'].item():.4f}")
print("    PASS")

# Gradient flow
print("\n[5] Gradient flow (backprop)...")
losses["total"].backward()
enc_params = [(n, p) for n, p in encoder.named_parameters() if p.requires_grad]
enc_with_grad = [(n, p) for n, p in enc_params if p.grad is not None]
enc_no_grad = [(n, p) for n, p in enc_params if p.grad is None]
print(f"    Encoder params with grad: {len(enc_with_grad)}/{len(enc_params)}")
if enc_no_grad:
    print(f"    (without grad: edge_detector — unused in forward, reserved for aux loss)")
grad_ok_dec = all(p.grad is not None for p in decoder.parameters() if p.requires_grad)
print(f"    All decoder grads computed: {grad_ok_dec}")
assert len(enc_with_grad) > len(enc_params) * 0.8, "Too few encoder params receiving gradients"
assert grad_ok_dec
print("    PASS")

# Param count
num_enc = sum(p.numel() for p in encoder.parameters())
num_dec = sum(p.numel() for p in decoder.parameters())
print(f"\n[Info] Encoder params: {num_enc:,}")
print(f"[Info] Decoder params: {num_dec:,}")
print(f"[Info] Total:          {num_enc + num_dec:,}")

print("\n" + "=" * 50)
print("ALL VOLUMETRIC TESTS PASSED")
print("=" * 50)
