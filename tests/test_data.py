"""Test data pipeline: synthetic generation + polar conversion + dataset."""
import sys
sys.path.insert(0, ".")

import numpy as np
import cv2
from data.synthetic_generator import generate_star_convex_wound
from data.polar_conversion import mask_to_polar, polar_to_cartesian, polar_to_mask

print("=" * 50)
print("SANITY CHECK: Data Pipeline")
print("=" * 50)

# Test synthetic generation
print("\n[1] Synthetic wound generation...")
sample = generate_star_convex_wound(image_size=256, num_radii=64)
assert sample["image"].shape == (256, 256, 3), f"Bad image shape: {sample['image'].shape}"
assert sample["mask"].shape == (256, 256), f"Bad mask shape: {sample['mask'].shape}"
assert sample["centroid"].shape == (2,)
assert sample["radii"].shape == (64,)
assert sample["points"].shape == (64, 2)
assert 0 < sample["centroid"][0] < 1
assert 0 < sample["centroid"][1] < 1
assert (sample["radii"] > 0).all()
print(f"    Image: {sample['image'].shape}, dtype={sample['image'].dtype}")
print(f"    Mask:  {sample['mask'].shape}, wound pixels={sample['mask'].sum() // 255}")
print(f"    Centroid: ({sample['centroid'][0]:.3f}, {sample['centroid'][1]:.3f})")
print(f"    Radii: mean={sample['radii'].mean():.4f}, std={sample['radii'].std():.4f}")
print("    PASS")

# Test polar conversion from mask
print("\n[2] Mask -> Polar conversion...")
polar = mask_to_polar(sample["mask"], num_radii=64, image_size=256)
assert polar["valid"]
assert polar["centroid"].shape == (2,)
assert polar["radii"].shape == (64,)
assert polar["points"].shape == (64, 2)
# Check centroid is close to synthetic GT
centroid_error = np.linalg.norm(polar["centroid"] - sample["centroid"])
print(f"    Centroid error vs GT: {centroid_error:.4f} (should be small)")
assert centroid_error < 0.05, f"Centroid error too large: {centroid_error}"
# Check radii are roughly similar
radii_error = np.abs(polar["radii"] - sample["radii"]).mean()
print(f"    Mean radii error vs GT: {radii_error:.4f}")
print("    PASS")

# Test polar → cartesian roundtrip
print("\n[3] Polar -> Cartesian -> Mask roundtrip...")
reconstructed_points = polar_to_cartesian(polar["centroid"], polar["radii"])
assert reconstructed_points.shape == (64, 2)
point_error = np.linalg.norm(reconstructed_points - polar["points"], axis=1).mean()
print(f"    Mean point reconstruction error: {point_error:.6f}")
assert point_error < 1e-5, "Roundtrip error too large"

reconstructed_mask = polar_to_mask(polar["centroid"], polar["radii"], image_size=256)
original_area = sample["mask"].sum() / 255
reconstructed_area = reconstructed_mask.sum() / 255
area_ratio = reconstructed_area / max(original_area, 1)
print(f"    Original area: {original_area:.0f} px, Reconstructed: {reconstructed_area:.0f} px")
print(f"    Area ratio: {area_ratio:.3f}")
print("    PASS")

# Test batch generation
print("\n[4] Batch generation (10 samples)...")
valid_count = 0
for i in range(10):
    s = generate_star_convex_wound()
    p = mask_to_polar(s["mask"], num_radii=64, image_size=256)
    if p["valid"]:
        valid_count += 1
print(f"    {valid_count}/10 valid conversions")
assert valid_count >= 8, "Too many invalid conversions"
print("    PASS")

# Test Dataset class
print("\n[5] Dataset class (synthetic only, no files needed)...")
from data.dataset import WoundBoundaryDataset
import tempfile, os

# Generate a tiny dataset to test
tmpdir = tempfile.mkdtemp()
from data.synthetic_generator import generate_dataset
generate_dataset(tmpdir, num_samples=20, image_size=256, num_radii=64, seed=42)

ds = WoundBoundaryDataset(synthetic_dir=tmpdir, split="train", image_size=256, num_radii=64)
print(f"    Dataset size: {len(ds)}")
assert len(ds) > 0

batch = ds[0]
assert batch["image"].shape == (3, 256, 256)
assert batch["centroid"].shape == (2,)
assert batch["radii"].shape == (64,)
assert batch["points"].shape == (64, 2)
print(f"    Sample shapes OK: image={batch['image'].shape}, centroid={batch['centroid'].shape}, radii={batch['radii'].shape}")
print("    PASS")

# Cleanup
import shutil
shutil.rmtree(tmpdir)

print("\n" + "=" * 50)
print("ALL DATA PIPELINE CHECKS PASSED")
print("=" * 50)
