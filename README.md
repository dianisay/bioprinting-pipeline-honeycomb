# WoundBioprinter

**Autonomous 3D wound reconstruction and honeycomb bioprinting via CNN-Transformer vision.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Phase 1** of the doctoral thesis *"CNN-Based ML for 3D Motion Planning and Control in In-Situ Robotic Bioprinters"* ? developed at [Tecnologico de Monterrey](https://tec.mx).
>
> For the MIT collaboration extension (geodesic toolpaths, DeepCurrents surface reconstruction), see [bioprinting-pipeline-livemesh](https://github.com/dianisay/bioprinting-pipeline-livemesh).
>
> For the unified multi-strategy system (honeycomb + geodesic per layer), see [bioprinting-pipeline-multi-strategy-toolpath](https://github.com/dianisay/bioprinting-pipeline-multi-strategy-toolpath).

---

## Overview

A camera-to-deposition pipeline for in-situ wound bioprinting:

```
8 RGB views (eye-in-hand)
        |
   PERCEIVE -------- CT-style volumetric encoder (8x ResNet-18 + 3D Transformer)
        |              PolarDecoder3DLayered -> boundary + depth + layer fill
        v
   SENSE ----------- Intel RealSense D405 depth validation
        |              Confidence-weighted fusion (predicted + measured)
        v
   PLAN ------------ Honeycomb lattice generation on wound void
        |              TSP optimization (MILP) for minimal nozzle travel
        |              Conformal mapping (UV -> curved surface XYZ)
        v
   EXECUTE --------- 8-DOF inverse kinematics (UR5 + XY gantry)
        |              PID + Super-Twisting control
        v
   FEEDBACK -------- Re-scan -> verify fill -> correct next layer
        |
        '---- closed loop ----'
```

---

## Key Contributions

### 1. Volumetric Wound Reconstruction (CT-style)

Instead of a single photo, we capture **8 orthogonal views** and fuse them into a 3D voxel grid ? analogous to CT reconstruction from 2D projections. A 3D Transformer then contextualizes depth from boundary geometry.

### 2. Layer-Aware Polar Decoder

`PolarDecoder3DLayered` predicts wound geometry as:
- **Boundary**: 64 radii in polar coordinates (guaranteed closed, ordered)
- **Depth profile**: per-angle wound depth in mm
- **Layer fill**: bioink amount per layer (cone-shaped deposition)

Direct bioprinting instructions ? no post-processing.

### 3. Closed-Loop Depth Feedback

An eye-in-hand RealSense D405 provides real-time depth validation:
- Before printing: fuse AI prediction with sensor measurement
- After each layer: re-scan, compute residual, correct next pass
- Result: **99% fill accuracy** with 15% initial prediction error (simulation)

### 4. Honeycomb Infill Strategy

Conformal honeycomb lattice + TSP-optimized nozzle path provides:
- Uniform structural support
- Minimal travel distance
- Feasibility validation (boundary-fit check)

---

## Project Structure

```
diana-bioprinting-pipeline/
|
+-- models/                        # Neural network architectures
|   +-- encoder.py                    ResNet-50 + 6-layer Transformer (single-view)
|   +-- volumetric_encoder.py         8x ResNet-18 + volumetric fusion + 3D Transformer
|   +-- volumetric_decoder.py         PolarDecoder3DLayered (boundary + depth + layers)
|   +-- polar_decoder.py              2D PolarDecoder (ablation baseline)
|   +-- detr_decoder.py               DETR-style decoder (ablation)
|   +-- autoregressive_decoder.py     Autoregressive decoder (ablation)
|
+-- modules/                       # Robotics + control
|   +-- honeycomb.py                  Conformal honeycomb lattice generation
|   +-- conformal_mapping.py          UV -> curved surface XYZ mapping
|   +-- tsp_solver.py                 TSP via MILP (PuLP)
|   +-- trajectory_planner.py         Full trajectory: hex grid -> waypoints
|   +-- wound_to_trajectory.py        Bridge: decoder output -> planner input
|   +-- depth_sensor.py               RealSense D405 model (sim + hardware)
|   +-- depth_fusion.py               Confidence-weighted depth blending
|   +-- closed_loop_controller.py     Scan-deposit-verify-correct loop
|   +-- robot_model.py                8-DOF UR5 + XY gantry kinematics
|   +-- inverse_kinematics.py         IK solver (L-BFGS-B + APF + STW)
|   +-- stl_analysis.py               Scaffold geometry extraction
|   +-- visualization_3d.py           3D plotting
|
+-- training/                      # Training & evaluation
|   +-- train.py                      Trainer with early stopping
|   +-- evaluate.py                   Test-set metrics
|   +-- ablation.py                   3-decoder comparison script
|
+-- data/                          # Data loading
|   +-- dataset.py                    Single-view wound dataset
|   +-- multiview_dataset.py          Multi-view synthetic generator + loader
|   +-- polar_conversion.py           Mask <-> polar coordinates
|   +-- synthetic_generator.py        Star-convex wound shapes
|
+-- notebooks/                     # Kaggle experiments
|   +-- 01-ablation-study-kaggle.ipynb     2D ablation (Polar vs DETR vs AR)
|   +-- 02_volumetric_ablation_kaggle.ipynb  4-variant volumetric ablation
|
+-- scripts/
|   +-- demo_pipeline.py              End-to-end demo (image -> robot)
|
+-- tests/                         # Automated tests
+-- configs/                       # Hyperparameters
+-- requirements.txt
+-- pyproject.toml
```

---

## Installation

```bash
git clone https://github.com/dianisay/bioprinting-pipeline-honeycomb.git
cd bioprinting-pipeline-honeycomb
pip install -e .
```

Or without editable install:
```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from models.volumetric_encoder import VolumetricWoundEncoder3D
from models.volumetric_decoder import PolarDecoder3DLayered
import torch

# 8-view wound images (B, 8, 3, 256, 256)
views = torch.randn(1, 8, 3, 256, 256)

encoder = VolumetricWoundEncoder3D(d_model=256, num_views=8, grid_size=8)
decoder = PolarDecoder3DLayered(d_model=256, num_radii=64, num_layers=4, max_depth_mm=5.0)

enc_out = encoder(views, return_voxel_grid=True)
pred = decoder(enc_out['features'])

print(f"Boundary: {pred['radii'].shape}")        # (1, 64) radii
print(f"Depth: {pred['depth'].shape}")            # (1, 64) mm
print(f"Layer fill: {pred['layer_amounts'].shape}")  # (1, 64, 4)
print(f"Voxel grid: {enc_out['voxel_grid'].shape}")  # (1, 256, 8, 8, 8)
```

---

## Kaggle Training

Both notebooks are Kaggle-ready. Upload this repo as a dataset, then:

### Notebook 02: 4-Variant Volumetric Ablation (main result)

Trains and compares:
1. **Baseline** ? single-view + 2D polar (no depth)
2. **WoundBioprinter** ? single-view + 3D polar (depth + layers)
3. **WoundBioprinter3D** ? RGB-D input + 3D polar
4. **Volumetric** ? 8-view CT-style + 3D polar (our contribution)

Metrics: Chamfer distance (mm), Depth MAE (mm), Layer MAE, Honeycomb Feasibility (%), inference time (ms).

Runtime: ~2-3 hours on P100.

### Notebook 01: 2D Decoder Ablation

Compares Polar vs DETR vs Autoregressive decoders (single-view, boundary-only).

---

## Closed-Loop Printing Architecture

```
          +---> DEPOSIT layer k
          |          |
     PLAN layer     SENSE (RealSense D405)
          ^          |
          |     VERIFY fill quality
          |          |
          +----<-- CORRECT (adjust layer k+1)
```

The `closed_loop_controller.py` orchestrates this cycle, achieving robust
deposition even with untrained models or noisy depth predictions.

---

## Relation to LiveMesh

This repository represents the **core bioprinting pipeline** developed at Tecnologico de Monterrey. The subsequent MIT CSAIL collaboration extended this work with:

- **Geodesic toolpaths** (boundary-parallel, curvature-adaptive)
- **DeepCurrents** surface reconstruction (mesh-free)
- **Optimal-transport coverage metrics**

That extension lives at [bioprinting-pipeline-livemesh](https://github.com/dianisay/bioprinting-pipeline-livemesh).

The unified system combining both approaches (AI-driven layer decomposition) is at [bioprinting-pipeline-multi-strategy-toolpath](https://github.com/dianisay/bioprinting-pipeline-multi-strategy-toolpath).

---

## Citation

```bibtex
@phdthesis{roldan2026bioprinting,
    title  = {CNN-Transformer-Based Machine Learning for 3D Motion Planning
              and Control in In-Situ Robotic Bioprinters for Superficial
              Tissue Regeneration},
    author = {Ayala Rold\'an, Diana Paola},
    school = {Tecnol\'ogico de Monterrey},
    year   = {2026},
}
```

## License

MIT
