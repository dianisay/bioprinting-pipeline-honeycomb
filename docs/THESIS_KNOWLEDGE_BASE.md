# Thesis Knowledge Base (Quick Retrieval)

## Document Identity
- **Author:** Diana Paola Ayala Roldan
- **Program:** Computational Sciences (Tecnológico de Monterrey)
- **Year:** 2026
- **Thesis title:** Convolutional Neural Network-Based Machine Learning for Three-Dimensional Motion Planning and Control in In-Situ Robotic Bioprinters for Superficial Tissue Regeneration
- **Main language:** English (with Spanish abstract)
- **Core claim:** A CNN-Transformer with volumetric CT-style multi-view fusion enables autonomous image-to-trajectory wound bioprinting.

## One-Paragraph Executive Summary
The thesis proposes and validates an end-to-end computational pipeline for autonomous in-situ robotic bioprinting of superficial wounds. Starting from 8 orthogonal RGB wound images, a VolumetricWoundEncoder3D reconstructs a 3D wound volume (boundary + depth + layer-wise fill) via CT-style feature fusion and a 3D Transformer. A conformal honeycomb infill module generates a normal-aligned deposition path optimized via TSP/MILP. An 8-DOF robotic system (UR5 + XY gantry) executes the toolpath with closed-loop visual monitoring. The ablation study additionally compares 2D decoder variants (Polar, DETR-style, Autoregressive) to validate the polar representation.

## Research Framing

### Problem Statement
- Current in-situ bioprinting remains semi-automated.
- Main gap is computational autonomy: perception + 3D planning + execution.
- Existing pipelines often depend on manual segmentation/calibration and do not guarantee valid closed-loop trajectories.
- Single-view approaches cannot capture wound depth or generate layer-wise instructions.

### Research Question
- Can a CNN-Transformer with volumetric multi-view fusion generate closed-loop 3D deposition trajectories from visual input for autonomous wound treatment?

### Hypothesis
- Volumetric CT-style approach will outperform single-view baselines on boundary accuracy, depth estimation, and honeycomb feasibility.
- Polar decoder will outperform parallel and autoregressive Cartesian decoders on 2D boundary metrics.
- Full integrated system will keep tracking error below 1 mm (simulation).

## Proposed System (6 Modules)
1. **Wound boundary detection + volumetric reconstruction**
   VolumetricWoundEncoder3D (8× ResNet-18 + CT-style volumetric fusion + 3D Transformer) outputs boundary, depth profile, and layer-wise fill instructions via PolarDecoder3DLayered.
2. **Ablation baselines (2D single-view)**
   CNN-Transformer (ResNet-50 + Transformer encoder) with three decoder variants: Polar, DETR-style, Autoregressive.
3. **3D trajectory generation**
   Conformal mapping + honeycomb lattice + TSP/MILP cell ordering + normal-aligned mapping back to 3D.
4. **Robot motion planning and control**
   IK + APF + Super-Twisting control + manipulability optimization for the 8-DOF system.
5. **Execution and real-time feedback**
   Camera monitoring during printing and post-deposition verification.
6. **Validation**
   In-silico evaluation with module and end-to-end metrics.

## Core Technical Innovations

### 1. CT-Style Volumetric Wound Reconstruction (main contribution)
- 8 orthogonal camera views fused into 3D voxel grid.
- Per-view ResNet-18 encoders → volumetric fusion → 3D Transformer.
- Three prediction heads: boundary (polar), depth profile, layer-wise fill amounts.
- Produces direct bioprinting instructions — no post-processing needed.

### 2. Layer-Aware Polar Decoding
- PolarDecoder3DLayered outputs per-layer radii, creating cone-shaped fill pattern.
- Guarantees ordered waypoints and closed loops by construction.
- Layer amounts in [0,1] map directly to bioink deposition commands.

### 3. Polar output representation (2D ablation contribution)
- Predict centroid + radii at fixed angles.
- Guarantees: ordered waypoints, closed loop by construction, graceful failure.
- Known limitation: assumes roughly star-convex wounds.

## Data and Training

### Module 1 (2D Ablation)
- Total training pairs: **2,934** (934 FUSeg real + 2,000 synthetic star-convex)
- Training: Adam, lr 1e-4, batch 8, early stopping patience 10
- Ablation compares three 2D decoders with identical encoder

### Module 2 (Volumetric)
- Synthetic multi-view dataset: configurable size (default 2,000 samples)
- 8 views per sample at 256×256 RGB
- Ground truth: centroid, 64 radii, depth profile, 4-layer fill amounts
- Training: Adam, VolumetricWoundLoss (boundary MSE + depth MAE + layer BCE)

## Key Quantitative Results

> **STATUS: PENDING — training has not been run on Kaggle yet.**
> The notebooks (`01_ablation_study_kaggle.ipynb` and `02_volumetric_ablation_kaggle.ipynb`) are ready to execute.
> All numbers below will be filled after GPU training.

### Ablation (2D, held-out test set)
| Decoder | Chamfer | Hausdorff | IoU | Closure | Ordering |
|---------|---------|-----------|-----|---------|----------|
| Parallel Cartesian (DETR) | TBD | TBD | TBD | TBD | TBD |
| Autoregressive Cartesian | TBD | TBD | TBD | TBD | TBD |
| **Polar (proposed)** | TBD | TBD | TBD | 0.00 mm | 100% |

### Volumetric (CT-style, test set)
| Metric | Value |
|--------|-------|
| Boundary Chamfer (mm) | TBD |
| Boundary IoU | TBD |
| Depth MAE (mm) | TBD |
| Layer-fill accuracy | TBD |
| Honeycomb feasibility (%) | TBD |
| Inference time (ms) | TBD |

### Trajectory + Robot Execution
| Metric | Value |
|--------|-------|
| Wound coverage (%) | TBD |
| TSP travel reduction vs naive (%) | TBD |
| RMS tracking error (mm) | TBD |
| Mean orientation error (deg) | TBD |

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| VolumetricWoundEncoder3D | ✅ Implemented + tested | `models/volumetric_encoder.py` |
| PolarDecoder3DLayered | ✅ Implemented + tested | `models/volumetric_decoder.py` |
| MultiViewWoundDataset | ✅ Implemented + tested | `data/multiview_dataset.py` |
| 2D Encoder (ResNet-50 + Transformer) | ✅ Implemented + tested | `models/encoder.py` |
| Polar/DETR/AR Decoders | ✅ Implemented + tested | `models/polar_decoder.py`, etc. |
| Training loop + evaluate + ablation | ✅ Implemented + tested | `training/` |
| Trajectory planner (honeycomb + TSP) | ✅ Implemented + tested | `modules/` |
| IK + APF + Super-Twisting | ✅ Implemented + tested | `modules/inverse_kinematics.py` |
| Structured logging | ✅ Implemented | `utils/logging_config.py` |
| Kaggle notebook 01 (2D ablation) | ✅ Ready to run | `notebooks/01_ablation_study_kaggle.ipynb` |
| Kaggle notebook 02 (volumetric) | ✅ Ready to run | `notebooks/02_volumetric_ablation_kaggle.ipynb` |
| GPU training execution | ❌ Not yet run | — |
| CoppeliaSim integration | ❌ Not yet done | — |
| Phantom validation | ❌ Not planned (future work) | — |

## Technology Stack
- **Language:** 100% Python
- **Deep Learning:** PyTorch (all models from scratch, no HuggingFace wrappers)
- **Robotics:** Custom numpy implementation (FK, Jacobian, IK with APF + STW)
- **Optimization:** PuLP for MILP/TSP
- **3D Geometry:** OpenCV, Open3D
- **Simulation:** CoppeliaSim via Python ZeroMQ API (planned)
- **Visualization:** matplotlib + plotly
- **Logging:** Python logging with rotating file handler

## Limitations (Explicitly Acknowledged)
- Star-convex assumption fails on highly concave/multi-lobed wounds.
- Simulation-to-reality gap remains significant.
- Single-cylinder parameterization not ideal for complex anatomy.
- Static wound assumption during execution.
- Bioink rheology not modeled in simulation.
- No biological validation (cell viability/tissue integration not evaluated).
- Volumetric approach uses synthetic data (no real multi-view wound images yet).

## Future Work (Proposed in Thesis)
- Multi-lobe / non-star-convex boundary handling.
- Domain adaptation for clinical imaging.
- Multi-patch surface parameterization.
- Real-time geometry updates during deposition (volumetric encoder enables this at ~45ms inference).
- Bioink-aware planning (rheology-informed control).
- Ex vivo → animal → clinical pilot progression.
- Dynamic collision avoidance in clinical environments.

## Fast Lookup Map
- **"What is the main contribution?"** → Volumetric CT-style encoder + layer-aware polar decoder + end-to-end modular autonomous pipeline.
- **"Which module does what?"** → See "Proposed System (6 Modules)".
- **"What needs to run on Kaggle?"** → Notebooks 01 and 02 (training not yet executed).
- **"What is validated?"** → Code passes all unit/integration tests; GPU training results pending.
- **"What are the biggest limitations?"** → See "Limitations".
- **"What's the hardware?"** → 8-DOF: UR5 (6R) + XY gantry (2P), eye-in-hand RGB camera.
