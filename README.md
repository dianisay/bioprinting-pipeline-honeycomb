# Robotic Bioprinting Pipeline

**A robot that heals wounds automatically using AI and 3D printing.**

---

## What Does This Do?

Imagine someone has a wound on their skin. This system:

1. **Looks** at the wound from 8 camera angles (like a CT scanner)
2. **Reconstructs** the full 3D wound shape -- boundary, depth, and layer-wise fill pattern
3. **Plans** a path for a robot arm to fill the wound with bio-material (like a 3D printer, but for skin)
4. **Moves** the robot arm along that path, printing healing material into the wound

All of this happens automatically -- no human needs to control the robot.

The system operates in **closed-loop**: after each layer of material is deposited,
an eye-in-hand depth sensor (Intel RealSense D405) re-scans the wound to verify
the fill and correct the next layer if needed.

---

## How It Works (The 6 Steps)

```
8 photos of wound (multi-view) + depth sensor
            |
[Step 1] AI reconstructs 3D wound volume (CT-style fusion)
            |
[Step 2] Predict boundary + depth + layer-wise fill instructions
            |
[Step 3] Depth sensor validates prediction (fusion)
            |
[Step 4] Plan a honeycomb filling pattern (like a beehive)
            |
[Step 5] Calculate how the robot arm should move
            |
[Step 6] Robot prints layer -> re-scan -> correct -> repeat
```

---

## What's Special About This?

Two key innovations:

### 1. CT-Style Volumetric Reconstruction
Instead of one photo, we use **8 orthogonal views** fused into a 3D voxel grid -- similar to how a CT scanner reconstructs 3D anatomy from 2D X-ray projections. A 3D Transformer then reasons about how depth relates to boundary.

### 2. Layer-Aware Polar Decoding
Instead of flat 2D boundaries, our `PolarDecoder3DLayered` predicts:
- **Boundary**: wound edge in 64 radial directions
- **Depth**: how deep the wound is at each angle
- **Layer fill**: how much bio-ink to deposit per layer (cone-shaped fill pattern)

This produces direct bioprinting instructions -- no post-processing needed.

### 3. Closed-Loop Depth Feedback
An eye-in-hand RGB-D sensor (Intel RealSense D405, ~$300) provides real-time depth
validation during printing:
- **Before printing**: sensor measurement fuses with AI prediction to refine depth estimate
- **After each layer**: re-scan verifies fill quality; correction adjusts next layer
- **Result**: 99% wound fill even with 15% initial prediction error (validated in simulation)

This makes the system robust to prediction inaccuracies and surface variability.

---

## Project Structure

```
diana-bioprinting-pipeline/
|
+-- models/                             # Neural networks
|   +-- encoder.py                      # Single-view encoder (ResNet-50 + Transformer)
|   +-- volumetric_encoder.py           # Multi-view CT-style encoder (8x ResNet-18 + 3D Transformer)
|   +-- volumetric_decoder.py           # Layer-aware polar decoder (boundary + depth + fill)
|   +-- polar_decoder.py                # 2D polar decoder (ablation baseline)
|   +-- detr_decoder.py                 # DETR-style decoder (ablation comparison)
|   +-- autoregressive_decoder.py       # Autoregressive decoder (ablation comparison)
|
+-- modules/                            # Robotics brain
|   +-- stl_analysis.py                 # Reads 3D scaffold shapes
|   +-- honeycomb.py                    # Creates honeycomb fill patterns
|   +-- conformal_mapping.py            # Maps flat patterns onto curved surfaces
|   +-- tsp_solver.py                   # Finds the shortest path between cells (MILP)
|   +-- trajectory_planner.py           # Full UV->XYZ trajectory generation
|   +-- wound_to_trajectory.py          # Bridge: decoder output -> trajectory planner input
|   +-- depth_sensor.py                 # RealSense D405 model (simulated + real interface)
|   +-- depth_fusion.py                 # Fuse predicted + measured depth (confidence-weighted)
|   +-- closed_loop_controller.py       # Scan-deposit-verify-correct printing loop
|   +-- robot_model.py                  # 8-DOF robot arm (UR5 + XY gantry)
|   +-- inverse_kinematics.py           # IK with APF + Super-Twisting control
|   +-- visualization_3d.py             # 3D plotting utilities
|
+-- training/                           # Training scripts
|   +-- train.py                        # Train one model
|   +-- evaluate.py                     # Evaluate on test set (Chamfer, IoU, etc.)
|   +-- ablation.py                     # Compare all decoder variants fairly
|
+-- data/                               # Data loading and generation
|   +-- dataset.py                      # Single-view wound dataset
|   +-- multiview_dataset.py            # Multi-view synthetic wound generator + loader
|   +-- polar_conversion.py             # Mask -> polar coordinate conversion
|   +-- synthetic_generator.py          # Star-convex wound shape generator
|
+-- notebooks/                          # Interactive experiments (Jupyter / Kaggle)
|   +-- 00_demo.ipynb                   # Quick visual demo
|   +-- 01_ablation_study_kaggle.ipynb  # 2D ablation (Polar vs DETR vs AR)
|   +-- 02_volumetric_ablation_kaggle.ipynb  # Volumetric CT-style training
|
+-- scripts/                            # Standalone scripts
|   +-- demo_pipeline.py                # End-to-end pipeline demo (image -> robot)
|
+-- utils/                              # Shared utilities
|   +-- logging_config.py               # Centralized logging (console + rotating file)
|   +-- metrics.py                      # Chamfer, Hausdorff, IoU, closure metrics
|   +-- visualization.py                # Plotting helpers
|
+-- tests/                              # Automated tests
|   +-- test_models.py                  # All decoders sanity check
|   +-- test_volumetric.py              # Volumetric encoder + decoder + loss + gradients
|   +-- test_training_pipeline.py       # Full train->eval loop (CPU, 2 epochs)
|   +-- test_trajectory_pipeline.py     # Honeycomb + TSP + IK integration
|
+-- logs/                               # Runtime logs (auto-generated, gitignored)
+-- configs/                            # Hyperparameters
+-- requirements.txt                    # Python packages needed
```

---

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/dianisay/diana-bioprinting-pipeline.git
cd diana-bioprinting-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests (verify everything works)
python tests/test_models.py
python tests/test_volumetric.py

# 4. Run end-to-end pipeline demo
python scripts/demo_pipeline.py

# 5. Run the demo notebook
jupyter lab notebooks/00_demo.ipynb
```

---

## Training on Kaggle

The neural networks need a GPU to train. We use Kaggle (free T4/P100 GPUs):

### Notebook 01: 2D Ablation Study
Compares three decoder architectures (Polar, DETR, Autoregressive) with the same single-view encoder. Produces Table 4.1 in the thesis.

### Notebook 02: Volumetric CT-Style Training
Trains the full `VolumetricWoundEncoder3D` + `PolarDecoder3DLayered` pipeline. This is the main contribution.

**Steps:**
1. Upload this repo to Kaggle as a dataset
2. Open `notebooks/02_volumetric_ablation_kaggle.ipynb`
3. Enable **GPU T4 x2** accelerator
4. Run all cells (~1-2 hours)
5. Download the `results/` folder (checkpoints + metrics)

The notebook generates synthetic multi-view data, trains the volumetric model, and outputs all metrics needed for Chapter 4 of the thesis.

---

## Logging

All modules log to `logs/pipeline.log` (rotating, 5MB max). To see live logs during training:

```bash
# Windows PowerShell
Get-Content logs\pipeline.log -Wait

# Linux/Mac
tail -f logs/pipeline.log
```

---

## Tools Used

| What | Tool |
|------|------|
| AI / Neural Networks | PyTorch (built from scratch) |
| Math & Optimization | NumPy, SciPy, PuLP |
| 3D Geometry | Open3D, OpenCV |
| Robot Simulation | CoppeliaSim |
| Visualization | Matplotlib, Plotly |
| Training (GPU) | Kaggle |

---

## Who Made This?

**Diana Paola Ayala Roldan**
PhD in Computational Sciences, Tecnologico de Monterrey (2026)

This is the code behind the thesis:
*"CNN-Transformer-Based Machine Learning for 3D Motion Planning and Control in In-Situ Robotic Bioprinters for Superficial Tissue Regeneration"*

---

## License

Academic use -- PhD thesis project.
