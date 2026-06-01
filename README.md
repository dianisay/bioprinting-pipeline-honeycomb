# Autonomous In-Situ Robotic Bioprinting Pipeline

**CNN-Transformer-Based Machine Learning for 3D Motion Planning and Control in In-Situ Robotic Bioprinters for Superficial Tissue Regeneration**

Diana Paola Ayala Roldán — PhD in Computational Sciences, Tecnológico de Monterrey (2026)

---

## Overview

End-to-end autonomous pipeline that takes a single RGB wound image and produces a 3D robotic deposition trajectory, executed with closed-loop visual monitoring. The system combines deep learning for wound boundary detection with computational geometry for trajectory generation and robot control.

## Architecture

```
RGB Image → [Module 1: CNN-Transformer] → Wound Boundary (polar)
         → [Module 2: Multi-View 3D Reconstruction] → Surface Mesh
         → [Module 3: Conformal Honeycomb + TSP] → 3D Toolpath
         → [Module 4: IK + PID Control] → Joint Commands
         → [Module 5: Closed-Loop Execution] → Deposited Material
```

## Key Contribution

A **polar-parameterized decoder** that predicts wound boundaries as radii at fixed angular intervals around a centroid. This guarantees:
- Ordered waypoints by construction
- Closed-loop contour (zero closure error)
- Graceful degradation for nearly star-convex wounds

## Project Structure

```
diana-bioprinting-pipeline/
├── models/              # Neural network architectures
│   ├── encoder.py       # ResNet-50 + Transformer encoder
│   ├── polar_decoder.py # Proposed polar decoder
│   ├── detr_decoder.py  # Ablation: DETR-style parallel
│   └── autoregressive_decoder.py  # Ablation: autoregressive
├── modules/             # Pipeline modules (2-5)
│   ├── reconstruction.py    # Multi-view 3D reconstruction
│   ├── trajectory.py        # Conformal honeycomb + TSP
│   ├── motion_planning.py   # IK + manipulability
│   ├── robot_control.py     # PID + CoppeliaSim interface
│   └── execution.py         # Closed-loop monitoring
├── training/            # Training scripts
│   ├── train.py         # Main training loop
│   ├── evaluate.py      # Evaluation metrics
│   └── ablation.py      # Ablation study runner
├── utils/               # Shared utilities
│   ├── kinematics.py    # FK/IK for 8-DOF system
│   ├── conformal.py     # Conformal mapping
│   ├── tsp_solver.py    # PuLP MILP solver
│   ├── metrics.py       # Chamfer, Hausdorff, IoU
│   └── visualization.py # Plotting utilities
├── notebooks/           # Jupyter notebooks (experiments + figures)
├── data/                # Dataset (not tracked in git)
├── configs/             # Hyperparameter configs
├── tests/               # Unit tests
├── results/             # Experimental results
├── figures/             # Generated figures for thesis
├── docs/                # Planning documents
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Tool |
|---|---|
| Deep Learning | PyTorch |
| Robotics/Kinematics | numpy + scipy + spatialmath-python |
| Optimization (TSP) | PuLP (CBC solver) |
| 3D Reconstruction | OpenCV + Open3D |
| Simulation | CoppeliaSim (Python ZeroMQ API) |
| Control | Custom PID (numpy) |
| Visualization | matplotlib + plotly |

## Quick Start

```bash
pip install -r requirements.txt
```

## Training (Kaggle)

The CNN-Transformer training is designed to run on Kaggle with GPU:
- Upload `models/`, `training/`, `utils/`, and `data/` to a Kaggle dataset
- Run the training notebook from `notebooks/01_train_cnn_transformer.ipynb`

## License

Academic use — PhD thesis project.
