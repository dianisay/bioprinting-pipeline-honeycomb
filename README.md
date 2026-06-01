# Robotic Bioprinting Pipeline

**A robot that heals wounds automatically using AI and 3D printing.**

---

## What Does This Do?

Imagine someone has a wound on their skin. This system:

1. **Looks** at the wound with a camera
2. **Understands** where the wound edges are (using AI — a neural network we built from scratch)
3. **Plans** a path for a robot arm to fill the wound with bio-material (like a 3D printer, but for skin)
4. **Moves** the robot arm along that path, printing healing material into the wound

All of this happens automatically — no human needs to control the robot.

---

## How It Works (The 5 Steps)

```
Photo of wound
      ↓
[Step 1] AI finds the wound boundary
      ↓
[Step 2] Build a 3D map of the wound surface
      ↓
[Step 3] Plan a honeycomb filling pattern (like a beehive)
      ↓
[Step 4] Calculate how the robot arm should move
      ↓
[Step 5] Robot prints bio-material into the wound
```

---

## What's Special About This?

We invented a new way for the AI to describe wound shapes: **polar coordinates**.

Instead of saying "the wound is at these 64 random points," we say:
- "The center is HERE"
- "The edge is THIS far away in each direction"

This guarantees the boundary is always a clean, closed shape — no gaps, no crossings.

---

## Project Structure

```
diana-bioprinting-pipeline/
│
├── models/                  ← The AI brain (neural networks)
│   ├── encoder.py           ← Looks at the image (ResNet-50 + Transformer)
│   ├── polar_decoder.py     ← Our invention: predicts wound boundary
│   ├── detr_decoder.py      ← Alternative method (for comparison)
│   └── autoregressive_decoder.py  ← Another alternative (for comparison)
│
├── modules/                 ← The robotics brain
│   ├── stl_analysis.py      ← Reads 3D scaffold shapes
│   ├── honeycomb.py         ← Creates honeycomb fill patterns
│   ├── conformal_mapping.py ← Maps flat patterns onto curved surfaces
│   ├── tsp_solver.py        ← Finds the shortest path between cells
│   ├── trajectory_planner.py← Puts it all together into a robot path
│   ├── robot_model.py       ← Describes our 8-joint robot arm
│   ├── inverse_kinematics.py← Figures out joint angles from positions
│   └── visualization_3d.py  ← Makes pretty 3D plots
│
├── training/                ← Scripts to train the AI
│   ├── train.py             ← Train one model
│   ├── evaluate.py          ← Test how good the model is
│   └── ablation.py          ← Compare all 3 methods fairly
│
├── notebooks/               ← Interactive demos (Jupyter)
│   ├── 00_demo.ipynb        ← START HERE — quick visual demo
│   └── 01_ablation_study_kaggle.ipynb  ← Full training (run on Kaggle GPU)
│
├── data/                    ← Input data (wound images, 3D meshes)
├── tests/                   ← Automated tests (verify everything works)
├── results/                 ← Output (trained models, figures, metrics)
├── figures/                 ← Generated images for the thesis
├── configs/                 ← Settings and hyperparameters
└── requirements.txt         ← Python packages needed
```

---

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/dianisay/diana-bioprinting-pipeline.git
cd diana-bioprinting-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo notebook
jupyter lab notebooks/00_demo.ipynb
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

## Training the AI (on Kaggle)

The neural network needs a GPU to train efficiently. We use Kaggle (free GPUs):

1. Upload this repo to Kaggle as a dataset
2. Open `notebooks/01_ablation_study_kaggle.ipynb`
3. Enable GPU accelerator
4. Run all cells (~2-3 hours)
5. Download results

---

## Who Made This?

**Diana Paola Ayala Roldán**
PhD in Computational Sciences, Tecnológico de Monterrey (2026)

This is the code behind the thesis:
*"CNN-Transformer-Based Machine Learning for 3D Motion Planning and Control in In-Situ Robotic Bioprinters for Superficial Tissue Regeneration"*

---

## License

Academic use — PhD thesis project.
