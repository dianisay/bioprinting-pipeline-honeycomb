# Unified offline pipeline

This document explains how the repository's separate parts form one pipeline,
how to run it, and what it does **not** show.

```powershell
python scripts/run_unified_pipeline.py
python -m pytest tests/test_unified_pipeline.py -q
```

The command processes the five frozen synthetic cases in
`robot/simulation/demo_config.json`. It writes these outputs to
`results/unified_pipeline/`:

- `summary.json`: for every case, which stages it passed, where it stopped,
  and why.
- `supported_case_toolpath.csv`: the staged honeycomb toolpath preview for
  case A.
- `unified_pipeline.png` / `.svg`: a six-panel figure plotted from those same
  values.

## The seven stages in plain language

A wound passes through the stages in order. Any stage can stop the run. A
stop is a valid outcome: it means "this step lacks enough information to
continue safely," not that the program crashed.

| # | Stage | Question it answers | Repository code it uses |
|---|---|---|---|
| 1 | OBSERVE | What does the camera see? It captures a colour image, plus a depth image when available. | `robot/simulation/observable_demo._sensor_observation` (the frozen synthetic sensor profiles) |
| 2 | IDENTIFY | Which pixels belong to the wound? | `pipeline.unified.identify_wound` (stands in for the notebook's wound masks) |
| 3 | SURFACE | How deep is the wound bed at each pixel? | `pipeline.unified.surface_from_depth`; in real data, this is the reconstruction from `notebooks/04_rgb_wound_3d_reconstruction.ipynb` |
| 4 | ROUTE | Is the measurement good enough to use? It checks resolution, depth coverage, noise, and RGB-depth alignment. | `observable_demo._route` (unchanged frozen quality gates) |
| 5 | TARGET | How high should the fill go? | It requires an operator-defined final target, because the sealed decision `STAGED_FILL_WITH_OPERATOR_FINAL_TARGET` rules out automatic inference. |
| 6 | HONEYCOMB | Where should the nozzle trace the cell walls? | `modules/honeycomb.create_hex_grid` and `hexagon_perimeter` (the Python translation of `MuffinFresa_ConformalMapping.m`) and `modules/conformal_mapping.compute_nozzle_orientations` |
| 7 | STAGED FILL | How much should be added now? It fills at most one layer, below the lower uncertainty envelope, then re-measures and re-plans. | `pipeline.unified.staged_honeycomb_fill` |

### What changed compared with the older pieces

- **The honeycomb now follows the measured wound surface instead of a
  cylinder.** The original planner (`modules/trajectory_planner.py`) mapped
  the hexagon lattice onto a known cylindrical scaffold. In the unified
  pipeline, each hexagon's wall height comes from the measured wound-bed
  height plus that stage's planned layer. The nozzle direction also comes from
  a plane fitted to the measured surface inside each cell. That is what
  *conformal* means here: the path follows the shape it is printing on.
- **Only whole cells inside the safe region are used.** A cell is used only
  if every one of its pixels has valid depth, lies at least the safety margin
  from the wound edge, avoids exclusion zones, and has low enough
  uncertainty. Cells that touch the edge are skipped. The skipped rim is
  reported as unresolved area instead of being guessed.
- **The fill is staged.** Each stage deposits at most one layer and never
  fills above the lower edge of the uncertainty envelope. After each stage,
  the simulated surface is re-measured and the next plan is built from that
  new measurement. The hidden synthetic truth is used only to score overfill
  afterward.
- **Two observable stopping rules end the fill:**
  `SUPPORTED_DEPTH_BELOW_ONE_LAYER` (less than one layer remains where
  filling is supported) and `NO_SAFE_SUPPORTED_REGION` (depth remains, but
  no whole honeycomb cell can be filled safely). In both cases, the plan is
  handed to an operator for review.

## Result on the five frozen cases

| Case | Stops at | Decision |
|---|---|---|
| A, supported RGB-D | STAGED_FILL | `PLAN_READY_FOR_OPERATOR_REVIEW_PRINT_DISABLED`: 3 stages, 23 honeycomb cells, 0 boundary violations, 0 overfill, then stops with `NO_SAFE_SUPPORTED_REGION` because an unfilled rim remains outside the whole cells |
| B, small boundary | ROUTE | `BOUNDARY_RESOLUTION_INSUFFICIENT` |
| C, incomplete depth | ROUTE | `SURFACE_INCOMPLETE` |
| D, RGB-depth misaligned | ROUTE | `RGB_DEPTH_CALIBRATION_FAILED` |
| E, reference disagreement | ROUTE | `EXTERNAL_TARGET_REQUIRED` |

If case A is run without an operator target, it stops at TARGET with
`OPERATOR_FINAL_TARGET_REQUIRED`. The tests in
`tests/test_unified_pipeline.py` check this behavior.

## Limits

- All inputs are synthetic. Units are synthetic scene metres at the
  renderer's declared scale; metric scale is not independently recovered.
- The colour-threshold segmenter works only on the synthetic phantom image.
  Real images require the notebook's masks.
- On this synthetic cavity, the honeycomb lattice covers about 55% of the
  wound area, and about 13% of the wound remains more than one layer below
  the target. That rim is reported, not filled.
- The toolpath is a Cartesian preview. No inverse kinematics, collision
  check, CoppeliaSim run, robot command, or physical printing is involved.
  `print_authorized` is always `false`.
- The legacy neural path (`scripts/demo_pipeline.py`, with an untrained
  encoder/decoder mapped onto a cylinder) is kept as historical code. It is
  not part of this pipeline.

## Patient-level run

`python scripts/run_patient_pipeline.py` carries the same chain onto a
rendered human leg (Syn3DWound `subject_1/Syn3DWound_0415_leg_strong_lighting_phi`).
The code is in `pipeline/patient.py`.

1. **Observe.** Read 40 rendered photographs, their wound masks, the COLMAP
   cameras, and the dataset's cropped leg mesh (11,087 points, 21,692
   triangles). The mesh is stored y-up, so each point (x, y, z) becomes
   (x, -z, y) to match the camera poses.
2. **Identify in 3D.** Project every mesh point into every photograph. Count
   a vote when the point faces that camera and its pixel in the mask is
   brighter than 40/255. A point is wound if more than half of the views that
   see it vote yes. Keep the largest connected wound patch (153 points).
3. **Surface.** Fit a plane through the skin within 2.5 wound radii (PCA).
   Its normal points toward the cameras, and the first in-plane axis follows
   the wound's long axis. Resample the mesh as a height map with 0.1 mDU
   pixels by linear interpolation.
4. **Target.** Fit a quadratic surface to a healthy ring, up to 0.4 wound
   radii outside the wound edge. It stands in for the operator's target and is
   not anatomical truth.
5. **Coverage audit.** Label every height-map pixel as outside the wound,
   edge margin (the outer 10% of half the wound width), no support,
   uncertainty gate, under one layer of room, or eligible. The rules are the
   same as before; the labels only show where each rule applies.
6. **Honeycomb.** Search 480 lattices: five hexagon sides from 0.78 mDU down
   to 0.40 mDU (two strand widths, the smallest printable side), six
   rotations (0-50 degrees), and 16 staggered offsets. Three planners are
   evaluated separately. `WHOLE_CELL` keeps whole hexagons that are fully
   eligible. `ADAPTIVE_WHOLE_CELL` adds smaller whole hexagons at the edge.
   `CLIPPED_BOUNDARY` also cuts boundary hexagons to the eligible region.
   A clipped piece must survive an opening by a disk two strands wide, cover
   at least a quarter of a hexagon, and stay one piece after a 1-pixel
   erosion. The software picks the first planner, in that order, that has
   zero violations and covers at least 90% of the eligible area (`COMPLETE`).
   Otherwise the plan is `PARTIAL`, or `REJECTED` if any violation occurs.
7. **Staged fill.** The chosen cells stay fixed. In each stage, a cell
   receives one whole layer (0.2 mDU) only if every pixel in it still has at
   least one layer of room below `target - uncertainty`. Re-measure and
   repeat.
8. **Place.** Map the cells and the nozzle path (nearest neighbour, then
   2-opt) back to leg coordinates, and project the wound boundary, the
   eligible boundary, the cells, and the uncovered eligible area into the
   photograph.

Result in our copy (`results/patient_pipeline/coverage_planners.csv`):

| Planner | Cells | Eligible coverage | Wound coverage | Status |
|---|---|---|---|---|
| First planner (one fixed lattice) | 8 | 42.4% | 22.9% | PARTIAL |
| Whole cells | 51 | 72.6% | 39.1% | PARTIAL |
| Adaptive whole cells | 51 | 72.6% | 39.1% | PARTIAL |
| Clipped boundary (selected) | 29 (9 whole, 20 clipped) | 94.2% | 50.8% | COMPLETE |

The first planner's 42 rejected cells all failed because the complete hexagon
did not fit inside the safe region. None failed for uncertainty or missing
support. The edge margin caps wound coverage at 53.9%. Every plan had zero
deposits outside the wound, in the margin, on unsupported pixels, or above the
lower envelope, and zero path crossings. The selected plan filled 29 cells in
stage 1 and 20 in stage 2, then stopped with `NO_SAFE_SUPPORTED_REGION`.
`COMPLETE` refers to the footprint coverage gate only: 7.7 mDU² of eligible
area is still at least one layer below the lower envelope, so the depth is not
finished. Whole cells cannot reach the gate because their best lattice already
uses the smallest printable side. Units are Syn3DWound dataset units, and
physical scale is not certified.
