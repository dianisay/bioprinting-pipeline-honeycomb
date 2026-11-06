# Final simulation integration

Run the deterministic, offline computational demo from the repository root:

```powershell
python -m robot.simulation.observable_demo
```

The run writes five input manifests, five case result records, failure receipts,
surface/trajectory data, aggregate metrics, a result table, an architecture SVG,
and MP4 previews under `results/coppelia_demo/`. The `LOW_COVERAGE` and
`MISALIGNED_DEPTH` stress profiles are reused unchanged from
`results/rgb_wound/sensor_generalization/sealed_11_route_replay.ipynb`.

The adapter consumes an observed mesh, boundary, target surface, deposition
height, layer thickness, safety margin, exclusions, support and uncertainty. It
does not require or generate scalar wound volume. The supported case uses a
synthetic externally specified plane, not inferred healthy skin or anatomical
ground truth. The other four cases exercise rejection decisions.

## Important execution boundary

This workspace has the repository's UR5 + XY-gantry kinematic model, but it does
not contain CoppeliaSim, a CoppeliaSim scene, or its remote API client. Therefore
the generated MP4s are labeled **offline computational previews**: they do not
show a CoppeliaSim camera, articulated robot execution, IK tracking, or collision
validation. No `.ttt` scene is generated. `metrics.json` records these items as
not run rather than treating missing tests as passes.

The current planner in `modules/trajectory_planner.py` is not called because it
reduces depth to one shell thickness and maps trajectories only onto a cylinder.
The demonstration's minimum adapter retains target XYZ and per-vertex height,
clips candidate layers to the boundary safety mask, and records the relevant
constraints. This remains a computational prototype, not a replacement planner
or a physical-print interface.

All outputs use synthetic scene-meter coordinates for visualization only. No
metric calibration, clinical threshold, material behavior, physical robot
motion, or printing is validated or authorized.
