# Claims and limitations

## Supported computational claims

- The repository implements and evaluates visible-surface reconstruction,
  synthetic sensor routing, quality/failure gates, and conservative,
  surface-conformal offline planning representations.
- Syn3DWound RGB/mesh comparisons measure shape against dataset geometry in
  dataset units or after evaluation-only similarity alignment. They do not
  establish independent physical scale.
- SALVE comparisons use ground-truth-derived similarity alignment. They are
  shape-only comparisons and do not prove independent metric-scale recovery.
- The final synthetic architecture decision is
  `STAGED_FILL_WITH_OPERATOR_FINAL_TARGET`.
- The five-case offline integration demo is a deterministic software
  simulation. It does not show CoppeliaSim or a physical robot.
- The unified pipeline (`scripts/run_unified_pipeline.py`) connects wound
  identification, the depth-derived surface, the frozen quality route, the
  operator target, the honeycomb lattice, and staged fill in one
  deterministic synthetic run. It shows that the software stages are
  compatible and that the run stops correctly when a stage cannot continue.
  It does not validate wound measurement, toolpath accuracy on a robot, or
  material deposition.
- The patient-level run (`scripts/run_patient_pipeline.py`) applies the same
  chain to a rendered Syn3DWound leg. It shows that the wound masks, cameras,
  and mesh can be combined into a staged honeycomb plan placed on a body
  surface. The geometry is mesh-rendered, the units are dataset units without
  certified physical scale, and the quadratic healthy-ring target is an
  illustrative operator-accepted target, not anatomical truth.
- The first patient-level planner covered only 42.4% of the eligible area
  (22.9% of the wound) and is reported as `PARTIAL`, not as a complete
  scaffold. The revised planner, which clips boundary cells to the eligible
  region, covers 94.2% of the eligible area (50.8% of the wound; the edge
  margin caps this at 53.9%) with zero safety violations. It passes the 90%
  coverage gate (`COMPLETE`). This status refers to footprint coverage only.
  The staged fill stopped with 7.7 mDU² of eligible area still at least one
  layer short, so the depth is not fully filled. Twenty of the 29 cells are
  clipped, non-hexagonal pores whose printability is untested. The simulated
  deposit raises each cell's whole footprint even though the path traces only
  its outline.

## Explicit limitations

- No real D405 wound dataset or physical depth-sensor validation was used.
  Synthetic depth is rendered from source meshes.
- The available observations do not identify the pre-wound healthy surface.
  Healthy-ring extrapolation and computed reference volumes are
  reference-dependent; they are not true wound volume or anatomical
  restoration.
- The clean-measurement validity gate failed: 0 of 24 evaluated variants were
  accepted. The strict calibration engineering gate also failed.
- The sealed healthy-disc TEST contains 55 patches but only one held-out body
  model (`subject_2`). Its 88.32% pointwise uncertainty-envelope coverage is
  below the 90% target. This is a lower-bound test of geometric extrapolation,
  not validation of wound-surface recovery.
- The 18-case final synthetic architecture comparison used two anatomical
  regions and one body model. Direct reference fill failed the predefined
  decision gate. Staged filling still requires an operator-defined final
  target; it is not clinically safe by virtue of the simulation.
- SALVE error values in millimeters are conditioned on GT similarity
  alignment and cannot be reported as independent RGB metric measurements.
- Percentages of wound width, hole diameter, or computed reference volume are
  normalized geometric metrics, not physical measurements.
- Views, pixels, patches, patch-ring records, repeated subsets, and synthetic
  configurations are correlated. No population-level confidence intervals
  are reported.
- No physical printing was performed, no CoppeliaSim robot execution was
  performed, and printing remains disabled.

## Prohibited interpretations

Do not describe the repository as a validated autonomous-healing system.
Do not claim CT-style reconstruction, a true anatomical target, clinically
meaningful wound volume, D405 validation, direct printing instructions, 99%
fill accuracy, or successful physical robot execution. Historical model
components remain in the repository but do not validate those claims.
