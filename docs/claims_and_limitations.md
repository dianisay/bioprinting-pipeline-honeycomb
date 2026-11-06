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
