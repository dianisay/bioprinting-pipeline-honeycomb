# Statistical and metric audit

This audit records the effective units and limitations for the published
descriptive results. The source of machine-readable values is
[`../paper/data/publication_evidence.json`](../paper/data/publication_evidence.json);
per-metric units, denominators, aggregation, data origin, and interval decisions
are in [`../paper/tables/metric_audit.csv`](../paper/tables/metric_audit.csv).

## Evaluation counts and independent units

| Evaluation | Count | Effective unit and dependency |
|---|---:|---|
| Syn3DWound inventory | 52 sequences inventoried | Only explicitly described subsets were evaluated; inventory size is not an evaluated sample count. |
| Focused sensor routing | 252 route/subset results across 4 sequence contexts | Route/subset records share wound sequences and source meshes; they are not independent samples. Per-route passes, failures, and total subsets are in Table 2. |
| SALVE shape comparison | 50 common frames, 2 devices, 2 selected published reconstructions | One PIS4 phantom split; point clouds/frames are repeated measurements. All reported mm metrics use GT-derived similarity alignment. |
| Clean measurement validity | 4 wound sequences; 24 evaluated variants; 0 accepted | Variants are deterministic perturbations/subsets, not 24 independent wounds. |
| Strict calibration holdout | 48 clean, 168 fault, 24 design-stress configurations; 480 diagnostic subsets | Configuration-level reporting uses the worst of two disjoint 20-view subsets. Five subsets were unavailable in the threshold-recall evaluation. |
| Healthy-disc hidden-surface TEST | 55 patches; 330 patch-ring records; 297 valid global-model records | One held-out body model (`subject_2`); 33 patch-ring/model records are invalid/unavailable. Query points and patches are correlated. |
| Final architecture decision | 18 carved cases: 2 regions × 3 depths × 3 edge conditions | Two anatomical regions on one available body model; six cases per edge-deformation level. |
| Offline integration demo | 5 cases; 1 three-stage trace | Deterministic software cases, not physical trials. |

## Aggregation and units

- Surface, boundary, depth, hidden-disc, and repeatability errors are expressed
  as fractions or percentages of wound width or hole diameter, as indicated in
  the data and figure axes.
- Reference-volume errors are percentages of a computed synthetic reference
  volume; they are not true wound-volume errors.
- SALVE mean/P95 distances are reported in millimeters only after
  GT-similarity alignment. They are shape-only values, not independent metric
  recovery.
- Configuration recalls and false-alarm rates are descriptive numerators over
  denominators; the underlying configurations are not independent patients or
  wounds.
- Unavailable metric denominators are marked as not retained in the metric
  audit rather than inferred from pass/fail counts.
- No inferential confidence intervals are reported. The effective independent
  wound/body-model counts are small and repeated views, pixels, subsets,
  patches, and synthetic configurations are correlated.
