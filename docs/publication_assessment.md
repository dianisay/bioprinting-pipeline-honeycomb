# Publication-readiness assessment

## Status: `PUBLICATION_READY_WITH_WRITING_GAPS`

The repository now contains a hash-linked evidence bundle, nine generated
figures (300-DPI PNG and SVG), five machine-readable/manuscript-ready tables,
a metric audit, an artifact inventory, a clean replay notebook, and explicit
claims, provenance, and reproduction documentation. The evidence is limited
to computational evaluations and offline synthetic demonstrations.

| Gate | Status | Evidence |
|---|---|---|
| Focused computational question | Pass with scope note | Whether the available observations support reference-dependent direct fill or require conservative staged planning with an operator target. |
| Explicit computational scope | Pass | No clinical, physical sensor, robot, or printing validation is claimed. |
| Dataset provenance | Pass | Syn3DWound, SALVE, legacy FUSeg, and procedural fixtures are separated in Table 1 and `docs/dataset_provenance.md`. |
| Reproducible final configurations | Partial | Evidence and output-generation scripts are committed; Figure 3 needs its identified public RGB example, and fresh evidence export needs sealed local result files. |
| Baselines and ablations | Pass for reported scope | Existing reference models, sensor routes, and calibration target ablations are retained and labeled. |
| Independent failures and rejection behavior | Pass | Failed calibration/measurement gates and refused synthetic cases are included. |
| Uncertainty and rejection behavior | Partial | Hidden-disc envelope coverage is reported as 88.32%, below target; the coverage denominator is not retained in the compact evidence export. |
| Statistical reporting | Partial | Clustered units and reasons for omitting inferential intervals are documented. Some original per-point counts are unavailable in the compact export and explicitly marked unavailable in `paper/tables/metric_audit.csv`. |
| Publication figures and tables | Pass | Nine figures, five tables in CSV/Markdown/LaTeX, and generation scripts are included. |
| Exact reproduction instructions | Pass with data requirement | `docs/reproduction.md` identifies commands, expected outputs, sealed inputs, and public-data requirements. |
| Manuscript and references | Writing gap | The draft has been rewritten to match the evidence, but a source-verified literature synthesis, venue formatting, and final author review remain. |

## Concrete remaining writing gaps

1. Complete a source-verified literature review and integrate only checked
   citations into `paper/main.tex`.
2. Obtain editorial/author review of the focused question, tables, figure
   captions, and the disclosed denominator limitations.
3. Decide the target venue and apply its final manuscript and supplementary
   material format.

These are publication-writing tasks, not requests for additional technical
experiments. No new experiment is proposed.
