# Reproduction guide

## Scope and reproducibility modes

The committed package reproduces the curated figures and tables from the
hash-linked `paper/data/publication_evidence.json` bundle. The evidence export
script is a separate step: it validates existing sealed local experiment
outputs and must not be used to rerun or retune completed experiments.

Use Python 3.10 or later from the repository root. In PowerShell:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/build_publication_package.py --replay
```

`--replay` regenerates the nine figures, their figure data and provenance,
the five CSV/Markdown/LaTeX tables, the metric audit, the artifact inventory,
and the selected-results HTML. It reads committed evidence rather than
recomputing experimental metrics. Figure 3 needs one public RGB file from
Syn3DWound (DOI [10.25919/5rwz-ts17](https://doi.org/10.25919/5rwz-ts17)).
The CSIRO record declares CC BY-NC-SA 4.0 and requires the attribution
recorded in `paper/data/figure03_provenance.json`. Because the license is
noncommercial, confirm compatibility or obtain permission before using the
image in a commercial publication. A public file-specific
endpoint makes selective download possible; the script requests only that
image, verifies its byte count and SHA-256, and does not download the dataset.
The source image is excluded from Git.

From the repository root, run these exact commands:

```powershell
python scripts/fetch_figure3_example.py
python scripts/build_publication_package.py --replay
```

The exported surface arrays, reviewed mask, error map and route data needed
by Figure 3 are already in `paper/data/`.

To re-export the compact evidence bundle, the original sealed local results
must be present at their manifest paths:

```powershell
python scripts/export_publication_evidence.py
python scripts/build_publication_package.py
```

The exporter fails on missing or changed seal inputs. The raw datasets and
sealed `results/rgb_wound/` outputs are intentionally not committed. Do not
replace a failed hash check with a fresh run and call it a replay.

## Offline demonstration

The deterministic procedural demonstration is generated with:

```powershell
python -m robot.simulation.observable_demo
```

It writes the five synthetic case records, traces, generated inputs, metrics,
and preview videos beneath `results/coppelia_demo/`. The command does not
launch CoppeliaSim, robot control, or deposition. All command authority and
printing remain disabled. These generated records are distinct from the
sealed 18-case architecture decision in the publication evidence.

## Unified pipeline

```powershell
python scripts/run_unified_pipeline.py
python -m pytest tests/test_unified_pipeline.py -q
```

This command chains observation, wound identification, the depth-derived
surface, routing, the operator target, the honeycomb lattice, and staged fill
into one run over the same five synthetic cases. It writes `summary.json`,
`supported_case_toolpath.csv`, and `unified_pipeline.png`/`.svg` to
`results/unified_pipeline/`. See [unified_pipeline.md](unified_pipeline.md).
Printing remains disabled.

## Patient-level pipeline figure

```powershell
python scripts/run_patient_pipeline.py
python -m pytest tests/test_patient_pipeline.py -q
```

This run needs the public Syn3DWound sequence
`subject_1/Syn3DWound_0415_leg_strong_lighting_phi` under
`data/rgb_wound/syn3dwound_leg/` (layout in
[dataset_provenance.md](dataset_provenance.md#syn3dwound-leg-sequence-for-the-patient-figure)).
Without it, the script exits with code 2 and names the missing files, and
the end-to-end test is skipped. The script writes `patient_pipeline.png`
(300 dpi) and `.pdf`, `summary.json`, `scaffold_toolpath_world.csv`,
`provenance.json` (input SHA-256 hashes and licence), and `caption.md` to
`results/patient_pipeline/`. It also writes the honeycomb coverage audit:
`coverage_audit.png`/`.pdf` (coverage maps and planner search),
`coverage_audit.json` (pixel categories, rejected-cell reasons, planner
results), `coverage_search.csv` (all 480 searched lattices), and
`coverage_planners.csv` (one row per planner). The run takes about a minute.
It is deterministic (seed 0). Printing remains disabled.

## Notebook

Open `notebooks/04_rgb_wound_3d_reconstruction.ipynb`. Its default
`ACTION="REPLAY"` is the view-only path for recorded results. Other actions
require exact external inputs and may create new outputs; they are not needed
to reproduce the committed publication package. See
[`../notebooks/README.md`](../notebooks/README.md) for action details.

## Expected outputs

- `paper/figures/figure_01` through `figure_09`: 300-DPI PNG, SVG, and
  provenance JSON.
- `paper/figure_data/figure_01.json` through `figure_09.json`: plotted values.
- `paper/tables/table_01_*` through `table_05_*`: CSV, Markdown, and LaTeX.
- `paper/tables/metric_audit.csv`: metric units, aggregation, denominator
  availability, data origin, and confidence-interval decisions.
- `docs/artifact_inventory.csv`: inventory and commit-safety classification.
- `notebooks/04_selected_results.html`: image-linked results gallery.

No raw data, model checkpoints, or caches are required for tests over analytic
fixtures. Public dataset actions are separate from this package replay.
