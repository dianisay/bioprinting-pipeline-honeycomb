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
recomputing experimental metrics. Figure 3 also reads the public RGB example
named and hash-identified in `paper/data/figure03_provenance.json`; place that
file in its declared Syn3DWound location first. The exported surface, mask,
error-map and route data needed by Figure 3 are in `paper/data/`.

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
