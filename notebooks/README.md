# RGB wound reconstruction notebook

The final Kaggle-compatible entry point is
[`04_rgb_wound_3d_reconstruction.ipynb`](./04_rgb_wound_3d_reconstruction.ipynb).
It includes source code, recorded results, manifests, and hashes. The original
execution outputs were large duplicated visualizations; they were removed from
the clean notebook after their hashes and byte counts were recorded in
[`04_replay_manifest.json`](./04_replay_manifest.json). The notebook's embedded
replay data and result hashes remain intact.

## Actions

Set `EXPERIMENT_MODE` and `ACTION` in the configuration cell before running:

- `ACTION="REPLAY"` (default): display sealed results only; it does not rerun
  acquisitions or experiments.
- `ACTION="RECONSTRUCT"`: run the configured classical RGB baseline on local
  dataset inputs.
- `ACTION="EVALUATE_PUBLISHED"`: evaluate identified author-published SALVE
  meshes under the explicit label
  `GT_SIMILARITY_ALIGNED_SHAPE_EVALUATION`. This evaluates shape after
  ground-truth similarity alignment and does not establish independent metric
  scale.
- `ACTION="NATIVE_SALVE"`: native-resolution SALVE CPU/GPU execution when the
  required data, licenses, and runtime are available. Historical interrupted
  CPU attempts remain labeled as incomplete.
- `ACTION="SENSOR_ROUTER"` and `ACTION="SENSOR_GENERALIZATION"`: replay or run
  the frozen synthetic mesh-rendered-depth routes. They are not physical
  sensor validations.
- `ACTION="CALIBRATION_SAFETY"` and
  `ACTION="STRICT_CALIBRATION_HOLDOUT"`: replay the synthetic target and
  calibration-safety evidence. The final strict holdout did not pass its
  engineering gate.
- `ACTION="CLEAN_MEASUREMENT_VALIDITY"`: replay the failed absolute-volume
  validity gate; the unknown healthy-skin closure is the dominant blocker.
- `ACTION="OBSERVABLE_GEOMETRY"`: replay offline, reference-labeled geometry
  and deposition-policy outputs. It does not authorize printing.

Only the recorded `REPLAY` states correspond to sealed final results. Live
actions require the exact declared input bundle and can create new outputs;
do not use them to retune or replace sealed results. Physical printing and
robot commands remain disabled.

The notebook keeps `INDEPENDENT_SCALE_EVALUATION` separate and labels it
`NOT PROVEN` for current SALVE data. Authoritative raw accuracy videos are
under `Files/data_release/Training_Data/raw`; COLMAP-derived transforms and
camera inputs are under `Accuracy/`, `Precision/`, `transforms.json`, and
`sparse/`. The requested Seymour II acquisition is not available from the
authors; GT-aligned shape scores must not be interpreted as automatic metric
scale.

For Kaggle, attach the public dataset inputs and reviewed-mask bundles
described by the notebook's configuration and
[`../docs/dataset_provenance.md`](../docs/dataset_provenance.md). Local
experiment caches and raw datasets are not stored in Git.
