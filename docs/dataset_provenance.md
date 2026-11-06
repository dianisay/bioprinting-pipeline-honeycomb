# Dataset provenance and expected layouts

Raw external datasets are not stored in Git. Respect the source dataset's
access conditions and licenses. The compact evidence bundle records source
paths, roles, and hashes; it is not a substitute for the original data.

## Syn3DWound

- **Origin:** Public synthetic wound dataset available through the CSIRO Data
  Access Portal.
- **Role:** Rendered RGB, shaded masks/labels, cropped meshes and COLMAP
  cameras/poses are used for selected shape comparisons, mesh-rendered-depth
  routes, calibration and geometry studies.
- **Limitations:** No native depth maps or certified physical scale were
  identified. Rendered labels are not clinical truth, and cropped meshes do
  not establish the unknown pre-wound surface.
- **Expected sequence layout:**

```text
data/rgb_wound/syn_audit/Syn3DWound/
  subject_1/
    <sequence>/
      images/
      sparse/
      transforms.json
      ...
```

The exact Figure 3 sequence is
`Syn3DWound_slough-pressure-ulcer-008_sacrum_weak_lighting_phi`; its input
image path and SHA-256 are in `paper/data/figure03_provenance.json`.

## SALVE

- **Origin:** Public silicone wound-phantom data and author-published
  reconstructions.
- **Role:** Evaluation-only comparison of selected author outputs against
  scanner geometry on the PIS4 Accuracy split.
- **Authoritative organization:** Raw accuracy videos are under
  `Files/data_release/Training_Data/raw`; COLMAP-derived data and transforms
  are under `Accuracy/` and `Precision/`; camera parameters are in
  `transforms.json` and `sparse/`.
- The paper's metric scale was obtained by manually aligning its
  reconstruction to the ground-truth point cloud. Accordingly, the reported
  SALVE evaluation is shape-only after GT similarity alignment.
- The requested Seymour II acquisition is not available from the authors.
- **Expected local layout:**

```text
data/rgb_wound/salve/
  Files/data_release/Training_Data/raw/
  Accuracy/
    .../transforms.json
    .../sparse/
  Precision/
    .../transforms.json
    .../sparse/
```

Do not infer that a local result is independently metric-scaled from its
coordinate units or from agreement with GT after alignment.

## Legacy and procedural data

- **FUSeg:** cited in historical model code only; it was not used for the
  completed reconstruction, sensor-routing, or safety results in this
  package. No wound photographs or masks are included.
- **Procedural offline simulation:** generated synthetic sensor inputs,
  target surfaces, traces and preview videos. These are software fixtures,
  not recordings from a camera, robot, patient, or CoppeliaSim.

See the notebook configuration for dataset-specific acquisition and
license acknowledgements. Never commit raw datasets, checkpoints, caches,
or uncurated intermediate result trees.
