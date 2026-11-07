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

### Syn3DWound leg sequence for the patient figure

`scripts/run_patient_pipeline.py` uses
`Syn3DWound/subject_1/Syn3DWound_0415_leg_strong_lighting_phi` (same DOI and
CC BY-NC-SA 4.0 licence). Download that one sequence folder from the CSIRO
collection and arrange it as:

```text
data/rgb_wound/syn3dwound_leg/
  cropped_mesh.obj
  BB.txt
  images/image_XXX.png          # 4000 x 2160 rendered photographs
  mask/mask_XXX.png             # shaded wound masks; wound pixels > 40/255
  jamal_0415_leg_strong_lighting_phi_colmap/
    .../cameras.txt             # SIMPLE_PINHOLE intrinsics
    .../images.txt              # world-to-camera poses, one per image
```

The script uses every image that has both a photograph and a mask (40 in our
copy). The SHA-256 of every input file is written to
`results/patient_pipeline/provenance.json`.

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
