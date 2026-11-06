| Case | Expected | Actual | Stages | Boundary | Exclusion | Unsupported fraction | Planning status | Match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_supported_rgbd | RECONSTRUCTION_ACCEPTED_FOR_SIMULATION | RECONSTRUCTION_ACCEPTED_FOR_SIMULATION | 3 | 0 | 0 | 0.0 | SIMULATION_PLAN_COMPLETE | True |
| B_boundary_resolution | BOUNDARY_RESOLUTION_INSUFFICIENT | BOUNDARY_RESOLUTION_INSUFFICIENT | 0 | 0 | 0 | 0.0 | REFUSE_TO_PLAN | True |
| C_incomplete_depth | SURFACE_INCOMPLETE | SURFACE_INCOMPLETE | 0 | 0 | 0 | 0.0 | NO_SAFE_DEPOSITION_REGION | True |
| D_rgb_depth_misalignment | RGB_DEPTH_CALIBRATION_FAILED | RGB_DEPTH_CALIBRATION_FAILED | 0 | 0 | 0 | 0.0 | REFUSE_TO_PLAN | True |
| E_reference_disagreement | EXTERNAL_TARGET_REQUIRED | EXTERNAL_TARGET_REQUIRED | 0 | 0 | 0 | 0.0 | REFUSE_TO_PLAN | True |
