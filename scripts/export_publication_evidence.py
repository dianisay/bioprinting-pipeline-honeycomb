"""Export a compact, hash-linked evidence bundle from sealed local results."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
NOTEBOOK = ROOT / "notebooks" / "04_rgb_wound_3d_reconstruction.ipynb"
RESULTS = ROOT / "results" / "rgb_wound"
DEMO = ROOT / "results" / "coppelia_demo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required sealed evidence file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_literal(cell_index: int, variable: str) -> Any:
    document = read_json(NOTEBOOK)
    source = "".join(document["cells"][cell_index].get("source", []))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{variable} was not found in notebook cell {cell_index}.")


def source_record(path: Path, role: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot record missing source: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "role": role,
    }


def _verify_hidden_disc_seal() -> dict[str, Any]:
    root = RESULTS / "healthy_skin_hidden_disc_v3"
    hashes = read_json(root / "manifest_hashes.json")["hashes"]
    for name, expected in hashes.items():
        actual = sha256(root / name)
        if actual != expected:
            raise ValueError(f"Hidden-disc v3 sealed hash mismatch: {name}")
    lock = read_json(root / "test_prediction_freeze.json")
    for key, path in (
        ("test_manifest_sha256", root / "test_manifest_PUBLIC.json"),
        ("frozen_decisions_sha256", root / "frozen_decisions.json"),
        ("test_predictions_sha256", root / "test_predictions.json"),
    ):
        if sha256(path) != lock[key]:
            raise ValueError(f"Hidden-disc TEST freeze mismatch: {path.name}")
    if lock["chosen_global_method"] != "thin_plate_spline":
        raise ValueError("The frozen hidden-disc global method changed.")
    return {
        "schema": lock["schema"],
        "global_method": lock["chosen_global_method"],
        "manifest_sha256": lock["test_manifest_sha256"],
        "decision_sha256": lock["frozen_decisions_sha256"],
        "prediction_sha256": lock["test_predictions_sha256"],
        "heldout_test_truth_unlocked": lock["heldout_test_truth_unlocked"],
        "all_manifest_hashes_verified": True,
    }


def export_hidden_disc_example() -> dict[str, Any]:
    root = RESULTS / "healthy_skin_hidden_disc_v3"
    public = read_json(root / "test_manifest_PUBLIC.json")
    truth = read_json(root / "test_truth_PRIVATE.json")
    predictions = read_json(root / "test_predictions.json")
    decisions = read_json(root / "frozen_decisions.json")
    patch_id = "subject_2_shoulder_d16cd16a_percent_20_circle_00"
    ring_width = 0.20
    pub = next(row for row in public["patches"] if row["id"] == patch_id)
    heldout = next(row for row in truth["patches"] if row["id"] == patch_id)
    pred = next(
        row for row in predictions["predictions"]
        if row["id"] == patch_id and float(row["ring_width_fraction"]) == ring_width
    )
    rings = pub["ring_widths"][str(ring_width)]
    width_key = str(ring_width)
    truth_z = np.asarray(heldout["ring_truth_z"][width_key], dtype=float)
    truth_indices = np.asarray(heldout["ring_truth_query_indices"][width_key], dtype=int)
    query_xy = np.asarray(rings["query_xy"], dtype=float)[truth_indices]
    method_names = decisions["uncertainty_ensemble"]["methods"]
    method_predictions = {
        name: np.asarray(pred["methods"][name]["prediction"], dtype=float)[truth_indices]
        for name in method_names
    }
    center = np.median(np.stack(list(method_predictions.values())), axis=0)
    half_width = (
        np.max(np.stack(list(method_predictions.values())), axis=0)
        - np.min(np.stack(list(method_predictions.values())), axis=0)
    ) / 2.0
    scale = float(decisions["uncertainty_ensemble"]["envelope_scale"])
    sample = {
        "schema": "hidden-disc-test-example-v3",
        "patch_id": patch_id,
        "split": "TEST",
        "body_model": heldout["body_model"],
        "anatomical_region": heldout["anatomical_region"],
        "hole_spec": pub["hole_spec"],
        "ring_width_fraction": ring_width,
        "ring_xy": rings["ring_xy"],
        "ring_z": rings["ring_z"],
        "query_xy": query_xy.tolist(),
        "heldout_truth_z": truth_z.tolist(),
        "tps_prediction_z": (
            np.asarray(pred["methods"]["thin_plate_spline"]["prediction"], dtype=float)[truth_indices]
            .tolist()
        ),
        "ensemble_center_z": center.tolist(),
        "envelope_low_z": (center - scale * half_width).tolist(),
        "envelope_high_z": (center + scale * half_width).tolist(),
        "ensemble_methods": method_names,
        "envelope_scale_frozen_on_selection": scale,
        "evaluation": "Frozen TEST illustration; local dataset units, no millimeter interpretation.",
        "manifest_sha256": sha256(root / "test_manifest_PUBLIC.json"),
        "prediction_sha256": sha256(root / "test_predictions.json"),
        "truth_sha256": sha256(root / "test_truth_PRIVATE.json"),
        "decision_sha256": sha256(root / "frozen_decisions.json"),
    }
    output = DATA / "figure08_hidden_disc_example.json"
    output.write_text(json.dumps(sample, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return sample


def export_reconstruction_example() -> dict[str, str]:
    import open3d as o3d

    route = (
        RESULTS / "sensor_generalization" / "runs" / "sacrum_phi"
        / "HIGH_RES_RGB_PLUS_DEPTH_A"
    )
    source_image = (
        ROOT / "data" / "rgb_wound" / "syn_audit" / "Syn3DWound" / "subject_1"
        / "Syn3DWound_slough-pressure-ulcer-008_sacrum_weak_lighting_phi"
        / "images" / "image_000.png"
    )
    mask = RESULTS / "sensor_generalization" / "blinded_masks" / "sacrum_phi" / "image_000.png"
    recovered_path = route / "wound_surface_DATASET_UNITS.ply"
    reference_path = route / "complete_local_surface_DATASET_UNITS.ply"
    error_path = route / "error_depth_boundary.png"
    for path in (source_image, mask, recovered_path, reference_path, error_path):
        if not path.is_file():
            raise FileNotFoundError(f"Figure 3 input is missing: {path}")

    recovered = o3d.io.read_triangle_mesh(str(recovered_path))
    reference = o3d.io.read_triangle_mesh(str(reference_path))
    if not len(recovered.vertices) or not len(reference.vertices):
        raise ValueError("Figure 3 source meshes are empty.")
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA / "figure03_surface_data.npz",
        recovered_vertices=np.asarray(recovered.vertices),
        recovered_faces=np.asarray(recovered.triangles),
        reference_vertices=np.asarray(reference.vertices),
        reference_faces=np.asarray(reference.triangles),
    )
    shutil.copyfile(mask, DATA / "figure03_reviewed_mask.png")
    shutil.copyfile(error_path, DATA / "figure03_error_heatmap.png")
    return {
        "input_image": source_image.relative_to(ROOT).as_posix(),
        "input_image_sha256": sha256(source_image),
        "reviewed_mask_source": mask.relative_to(ROOT).as_posix(),
        "reviewed_mask_sha256": sha256(mask),
        "recovered_mesh_source": recovered_path.relative_to(ROOT).as_posix(),
        "recovered_mesh_sha256": sha256(recovered_path),
        "reference_mesh_source": reference_path.relative_to(ROOT).as_posix(),
        "reference_mesh_sha256": sha256(reference_path),
        "error_map_source": error_path.relative_to(ROOT).as_posix(),
        "error_map_sha256": sha256(error_path),
        "exported_mesh_data": "paper/data/figure03_surface_data.npz",
    }


def build_evidence() -> dict[str, Any]:
    notebook = notebook_literal(8, "RECORDED_RESULTS")
    gen_path = RESULTS / "sensor_generalization" / "analysis" / "summary.json"
    generalization = read_json(gen_path)
    calibration_path = RESULTS / "calibration_safety" / "v2" / "analysis" / "summary.json"
    calibration = read_json(calibration_path)
    strict_path = RESULTS / "strict_calibration_holdout" / "analysis" / "summary.json"
    strict = read_json(strict_path)
    clean_path = RESULTS / "clean_measurement_validity" / "summary.json"
    clean = read_json(clean_path)
    ref_path = (
        RESULTS / "clean_measurement_validity" / "sequences" / "sacrum_phi"
        / "reference_ring_ablation.json"
    )
    reference_models = read_json(ref_path)
    hidden_path = RESULTS / "healthy_skin_hidden_disc_v3" / "final_report.json"
    hidden = read_json(hidden_path)
    decision_path = RESULTS / "final_simulation_decision_v2" / "decision_report.json"
    architecture = read_json(decision_path)
    demo_path = DEMO / "metrics.json"
    demo = read_json(demo_path)
    demo_trace_path = DEMO / "closed_loop_trace.json"
    demo_trace = read_json(demo_trace_path)

    focus_routes = {
        "RGB_ONLY_60_VIEW",
        "GENERIC_ALIGNED_RGBD",
        "HIGH_RES_RGB_PLUS_DEPTH",
        "LOW_DEPTH_COVERAGE",
        "HIGH_DEPTH_NOISE",
        "MISALIGNED_DEPTH",
    }
    route_rows = [row for row in generalization["aggregate"] if row["route"] in focus_routes]
    if {row["route"] for row in route_rows} != focus_routes:
        raise ValueError("Focused route evidence is incomplete.")
    sensitivity_names = {
        "GENERIC_ALIGNED_RGBD", "OFFSET_0px", "OFFSET_1px", "OFFSET_2px",
        "OFFSET_4px", "OFFSET_8px", "MISALIGNED_DEPTH",
    }
    sensitivity_routes = [
        row for row in generalization["aggregate"] if row["route"] in sensitivity_names
    ]

    thresholds = []
    for basis in ("total", "incremental"):
        for threshold, record in strict["volume_recall"][basis].items():
            thresholds.append(
                {
                    "evaluation": "Strict holdout, " + basis + " error",
                    "threshold": threshold,
                    "numerator": record["detected_configurations"],
                    "denominator": record["dangerous_configurations"],
                    "rate": record["recall"],
                    "unit": "fraction of dangerous configurations detected",
                    "aggregation": "configuration-level; worst of two disjoint 20-view subsets",
                    "unavailable_subsets": record["measurement_unavailable_subsets"],
                }
            )
    target_summary = calibration["target_detector_summary"]
    nonclean_target_configurations = (
        calibration["heldout_configuration_count"] - calibration["heldout_clean_count"]
    )
    target_rows = [
        {
            "target": "stepped",
            "fault_rejections": round(
                target_summary["fault_rejection_rate_nonclean"] * nonclean_target_configurations
            ),
            "fault_configurations": nonclean_target_configurations,
            "fault_rejection_rate": target_summary["fault_rejection_rate_nonclean"],
            "clean_false_alarms": target_summary["clean_false_alarm_count"],
            "clean_controls": target_summary["clean_test_count"],
            "clean_false_alarm_rate": target_summary["clean_false_alarm_rate"],
        },
        {
            "target": "planar",
            "fault_rejections": round(
                target_summary["planar_fault_rejection_rate_nonclean"] * nonclean_target_configurations
            ),
            "fault_configurations": nonclean_target_configurations,
            "fault_rejection_rate": target_summary["planar_fault_rejection_rate_nonclean"],
            "clean_false_alarms": target_summary["planar_clean_false_alarm_count"],
            "clean_controls": target_summary["clean_test_count"],
            "clean_false_alarm_rate": target_summary["planar_clean_false_alarm_rate"],
        },
    ]

    selected_references = []
    method_names = sorted({row["method"] for row in reference_models})
    for method in method_names:
        rows = [
            row for row in reference_models
            if row["method"] == method and float(row["ring_width"]) == 0.20
            and not row["incomplete"]
        ]
        if not rows:
            continue
        row = rows[0]
        selected_references.append(
            {
                "method": method,
                "ring_width_fraction": 0.20,
                "healthy_samples": row["healthy_samples"],
                "blocked_holdout_rmse_dataset_units": row["blocked_holdout_rmse"],
                "reference_defined_volume_dataset_units3": row["measurements"]["volume"],
                "extrapolation_z_range_dataset_units": row["extrapolation_z_range"],
                "boundary_reference_range_dataset_units": row["boundary_reference_range"],
                "boundary_continuity_rmse_dataset_units": row["boundary_continuity_rmse"],
                "healthy_ring_only": row["healthy_ring_only"],
                "wound_bed_used_for_fit": row["wound_bed_used_for_fit"],
            }
        )

    clean_fixed = clean["fixed_closure_surface_volume_error_by_sequence"]
    def numeric_values(value: Any):
        if isinstance(value, dict):
            for nested in value.values():
                yield from numeric_values(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from numeric_values(nested)
        elif isinstance(value, (int, float)) and np.isfinite(value):
            yield abs(float(value))

    clean_fixed_values = list(numeric_values(clean_fixed))
    sources = [
        source_record(NOTEBOOK, "Kaggle-compatible source code and embedded sealed records"),
        source_record(gen_path, "Sealed 252-run synthetic sensor generalization aggregate"),
        source_record(calibration_path, "Sealed synthetic calibration-target v2 summary"),
        source_record(strict_path, "Sealed strict calibration holdout summary"),
        source_record(clean_path, "Sealed clean measurement validity summary"),
        source_record(ref_path, "Sealed healthy-ring reference model comparison"),
        source_record(hidden_path, "Sealed healthy-disc v3 final report"),
        source_record(RESULTS / "healthy_skin_hidden_disc_v3" / "test_prediction_freeze.json",
                      "Frozen TEST manifest, decision, and prediction hash receipt"),
        source_record(RESULTS / "healthy_skin_hidden_disc_v3" / "completed_hidden_disc_seal.json",
                      "Completed hidden-disc v3 seal"),
        source_record(decision_path, "Sealed 18-case final architecture decision"),
        source_record(demo_path, "Offline five-case integration demo metrics"),
        source_record(demo_trace_path, "Offline three-stage scan-plan-update trace"),
        source_record(DEMO / "reproducibility_manifest.json", "Synthetic input hashes and runtime receipt"),
        source_record(DEMO / "geometry" / "supported_case_surfaces.npz", "Synthetic geometry for staged-fill figure"),
    ]

    routes = []
    for row in route_rows:
        routes.append(
            {
                **row,
                "mean_surface_error_percent_wound_width": (
                    100 * row["mean_surface_error_over_width_mean_available_only"]
                    if row["mean_surface_error_over_width_mean_available_only"] is not None
                    else None
                ),
                "worst_surface_error_percent_wound_width": (
                    100 * row["mean_surface_error_over_width_worst_available"]
                    if row["mean_surface_error_over_width_worst_available"] is not None
                    else None
                ),
                "mean_reference_volume_error_percent": (
                    100 * row["volume_relative_error_mean_available_only"]
                    if row["volume_relative_error_mean_available_only"] is not None
                    else None
                ),
                "worst_reference_volume_error_percent": (
                    100 * row["volume_relative_error_worst_available"]
                    if row["volume_relative_error_worst_available"] is not None
                    else None
                ),
                "repeatability_percent_wound_width": (
                    100 * row["repeatability_mean_over_width_mean_available_only"]
                    if row["repeatability_mean_over_width_mean_available_only"] is not None
                    else None
                ),
            }
        )

    published = []
    for row in notebook["published_comparison"]:
        if (row.get("device"), row.get("method")) in {
            ("iphone", "NAngelo"),
            ("logitech", "NF"),
        }:
            published.append(
                {
                    "device": row["device"],
                    "method": row["method"],
                    "common_input_frames": row["common_input_frames"],
                    "mean_point_to_surface_mm": row["point_to_surface_mean"],
                    "p95_point_to_surface_mm": row["point_to_surface_p95"],
                    "completeness_fraction_at_1mm": row["completeness"],
                    "gt_samples": row["gt_samples"],
                    "predicted_points": row["predicted_points"],
                    "alignment": "GT-derived similarity alignment",
                    "interpretation": "shape-only evaluation; independent metric scale NOT PROVEN",
                }
            )

    quantities = [
        {
            "category": "DIRECTLY OBSERVABLE",
            "examples": "Visible wound-bed surface, visible 3D boundary, local normals, visible area, support/confidence.",
            "basis": "RGB/depth or mesh observations, with their declared support and uncertainty.",
        },
        {
            "category": "REFERENCE-DEPENDENT",
            "examples": "Depth relative to a declared reference, fill height, layer count, reference-defined volume.",
            "basis": "Requires an explicitly named and versioned reference surface or target policy.",
        },
        {
            "category": "EXTERNALLY SPECIFIED",
            "examples": "Operator-selected target surface, declared layer thickness, safety margin, material spacing.",
            "basis": "User or external input; record its source and parameters.",
        },
        {
            "category": "NOT IDENTIFIABLE",
            "examples": "True pre-wound healthy surface, true missing-tissue volume, anatomical restoration volume.",
            "basis": "No such ground truth is present in the current observations or datasets.",
        },
    ]

    simulation_rows = []
    for case in demo["case_results"]:
        metrics = case["metrics"]
        simulation_rows.append(
            {
                "case_id": case["case_id"],
                "expected_decision": metrics["expected_decision"],
                "actual_decision": case["route"]["decision"],
                "recommended_action": case["route"]["recommended_action"],
                "stage_count": metrics.get("stage_count", 0),
                "boundary_violations": metrics.get("boundary_violations", 0),
                "exclusion_zone_violations": metrics.get("exclusion_zone_violations", 0),
                "unsupported_deposition_fraction": metrics.get("unsupported_deposition_fraction", 0.0),
                "planning_decision": metrics["planning_decision"],
                "route_expected_match": metrics["route_expected_match"],
                "final_status": "OFFLINE_SYNTHETIC_COMPUTATION_ONLY",
            }
        )

    evidence = {
        "schema": "publication-evidence-package-v1",
        "scientific_scope": (
            "Uncertainty-aware multimodal computational framework for visible 3D wound "
            "reconstruction, sensor routing, failure detection, and conservative "
            "surface-conformal deposition planning."
        ),
        "source_manifest": sources,
        "dataset_table": [
            {
                "dataset": "Syn3DWound",
                "origin": "Public synthetic wound dataset, CSIRO Data Access Portal",
                "modalities": "Rendered RGB, shaded masks/labels, cropped meshes, COLMAP cameras/poses.",
                "role": "Primary source for RGB shape evaluation, mesh-rendered depth, sensor routing, calibration, hidden-disc and geometry studies.",
                "sample_scope": "52 sequences inventoried; only explicitly listed subsets were evaluated. Sensor generalization used 4 sequence contexts and 252 correlated route/subset results.",
                "limitations": "No native depth maps or certified physical scale; local meshes are crops; synthetic renderer labels are not clinical truth.",
                "safe_to_commit": "No raw source data; download separately.",
            },
            {
                "dataset": "SALVE",
                "origin": "Public silicone wound-phantom dataset and author-published reconstructions.",
                "modalities": "RGB video, camera transforms/intrinsics, scanner point cloud/GT, published aligned meshes.",
                "role": "Evaluation-only comparison of selected author-published reconstructions on the PIS4 Accuracy split.",
                "sample_scope": "50 common frames from one phantom wound; 2,250 RGB/calibration/scanner-GT inputs and 102 published aligned outputs were acquired and inventoried.",
                "limitations": "GT similarity alignment is used; results evaluate shape only. Independent metric-scale recovery is not proven. The requested Seymour II acquisition is not available from the authors.",
                "safe_to_commit": "No raw source data or meshes; download separately.",
            },
            {
                "dataset": "FUSeg (legacy)",
                "origin": "Public clinical wound-segmentation benchmark referenced by historical model code.",
                "modalities": "RGB photographs and segmentation masks.",
                "role": "Not used as evidence in the completed reconstruction, sensor-routing, or safety results reported here.",
                "sample_scope": "No current-milestone evaluation count is claimed.",
                "limitations": "Legacy model components and notebook drafts do not establish clinical validation or generalization.",
                "safe_to_commit": "No raw images or masks.",
            },
            {
                "dataset": "Procedural offline simulation",
                "origin": "Deterministic synthetic geometry and explicitly defined sensor corruptions.",
                "modalities": "Generated RGB, mesh-rendered depth, target surfaces, geometry and plan traces.",
                "role": "Five-case offline sensor-routing and staged-plan/update demonstration; separate 18-case final architecture decision.",
                "sample_scope": "5 computational integration cases; 18 synthetic decision cases.",
                "limitations": "No physical camera, robot, CoppeliaSim, clinical data, or printing. Scene units are illustrative only.",
                "safe_to_commit": "Curated generated evidence may be committed; no raw external data.",
            },
        ],
        "focused_routes": routes,
        "alignment_sensitivity_routes": [
            {
                **row,
                "repeatability_percent_wound_width": (
                    100 * row["repeatability_mean_over_width_mean_available_only"]
                    if row["repeatability_mean_over_width_mean_available_only"] is not None
                    else None
                ),
                "mean_reference_volume_error_percent": (
                    100 * row["volume_relative_error_mean_available_only"]
                    if row["volume_relative_error_mean_available_only"] is not None
                    else None
                ),
            }
            for row in sensitivity_routes
        ],
        "calibration": {
            "strict_holdout": {
                "status": strict["status"],
                "clean_configurations": strict["clean_configurations"],
                "faulty_configurations": strict["faulty_configurations"],
                "target_design_stress_configurations": strict["target_design_stress_configurations"],
                "diagnostic_wound_subsets": strict["evaluated_wound_subsets"],
                "clean_false_alarm_count": round(
                    strict["clean_false_alarm_rate"] * strict["clean_configurations"]
                ),
                "clean_false_alarm_denominator": strict["clean_configurations"],
                "threshold_recall": thresholds,
                "all_research_measurement_gate_clean_controls_failed": (
                    strict["clean_baselines_failing_research_measurement_gate"]
                ),
                "print_authorized": strict["print_authorized"],
            },
            "target_ablation": {
                "synthetic_v2_configurations": calibration["heldout_configuration_count"],
                "stepped_vs_planar": target_rows,
                "target_design_stress_is_not_physical_validation": True,
            },
        },
        "clean_measurement": {
            "status": clean["status"],
            "accepted": clean["clean_cases_accepted"],
            "rejected": clean["clean_cases_rejected"],
            "dominant_blocker": clean["dominant_remaining_blocker"],
            "analytic_integration_max_relative_error": clean["analytic"]["max_error_finest"],
            "analytic_integration_unit": "relative error",
            "fixed_computed_closure_max_volume_error_fraction": (
                max(clean_fixed_values) if clean_fixed_values else None
            ),
            "official_clinical_volume": clean.get("official_clinical_volume"),
            "known_pre_wound_surface": clean.get("true_healthy_skin_labels_available"),
            "certified_physical_scale": clean.get("physical_scale_certified"),
            "volume_label": clean.get("reference_volume_label"),
        },
        "reference_surface_models": selected_references,
        "healthy_disc": {
            "status": hidden["status"],
            "interpretation": hidden["interpretation"],
            "global_method": hidden["global_method"],
            "test_body_models": hidden["test_body_models"],
            "test_patch_count": hidden["test_patch_count"],
            "test_patch_ring_records": hidden["test_patch_ring_records"],
            "valid_global_method_records": hidden["global_method_test_performance"]["available"],
            "global_method_total_records": hidden["global_method_test_performance"]["total"],
            "macro_mae_fraction_of_diameter": hidden["global_method_test_performance"]["macro_mae_fraction_of_diameter"],
            "macro_p95_fraction_of_diameter": hidden["global_method_test_performance"]["macro_p95_fraction_of_diameter"],
            "uncertainty_pointwise_coverage": hidden["uncertainty_envelope"]["pointwise_coverage"],
            "uncertainty_macro_patch_coverage": hidden["uncertainty_envelope"]["macro_patch_coverage"],
            "uncertainty_target_coverage": hidden["uncertainty_target_coverage"],
            "worst_confidently_wrong": hidden["worst_confidently_wrong"],
            "strict_split": hidden["strict_split"],
            "seal": _verify_hidden_disc_seal(),
            "physical_units": "dataset units and declared synthetic layer fractions only",
        },
        "architecture_decision": {
            "decision": architecture["decision"],
            "case_count": len(architecture["per_case"]),
            "body_model_scope": architecture["limitations"][0],
            "direct_reference_gate": architecture["direct_reference_only_gate"],
            "failures_by_edge_condition": architecture["failure_counts_by_edge_condition"],
            "no_physical_printing": architecture["no_physical_printing"],
            "no_planner_or_robot_integration": architecture["no_planner_or_robot_integration"],
        },
        "offline_demo": {
            "status": demo["status"],
            "case_count": demo["case_count"],
            "closed_loop": demo["closed_loop"],
            "coppeliasim": demo["coppeliasim"],
            "print_authorized": demo["print_authorized"],
            "physical_robot_commands_enabled": demo["physical_robot_commands_enabled"],
            "case_rows": simulation_rows,
            "trace_stage_history": [
                {
                    key: stage[key]
                    for key in (
                        "stage", "eligible_vertex_count", "deposition_point_count",
                        "plan_changed_from_previous", "unsupported_deposition_fraction",
                        "boundary_violation_count", "exclusion_zone_violation_count",
                        "remaining_supported_depth_m",
                        "remaining_supported_depth_after_update_m",
                    )
                }
                for stage in demo_trace["stage_history"]
            ],
            "target_is_anatomical_truth": demo_trace["target_is_anatomical_truth"],
        },
        "salve_shape_only": published,
        "syn3dwound_rgb_shape": {
            "support_fraction": notebook["syn_supported_surface"]["stable_and_multiview_supported_fraction"],
            "mean_point_to_surface_fraction_of_wound_width": notebook["syn_supported_surface"]["point_to_surface_mean_over_width"],
            "p95_point_to_surface_fraction_of_wound_width": notebook["syn_supported_surface"]["point_to_surface_p95_over_width"],
            "completeness_fraction_at_5pct_width": notebook["syn_supported_surface"]["wound_bed_completeness_at_5pct_width"],
            "pairwise_repeatability_chamfer_fraction_of_width": notebook["our_algorithm_repeatability"]["pairwise_chamfer_over_width"],
            "physical_metric_scale_valid": notebook["syn_supported_surface"]["physical_metric_scale_valid"],
            "interpretation": "Shape comparison after evaluation-only similarity alignment; not metric-scale recovery.",
        },
        "quantity_table": quantities,
        "statistical_audit": {
            "independent_unit": (
                "Dataset/model-specific sequence, body model, or frozen fault configuration—not pixels, views, "
                "patch-ring pairs, or repeated subsets."
            ),
            "no_inferential_confidence_intervals": True,
            "reason": (
                "Small numbers of independent wounds/body models and correlated views, subsets, patches, and "
                "fault variants do not justify population-level intervals."
            ),
            "route_runs": generalization["route_subset_count"],
            "route_sequence_contexts": len(generalization["sequences"]),
            "strict_clean_configs": strict["clean_configurations"],
            "strict_fault_configs": strict["faulty_configurations"],
            "strict_design_stress_configs": strict["target_design_stress_configurations"],
            "strict_wound_subsets": strict["evaluated_wound_subsets"],
            "hidden_disc_patches": hidden["test_patch_count"],
            "hidden_disc_patch_ring_records": hidden["test_patch_ring_records"],
            "hidden_disc_valid_model_records": hidden["global_method_test_performance"]["available"],
            "architecture_synthetic_cases": len(architecture["per_case"]),
            "offline_demo_cases": demo["case_count"],
            "percent_wound_width_metrics": [
                "surface error", "boundary error", "depth error", "hidden-disc error",
                "repeatability", "misalignment offsets normalized by wound width when specified",
            ],
            "physical_unit_metrics": [
                "SALVE point-to-surface millimeters only after GT similarity alignment; shape-only interpretation."
            ],
        },
    }
    shutil.copyfile(
        DEMO / "geometry" / "supported_case_surfaces.npz",
        DATA / "figure09_staged_geometry.npz",
    )
    return evidence


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence()
    evidence_path = DATA / "publication_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    hidden = export_hidden_disc_example()
    reconstruction = export_reconstruction_example()
    (DATA / "figure03_provenance.json").write_text(
        json.dumps(reconstruction, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "evidence": evidence_path.relative_to(ROOT).as_posix(),
                "source_count": len(evidence["source_manifest"]),
                "route_count": len(evidence["focused_routes"]),
                "figure08_test_points": len(hidden["heldout_truth_z"]),
                "figure03_sources": reconstruction,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
