"""Tests for the patient-level pipeline (pipeline/patient.py and scripts/run_patient_pipeline.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import coverage, patient  # noqa: E402

FRAME = {"origin": np.zeros(3), "e1": np.array([1.0, 0, 0]), "e2": np.array([0, 1.0, 0]),
         "normal": np.array([0, 0, 1.0]), "wound_radius": 0.01}


def synthetic_field(depth=0.0007, pixel=0.0001, half=0.012):
    axis = np.arange(-half, half + pixel / 2, pixel)
    gx, gy = np.meshgrid(axis, axis)
    inside = (gx / 0.008) ** 2 + (gy / 0.003) ** 2 <= 1.0
    height = 0.02 * gx ** 2 - np.where(inside, depth, 0.0)
    ys, xs = np.nonzero(inside)
    ring = ~inside & ((gx / 0.0105) ** 2 + (gy / 0.0055) ** 2 <= 1.0)
    return {
        "axis": axis, "gx": gx, "gy": gy, "pixel_du": pixel, "height": height,
        "valid": np.ones_like(inside), "wound_mask": inside, "ring_mask": ring,
        "centroid_px": [float(xs.mean()), float(ys.mean())],
        "boundary_width_px": float(max(np.ptp(xs), np.ptp(ys)) + 1),
        "narrow_width_px": float(min(np.ptp(xs), np.ptp(ys)) + 1),
    }


def test_quaternion_is_rotation():
    rng = np.random.default_rng(1)
    q = rng.normal(size=4)
    R = patient.quaternion_to_matrix(q / np.linalg.norm(q))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(patient.quaternion_to_matrix([1, 0, 0, 0]), np.eye(3))


def test_colmap_images_parser_skips_empty_point_lines(tmp_path):
    text = ("# header\n"
            "0 1 0 0 0 0.1 0.2 1.3 0 camera_0\n\n"
            "1 1 0 0 0 0 0 1 0 camera_12\n\n")
    path = tmp_path / "images.txt"
    path.write_text(text)
    records = patient.read_colmap_images(path)
    assert [r["file_stem"] for r in records] == ["000", "012"]
    assert np.allclose(records[0]["t"], [0.1, 0.2, 1.3])


def test_quadratic_target_recovers_healthy_skin():
    field = synthetic_field()
    fit = patient.fit_quadratic_target(field)
    expected = 0.02 * field["gx"] ** 2
    assert fit["ring_rms_du"] < 1e-9
    assert np.allclose(fit["target"][field["wound_mask"]], expected[field["wound_mask"]], atol=1e-9)


def test_local_frame_is_orthonormal_and_follows_long_axis():
    rng = np.random.default_rng(2)
    xy = rng.uniform(-0.03, 0.03, size=(4000, 2))
    vertices = np.column_stack((xy, np.zeros(len(xy))))
    wound = (xy[:, 0] / 0.01) ** 2 + (xy[:, 1] / 0.003) ** 2 <= 1
    frame = patient.local_frame(vertices, wound, toward=np.array([0, 0, 1.0]), radius_factor=2.5)
    basis = np.vstack((frame["e1"], frame["e2"], frame["normal"]))
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-9)
    assert frame["normal"][2] > 0.99
    assert abs(frame["e1"][0]) > 0.99
    points = rng.normal(size=(5, 3))
    assert np.allclose(patient.to_world(patient.to_local(points, frame), frame), points)


def test_staged_fill_is_conservative_and_stops():
    field = synthetic_field()
    target = patient.fit_quadratic_target(field)["target"]
    result = patient.staged_fill(field, target, FRAME)
    summary = result["summary"]
    layer = patient.PATIENT_DEFAULTS["layer_thickness_du"]
    assert summary["stage_count"] >= 1
    assert summary["stop_reason"] in {"SUPPORTED_DEPTH_BELOW_ONE_LAYER", "NO_SAFE_SUPPORTED_REGION"}
    assert summary["boundary_violations"] == 0
    assert summary["unsupported_cells"] == 0
    assert summary["max_overfill_vs_target_du"] == 0.0
    for stage in result["stages"]:
        values = np.unique(stage["_deposit"])
        assert set(np.round(values / layer, 9)) <= {0.0, 1.0}  # whole layers only
        assert not (stage["_deposit"] > 0)[~result["supported"]].any()
        rotations = stage["_toolpath"]["rotations"]
        assert rotations.shape == (3, 3, stage["toolpath_points"])


def test_staged_fill_refuses_when_uncertainty_too_large():
    field = synthetic_field()
    target = patient.fit_quadratic_target(field)["target"]
    params = dict(patient.PATIENT_DEFAULTS, measurement_uncertainty_du=0.00015)
    result = patient.staged_fill(field, target, FRAME, params)
    assert result["summary"]["stage_count"] == 0
    assert result["summary"]["lattice_cells_inside_supported_region"] == 0


def test_script_fails_clearly_without_data(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_patient_pipeline

    code = run_patient_pipeline.main(["--scene", str(tmp_path / "missing"), "--output", str(tmp_path / "out")])
    assert code == 2
    assert "Syn3DWound leg scene not found" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(not (patient.DEFAULT_SCENE / "cropped_mesh.obj").is_file(),
                    reason="public Syn3DWound leg sequence not downloaded")
def test_patient_scene_end_to_end():
    result = patient.run_patient()
    summary = result["fill"]["summary"]
    assert result["labels"]["wound"].sum() > 50
    assert summary["stage_count"] >= 1
    assert summary["boundary_violations"] == 0
    assert summary["unsupported_cells"] == 0
    assert summary["max_overfill_vs_target_du"] == 0.0
    assert sum(summary["violations"].values()) == 0
    if summary["status"] == "COMPLETE":
        assert summary["eligible_coverage"] >= summary["coverage_gate"]
    # The audited previous planner is reproduced exactly by the generic staged fill.
    v1 = result["coverage"]["results"]["V1_LATTICE"]
    assert np.allclose(np.nan_to_num(v1["final_height"]), np.nan_to_num(result["fill_v1"]["final_height"]))
    assert v1["summary"]["active_cells_by_stage"] == [s["active_cells"] for s in result["fill_v1"]["stages"]]


# ----------------------------------------------------------------------------- coverage planner

SMALL_SEARCH = {"side_steps": 2, "rotations_deg": (0, 30), "offset_steps": 2, "diagnostic_side_strand_widths": (1.0,)}


@pytest.fixture(scope="module")
def synthetic_plan():
    field = synthetic_field()
    target = patient.fit_quadratic_target(field)["target"]
    fill_v1 = patient.staged_fill(field, target, FRAME)
    plan = coverage.plan_patient_coverage(field, target, FRAME, patient.PATIENT_DEFAULTS, patient.to_world,
                                          fill_v1["honeycomb"], SMALL_SEARCH)
    return field, target, fill_v1, plan


def test_eligibility_reuses_unchanged_safety_rules():
    field = synthetic_field()
    target = patient.fit_quadratic_target(field)["target"]
    params = patient.PATIENT_DEFAULTS
    elig = coverage.eligibility(field, target, params)
    uncertainty = np.where(field["valid"], params["measurement_uncertainty_du"], np.inf)
    assert np.array_equal(elig["supported"], patient._supported(field, uncertainty, params))
    assert not (elig["eligible"] & ~elig["supported"]).any()
    assert (elig["category"][field["wound_mask"]] != coverage.OUTSIDE_WOUND).all()
    assert (elig["category"][~field["wound_mask"]] == coverage.OUTSIDE_WOUND).all()


def test_segment_crossings_counts_proper_crossings_only():
    crossing = np.array([[[0, 0], [2, 2]], [[0, 2], [2, 0]]], float)
    touching = np.array([[[0, 0], [1, 1]], [[1, 1], [2, 0]]], float)
    apart = np.array([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], float)
    assert coverage.segment_crossings(crossing) == 1
    assert coverage.segment_crossings(touching) == 0
    assert coverage.segment_crossings(apart) == 0


def test_generic_staged_fill_reproduces_previous_planner(synthetic_plan):
    _, _, fill_v1, plan = synthetic_plan
    v1 = plan["results"]["V1_LATTICE"]
    assert np.allclose(np.nan_to_num(v1["final_height"]), np.nan_to_num(fill_v1["final_height"]))
    assert v1["summary"]["active_cells_by_stage"] == [s["active_cells"] for s in fill_v1["stages"]]
    assert v1["summary"]["stop_reason"] == fill_v1["summary"]["stop_reason"]


def test_every_planner_keeps_all_safety_invariants(synthetic_plan):
    field, target, _, plan = synthetic_plan
    eligible = plan["eligibility"]["eligible"].ravel()
    for name, result in plan["results"].items():
        summary = result["summary"]
        assert sum(summary["violations"].values()) == 0, name
        assert summary["max_overfill_vs_target_du"] == 0.0, name
        for cell in result["cells"]:
            assert eligible[cell["pixels"]].all(), name
        total = sum(stage["_deposit"] for stage in result["stages"])
        assert not (total > 0)[~plan["eligibility"]["supported"]].any(), name
        assert summary["status"] in {"COMPLETE", "PARTIAL"}
        assert (summary["status"] == "COMPLETE") == (summary["eligible_coverage"] >= summary["coverage_gate"])


def test_clipped_cells_are_printable_pieces(synthetic_plan):
    _, _, _, plan = synthetic_plan
    params = plan["coverage_parameters"]
    for cell in plan["results"]["CLIPPED_BOUNDARY"]["cells"]:
        if cell["kind"] != "clipped":
            continue
        hex_area = 1.5 * np.sqrt(3) * cell["side_px"] ** 2
        assert len(cell["pixels"]) >= params["clip_min_area_fraction"] * hex_area
        assert np.allclose(cell["path_px"][0], cell["path_px"][-1])  # closed wall


def test_coverage_selection_and_status(synthetic_plan):
    _, _, _, plan = synthetic_plan
    best = max(coverage.PLANNER_PREFERENCE, key=lambda n: plan["results"][n]["summary"]["eligible_coverage"])
    passing = [n for n in coverage.PLANNER_PREFERENCE if plan["results"][n]["summary"]["status"] == "COMPLETE"]
    assert plan["selected"] == (passing[0] if passing else best)
    assert (plan["binding_constraint"] is None) == bool(passing)


def test_coverage_rejects_when_uncertainty_too_large():
    field = synthetic_field()
    target = patient.fit_quadratic_target(field)["target"]
    params = dict(patient.PATIENT_DEFAULTS, measurement_uncertainty_du=0.00015)
    plan = coverage.plan_patient_coverage(field, target, FRAME, params, patient.to_world, None, SMALL_SEARCH)
    assert not plan["eligibility"]["eligible"].any()
    for name in coverage.PLANNER_PREFERENCE:
        assert plan["results"][name]["summary"]["status"] == "REJECTED"
        assert plan["results"][name]["summary"]["stage_count"] == 0
