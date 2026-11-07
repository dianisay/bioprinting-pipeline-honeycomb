"""End-to-end tests for the unified offline pipeline (no printing, no robot)."""

import numpy as np
import pytest

from pipeline.unified import load_config, public_summary, run_all_cases, run_case

EXPECTED = {
    "A_supported_rgbd": ("STAGED_FILL", "PLAN_READY_FOR_OPERATOR_REVIEW_PRINT_DISABLED"),
    "B_boundary_resolution": ("ROUTE", "BOUNDARY_RESOLUTION_INSUFFICIENT"),
    "C_incomplete_depth": ("ROUTE", "SURFACE_INCOMPLETE"),
    "D_rgb_depth_misalignment": ("ROUTE", "RGB_DEPTH_CALIBRATION_FAILED"),
    "E_reference_disagreement": ("ROUTE", "EXTERNAL_TARGET_REQUIRED"),
}


@pytest.fixture(scope="module")
def results():
    return {r["case_id"]: r for r in run_all_cases()}


@pytest.fixture(scope="module")
def supported(results):
    return results["A_supported_rgbd"]


def test_every_case_stops_at_expected_stage(results):
    assert {cid: (r["final_stage"], r["final_decision"]) for cid, r in results.items()} == EXPECTED


def test_refused_cases_produce_no_toolpath_and_never_authorize_printing(results):
    for cid, r in results.items():
        assert r["print_authorized"] is False
        assert r["physical_robot_commands_enabled"] is False
        if cid != "A_supported_rgbd":
            assert r["plan"] is None


def test_identification_feeds_router_and_matches_renderer_mask(supported):
    identify = next(s for s in supported["stage_trace"] if s["stage"] == "IDENTIFY")
    assert identify["mask_iou_vs_renderer_mask_evaluation_only"] > 0.95
    assert identify["boundary_width_px"] >= 100


def test_accepted_reconstruction_without_operator_target_is_refused():
    config = load_config()
    case = dict(next(c for c in config["cases"] if c["case_id"] == "A_supported_rgbd"), target_available=False)
    result = run_case(case, config)
    assert result["final_stage"] == "TARGET"
    assert result["final_decision"] == "OPERATOR_FINAL_TARGET_REQUIRED"
    assert result["plan"] is None


def test_staged_fill_is_conservative_and_adaptive(supported, results):
    plan = supported["plan"]
    summary = plan["summary"]
    layer = load_config()["simulation"]["layer_thickness_m"]
    assert summary["stage_count"] >= 2
    assert summary["stop_reason"] in ("SUPPORTED_DEPTH_BELOW_ONE_LAYER", "NO_SAFE_SUPPORTED_REGION")
    assert summary["boundary_violations"] == 0
    assert summary["unsupported_cells"] == 0
    assert summary["max_overfill_m_evaluation_only"] <= 1e-12
    hashes = [s["plan_hash"] for s in plan["stages"]]
    assert len(set(hashes)) == len(hashes)
    for stage in plan["stages"]:
        assert stage["max_supported_remaining_before_m"] >= layer


def test_toolpath_is_conformal_to_the_measured_surface(supported):
    plan = supported["plan"]
    offset = load_config()["simulation"]["tool_to_surface_offset_m"]
    for stage in plan["stages"]:
        tp = stage["_toolpath"]
        assert np.allclose(np.linalg.norm(tp["normals"], axis=1), 1.0)
        assert np.all(tp["normals"][:, 2] > 0.5)
        assert np.allclose(tp["rotations"][:, 2, :].T, -tp["normals"])
        rc = tp["pixel_rc"][tp["deposit_flag"]]
        assert plan["supported"][rc[:, 0], rc[:, 1]].all()
        surface = tp["points_m"][tp["deposit_flag"]] - tp["normals"][tp["deposit_flag"]] * offset
        top = stage["_surface_after"][rc[:, 0], rc[:, 1]]
        assert np.allclose(surface[:, 2], top, atol=1e-9)


def test_pipeline_is_deterministic_and_serialisable(supported):
    import json

    again = run_case(next(c for c in load_config()["cases"] if c["case_id"] == "A_supported_rgbd"))
    assert [s["plan_hash"] for s in again["plan"]["stages"]] == [s["plan_hash"] for s in supported["plan"]["stages"]]
    json.dumps(public_summary(supported), allow_nan=False)
