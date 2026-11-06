import json
from pathlib import Path

import numpy as np

from robot.simulation.observable_demo import (
    DEFAULT_CONFIG,
    _candidate_plan,
    _circle_mesh,
    _route,
    _sensor_observation,
    run_demo,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    document = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return document, dict(document["simulation"], seed=document["seed"])


def test_sensor_modes_use_sealed_profiles_and_gate_by_quality():
    document, config = _config()
    cases = {case["case_id"]: case for case in document["cases"]}

    rgb_only = _sensor_observation(cases["B_boundary_resolution"], config, None)
    assert not rgb_only["depth_mask"].any()
    assert rgb_only["depth_coverage"] == 0.0
    assert _route(cases["B_boundary_resolution"], config, rgb_only)["decision"] == (
        "BOUNDARY_RESOLUTION_INSUFFICIENT"
    )

    low_coverage = _sensor_observation(
        cases["C_incomplete_depth"],
        config,
        document["sensor_profiles"]["LOW_COVERAGE"],
    )
    assert low_coverage["depth_coverage"] < config["minimum_depth_coverage"]
    assert _route(cases["C_incomplete_depth"], config, low_coverage)["decision"] == "SURFACE_INCOMPLETE"

    misaligned = _sensor_observation(
        cases["D_rgb_depth_misalignment"],
        config,
        document["sensor_profiles"]["MISALIGNED_DEPTH"],
    )
    assert misaligned["rgb_depth_boundary_disagreement"] > (
        config["maximum_rgb_depth_boundary_disagreement"]
    )
    assert _route(cases["D_rgb_depth_misalignment"], config, misaligned)["decision"] == (
        "RGB_DEPTH_CALIBRATION_FAILED"
    )


def test_candidate_adapter_uses_explicit_surface_and_constraints():
    document, config = _config()
    vertices, faces, radial, _ = _circle_mesh(
        config["radial_segments"], config["angular_segments"], 0.03
    )
    target = np.full(len(vertices), vertices[:, 2].max())
    normals = np.tile([0.0, 0.0, 1.0], (len(vertices), 1))
    plan = _candidate_plan(
        vertices,
        faces,
        radial,
        vertices[:, 2],
        normals,
        target,
        np.zeros(len(vertices)),
        config,
    )
    assert plan["eligible"].any()
    assert plan["trajectory"].shape[1] == 3
    assert plan["tool_to_surface_offset_m"] == config["tool_to_surface_offset_m"]
    assert plan["exclusion_zones"] == config["exclusion_zones"]
    assert "volume" not in plan


def test_five_case_run_seals_outputs_and_adaptive_decisions(tmp_path):
    report = run_demo(DEFAULT_CONFIG, tmp_path)
    assert report["case_count"] == 5
    assert report["status"] == "OFFLINE_COMPUTATIONAL_DEMO_COMPLETE_COPPELIASIM_NOT_RUN"
    assert report["coppeliasim"]["installed"] is False
    assert report["coppeliasim"]["scene_file_created"] is False
    assert report["closed_loop"]["stage_count"] >= 2
    assert report["closed_loop"]["second_plan_changed"] is True
    assert report["closed_loop"]["stop_decision"] in (
        "SIMULATION_PLAN_COMPLETE",
        "NO_SAFE_DEPOSITION_REGION",
    )
    assert report["closed_loop"]["boundary_violations"] == 0
    assert report["closed_loop"]["exclusion_zone_violations"] == 0
    assert report["closed_loop"]["unsupported_deposition_fraction"] == 0.0
    assert "collision_count" not in report["closed_loop"]

    expected_cases = {
        "A_supported_rgbd",
        "B_boundary_resolution",
        "C_incomplete_depth",
        "D_rgb_depth_misalignment",
        "E_reference_disagreement",
    }
    assert {path.stem for path in (tmp_path / "case_manifests").glob("*.json")} == expected_cases
    assert len(list((tmp_path / "failure_receipts").glob("*.json"))) == 4
    for path in (tmp_path / "inputs").glob("*/sensor_observation.npz"):
        with np.load(path) as arrays:
            assert "depth_m" in arrays and "rgb_mask" in arrays
    assert (tmp_path / "offline_computational_preview.mp4").stat().st_size > 0
    assert (tmp_path / "closed_loop_trace.json").is_file()
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["print_authorized"] is False
    mode_smoke = json.loads((tmp_path / "mode_smoke_tests.json").read_text(encoding="utf-8"))
    assert mode_smoke["RGB_ONLY"]["route"]["decision"] == "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION"
    assert mode_smoke["RGB_ONLY"]["depth_stream_present"] is False
    assert mode_smoke["RGBD"]["route"]["decision"] == "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION"
    assert [item["decision"] for item in mode_smoke["DEGRADED_RGBD"]["routes"]] == [
        "SURFACE_INCOMPLETE",
        "RGB_DEPTH_CALIBRATION_FAILED",
    ]
