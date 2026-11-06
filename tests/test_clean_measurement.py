"""Known-closure numerical fixtures, not clinical or physical validation."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "04_rgb_wound_3d_reconstruction.ipynb"


@pytest.fixture(scope="module")
def measurement():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("metadata", {}).get("clean_measurement_validity")
                and "CLEAN_MEASUREMENT_SOURCES =" in "".join(c["source"]))
    source = "".join(cell["source"])
    assert hashlib.sha256(source.encode()).hexdigest() == cell["metadata"]["source_sha256"]
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CLEAN_MEASUREMENT_SOURCES" for t in n.targets))
    sources = ast.literal_eval(assignment.value)
    scope = {}
    exec(compile(sources["measurement_kernel.py"], str(NOTEBOOK), "exec"), scope)
    exec(compile(sources["measurement_reference.py"], str(NOTEBOOK), "exec"), scope)
    scope["embedded_sources"] = sources
    return scope


def supported_evidence():
    return dict(oracle_verified=True, analytic_verified=True, integration_residual=0., support=1.,
        reference_volume_fraction=.01, boundary_volume_fraction=.01, depth_signal_to_uncertainty=20.,
        volume_uncertainty_fraction=.05, depth_uncertainty_fraction=.05, closure_identifiable=True)


def test_positive_triangle_clipping_recovers_exact_crossing(measurement):
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    result = measurement["m_integrate"](v, np.array([[0, 1, 2]]), np.array([1., -1., -1.]))
    assert result["volume"] == pytest.approx(1 / 24)
    assert result["legacy_positive_vertex_volume"] == pytest.approx(1 / 6)
    assert result["signed_shell_agreement_applicable"]
    assert not result["legacy_vertex_cap_same_as_zero_clipped_closure"]
    assert result["signed_closed_shell_volume"] == pytest.approx(1 / 24)
    assert result["integration_residual"] < 1e-12
    assert result["legacy_clipping_bias_fraction"] == pytest.approx(3.)


def test_closed_shell_orientation_scale_and_coordinates(measurement):
    v = np.array([[0., 0., -.2], [1., 0., -.2], [1., 1., -.2], [0., 1., -.2]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    m = measurement["m_integrate"](v, f, np.zeros(4))
    assert m["volume"] == pytest.approx(.2)
    assert m["signed_closed_shell_volume"] == pytest.approx(.2)
    assert m["topology"]["closed_shell_boundary_edges"] == 0
    assert measurement["m_integrate"](v, f[:, ::-1], np.zeros(4))["volume"] == pytest.approx(.2)
    scaled = measurement["m_integrate"](v * 3, f, np.zeros(4))
    assert scaled["volume"] == pytest.approx(.2 * 27)
    assert scaled["bed_area"] == pytest.approx(m["bed_area"] * 9)
    assert scaled["perimeter"] == pytest.approx(m["perimeter"] * 3)
    assert scaled["max_depth"] == pytest.approx(m["max_depth"] * 3)
    translated = measurement["m_integrate"](v + [100., -50., 9.], f, np.full(4, 9.))
    assert translated["volume"] == pytest.approx(.2)
    assert translated["signed_closed_shell_volume"] == pytest.approx(.2)


def test_duplicate_degenerate_and_nonfinite_meshes_fail_explicitly(measurement):
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    with pytest.raises(ValueError, match="Duplicate faces"):
        measurement["m_integrate"](v, np.array([[0, 1, 2], [2, 1, 0]]), np.ones(3))
    with pytest.raises(ValueError, match="Degenerate"):
        measurement["m_integrate"](v, np.array([[0, 0, 2]]), np.ones(3))
    with pytest.raises(ValueError, match="Finite"):
        measurement["m_integrate"](v * np.nan, np.array([[0, 1, 2]]), np.ones(3))


@pytest.mark.parametrize("method", ["plane", "ransac_plane", "quadratic", "thin_plate_spline",
                                   "bspline", "local_rbf"])
def test_healthy_only_models_recover_known_plane(measurement, method):
    theta = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    xy = np.vstack([np.column_stack((r * np.cos(theta), r * np.sin(theta))) for r in (.4, .6, .8)])
    z = .02 + .03 * xy[:, 0] - .01 * xy[:, 1]
    query = np.array([[0., 0.], [.05, -.05]])
    fit = measurement["m_fit_reference"](xy, z, method)
    np.testing.assert_allclose(fit(query), .02 + .03 * query[:, 0] - .01 * query[:, 1], atol=1e-4)


@pytest.mark.parametrize("field,value,decision", [
    ("oracle_verified", False, "VOLUME_INTEGRATION_FAILED"),
    ("analytic_verified", False, "VOLUME_INTEGRATION_FAILED"),
    ("integration_residual", .011, "VOLUME_INTEGRATION_FAILED"),
    ("support", .94, "WOUND_SURFACE_INCOMPLETE"),
    ("reference_volume_fraction", .11, "REFERENCE_SURFACE_UNSTABLE"),
    ("closure_identifiable", False, "REFERENCE_SURFACE_UNSTABLE"),
    ("boundary_volume_fraction", .11, "BOUNDARY_UNCERTAIN"),
    ("depth_signal_to_uncertainty", 4.9, "SHALLOW_DEPTH_UNRESOLVED"),
    ("depth_uncertainty_fraction", .21, "SHALLOW_DEPTH_UNRESOLVED"),
    ("volume_uncertainty_fraction", .11, "MEASUREMENT_INCONCLUSIVE"),
    ("volume_uncertainty_fraction", None, "MEASUREMENT_INCONCLUSIVE"),
    ("volume_uncertainty_fraction", np.nan, "MEASUREMENT_INCONCLUSIVE")])
def test_gate_vetoes_unsupported_measurement_without_printing(measurement, field, value, decision):
    result = measurement["m_gate"]({**supported_evidence(), field: value})
    assert result["decision"] == decision
    assert not result["accepted"]
    assert not result["print_authorized"]


def test_gate_has_a_known_closure_acceptance_region_but_not_printing(measurement):
    gate = measurement["m_gate"](supported_evidence())
    assert gate["decision"] == "MEASUREMENT_SUPPORTED"
    assert not gate["print_authorized"]
    evidence = supported_evidence()
    del evidence["closure_identifiable"]
    assert measurement["m_gate"](evidence)["decision"] == "MEASUREMENT_INCONCLUSIVE"
    assert measurement["m_gate"]({**supported_evidence(), "closure_identifiable": "unknown"})[
        "decision"] == "MEASUREMENT_INCONCLUSIVE"


def test_new_runtime_does_not_embed_or_tune_calibration_detector(measurement):
    sources = measurement["embedded_sources"]
    assert "calibration_target_v2.py" not in sources
    for source in sources.values():
        ast.parse(source)
    assert measurement["MEASUREMENT_POLICY"]["volume_uncertainty_fraction_max"] == .10
    assert measurement["MEASUREMENT_POLICY"]["analytic_relative_tolerance"] == .01
