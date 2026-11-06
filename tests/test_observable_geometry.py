"""Offline geometry conventions, not anatomical or printer safety validation."""

import ast
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import pytest


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "04_rgb_wound_3d_reconstruction.ipynb"


@pytest.fixture(scope="module")
def geometry():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("metadata", {}).get("observable_geometry")
                and "OBSERVABLE_SOURCES =" in "".join(c["source"]))
    source = "".join(cell["source"])
    assert hashlib.sha256(source.encode()).hexdigest() == cell["metadata"]["source_sha256"]
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "OBSERVABLE_SOURCES" for t in n.targets))
    sources = ast.literal_eval(assignment.value)
    module = types.ModuleType("observable_geometry_test")
    exec(compile(sources["observable_geometry.py"], str(NOTEBOOK), "exec"), module.__dict__)
    module.embedded_sources = sources
    return module


@pytest.fixture
def patch_mesh():
    x, y = np.meshgrid(np.linspace(-1., 1., 9), np.linspace(-1., 1., 9))
    v = np.column_stack((x.ravel(), y.ravel(), np.zeros(81)))
    faces = []
    for r in range(8):
        for c in range(8):
            i = r*9+c
            faces.extend([[i, i+1, i+10], [i, i+10, i+9]])
    return v, np.asarray(faces)


def maps_for(geometry, patch_mesh, **kwargs):
    v, f = patch_mesh
    return geometry.geometry_maps(v, f, v.copy(), 2., 200., **kwargs)


def test_observed_geometry_does_not_require_volume_or_closure(geometry, patch_mesh):
    maps = maps_for(geometry, patch_mesh)
    assert maps["status"] == "OBSERVED_GEOMETRY_SUPPORTED"
    assert maps["visible_surface_area"] == pytest.approx(4.)
    np.testing.assert_allclose(maps["normals"], np.tile([0., 0., 1.], (81, 1)))
    assert maps["geodesic"].min() == 0
    assert maps["geodesic"].max() == pytest.approx(1.)
    assert maps["confidence"].min() == 1
    assert not any("volume" in key for key in maps)


def test_boundary_coordinate_frame_is_rigid_and_boundary_anchored(geometry, patch_mesh):
    maps = maps_for(geometry, patch_mesh)
    np.testing.assert_allclose(maps["frame_axes"].T@maps["frame_axes"], np.eye(3), atol=1e-14)
    np.testing.assert_allclose(maps["coordinates"]@maps["frame_axes"].T+maps["frame_origin"],
                               maps["vertices"], atol=1e-14)
    assert np.linalg.det(maps["frame_axes"]) == pytest.approx(1.)


def test_uniform_offset_is_one_normal_unit_not_anatomical_depth(geometry, patch_mesh):
    maps = maps_for(geometry, patch_mesh)
    plan = geometry.make_plan(maps, "UNIFORM_OFFSET", dict(thickness=1.), .25)
    np.testing.assert_allclose(plan["height"], 1.)
    np.testing.assert_allclose(plan["target"][:, 2], 1.)
    assert len(plan["layers"]) == 4
    for layer in plan["layers"]:
        np.testing.assert_allclose(layer["vertices"][:, 2], layer["number"]*.25)
        assert not maps["no_print"][layer["faces"]].any()
        assert layer["topology"]["self_intersections"] == 0
    assert not plan["print_authorized"]


def test_uniform_offset_follows_sloped_surface_normals(geometry, patch_mesh):
    v, f = patch_mesh
    v[:, 2] = .2*v[:, 0]
    maps = maps_for(geometry, (v, f))
    plan = geometry.make_plan(maps, "UNIFORM_OFFSET", dict(thickness=.1), .025)
    expected = np.tile(np.array([-.2, 0., 1.])/np.sqrt(1.04)*.1, (len(v), 1))
    np.testing.assert_allclose(plan["displacement"], expected, atol=1e-14)
    assert not plan["print_authorized"]


@pytest.mark.parametrize("fraction", [.25, .5, .75])
def test_staged_fill_and_partial_last_layers(geometry, patch_mesh, fraction):
    maps = maps_for(geometry, patch_mesh)
    target = np.full(81, .7)
    plan = geometry.make_plan(maps, "STAGED_PARTIAL_FILL", dict(fraction=fraction), .1,
                              target_z=target, reference_surface="Explicit synthetic cap")
    np.testing.assert_allclose(plan["height"], .7*fraction)
    np.testing.assert_allclose(plan["layers"][-1]["vertices"][:, 2], .7*fraction)
    assert max(layer["vertices"][:, 2].max() for layer in plan["layers"]) <= .7*fraction+1e-14


@pytest.mark.parametrize("quadratic", [False, True])
def test_operator_surface_reproduces_defined_planar_or_quadratic_target(geometry, patch_mesh, quadratic):
    maps = maps_for(geometry, patch_mesh)
    target = .5+.1*maps["vertices"][:, 0]**2 if quadratic else np.full(81, .5)
    plan = geometry.make_plan(maps, "OPERATOR_DEFINED_TARGET_SURFACE", {}, .1,
                              target_z=target, reference_surface="Defined target")
    np.testing.assert_allclose(plan["target"][:, 2], target)
    assert plan["status"] == "EXPLICIT_POLICY_VALID"


def test_harmonic_cap_holds_every_boundary_vertex(geometry, patch_mesh):
    v, f = patch_mesh
    v[:, 2] = -.4*(1-v[:, 0]**2)*(1-v[:, 1]**2)
    cap = geometry.boundary_cap(v, f)
    np.testing.assert_allclose(cap, 0., atol=1e-13)
    maps = maps_for(geometry, (v, f))
    plan = geometry.make_plan(maps, "BOUNDARY_INTERPOLATED_CAP", {}, .1)
    np.testing.assert_allclose(plan["target"][:, 2], 0., atol=1e-13)
    assert "convention" in plan["reference_surface"]


def test_operator_required_cannot_be_success_shaped_empty_fallback(geometry, patch_mesh):
    plan = geometry.make_plan(maps_for(geometry, patch_mesh), "OPERATOR_DEFINED_TARGET_SURFACE", {}, .1)
    assert plan["status"] == "OPERATOR_REFERENCE_REQUIRED"
    assert not plan["layers"]
    assert plan["no_print"].all()


def test_maximum_height_subtracts_uncertainty_and_never_exceeds_limit(geometry, patch_mesh):
    maps = maps_for(geometry, patch_mesh, base_uncertainty=.001)
    plan = geometry.make_plan(maps, "MAXIMUM_SAFE_HEIGHT", dict(maximum_height=.2), .1,
                              target_z=np.ones(81), reference_surface="Engineering envelope")
    np.testing.assert_allclose(plan["height"], .199)
    assert plan["height"].max() <= .2


@pytest.mark.parametrize("policy,parameters", [
    ("UNIFORM_OFFSET", {"thickness": 0.}),
    ("UNIFORM_OFFSET", {"thickness": np.nan}),
    ("UNIFORM_OFFSET", {"thickness": True}),
    ("STAGED_PARTIAL_FILL", {"fraction": 1.5}),
    ("MAXIMUM_SAFE_HEIGHT", {"maximum_height": -.1}),
    ("MAXIMUM_SAFE_HEIGHT", {"maximum_height": .1, "uncertainty_clearance": -1}),
    ("UNKNOWN", {})])
def test_invalid_policy_parameters_raise_explicit_error(geometry, patch_mesh, policy, parameters):
    with pytest.raises(ValueError):
        geometry.make_plan(maps_for(geometry, patch_mesh), policy, parameters, .1,
                           target_z=np.ones(81), reference_surface="Fixture")


@pytest.mark.parametrize("target,reference", [
    (np.ones(80), "Fixture"), (np.full(81, np.nan), "Fixture"),
    (np.full(81, -.1), "Fixture"), (np.ones(81), None)])
def test_invalid_registered_targets_raise(geometry, patch_mesh, target, reference):
    with pytest.raises(ValueError):
        geometry.make_plan(maps_for(geometry, patch_mesh), "OPERATOR_DEFINED_TARGET_SURFACE", {}, .1,
                           target_z=target, reference_surface=reference)


def test_reference_envelope_excludes_unstable_models_not_average_them(geometry, patch_mesh):
    v, _ = patch_mesh
    references = {str(i): dict(z=np.full(81, height), holdout_rmse_over_width=.001,
                             angular_coverage=1.) for i, height in enumerate([.2, .3, .4])}
    references["unstable"] = dict(z=np.full(81, 100.), holdout_rmse_over_width=.2, angular_coverage=1.)
    envelope = geometry.reference_envelope(v, references, .1)
    np.testing.assert_allclose(envelope["minimum"], .2)
    np.testing.assert_allclose(envelope["median"], .3)
    np.testing.assert_allclose(envelope["maximum"], .4)
    assert "unstable" in envelope["excluded"]
    assert envelope["flagged"].all()
    plan = geometry.make_plan(maps_for(geometry, patch_mesh), "BOUNDARY_INTERPOLATED_CAP", {}, .1,
                              envelope=envelope)
    assert plan["status"] == "REFERENCE_MODEL_DISAGREEMENT"
    assert not plan["layers"]


def test_insufficient_reference_models_are_not_a_narrow_envelope(geometry, patch_mesh):
    v, _ = patch_mesh
    refs = {"one": dict(z=np.ones(81), holdout_rmse_over_width=.001, angular_coverage=1.)}
    assert not geometry.reference_envelope(v, refs, .1)["available"]


def test_uncertainty_veto_preserves_nominal_but_not_eligible_layers(geometry, patch_mesh):
    maps = maps_for(geometry, patch_mesh, base_uncertainty=.01)
    plan = geometry.make_plan(maps, "UNIFORM_OFFSET", dict(thickness=.012), .004)
    assert plan["status"] == "TARGET_SURFACE_UNSUPPORTED"
    assert not plan["layers"]
    assert len(plan["diagnostic_layers"]) == 3
    assert plan["no_print"].all()


def test_uncertainty_rule_uses_declared_layer_not_default_visualization_layer(geometry, patch_mesh):
    maps = maps_for(geometry, patch_mesh, base_uncertainty=.01)
    assert maps["no_print"].all()
    plan = geometry.make_plan(maps, "UNIFORM_OFFSET", dict(thickness=.3), .1)
    assert plan["status"] == "EXPLICIT_POLICY_VALID"
    assert len(plan["layers"]) == 3


def test_no_independent_subset_and_low_boundary_resolution_fail_separately(geometry, patch_mesh):
    v, f = patch_mesh
    missing = geometry.geometry_maps(v, f, None, 2., 200.)
    assert missing["status"] == "GEOMETRY_INCONCLUSIVE"
    coarse = geometry.geometry_maps(v, f, v.copy(), 2., 69.)
    assert coarse["status"] == "BOUNDARY_UNCERTAIN"
    assert coarse["no_print"].all()


def test_incomplete_support_and_missing_evidence_fail_closed(geometry, patch_mesh):
    maps = maps_for(geometry, patch_mesh)
    assert geometry.geometry_status(.94, 200., True, maps["topology"]) == "SURFACE_INCOMPLETE"
    assert geometry.geometry_status(np.nan, 200., True, maps["topology"]) == "GEOMETRY_INCONCLUSIVE"
    assert geometry.geometry_status(1., 200., True, {}) == "GEOMETRY_INCONCLUSIVE"


def test_cleaning_never_invents_surface_or_fills_holes(geometry, patch_mesh):
    v, f = patch_mesh
    duplicate = np.vstack([f, f[0], [0, 0, 1]])
    cleaned_v, cleaned_f, audit = geometry.clean_mesh(v, duplicate)
    assert len(cleaned_f) == len(f)
    assert audit["faces_removed"] == 2
    assert audit["inferred_vertices_added"] == 0
    np.testing.assert_allclose(cleaned_v, v)


def test_triangle_union_contains_holes_not_just_convex_hull(geometry, patch_mesh):
    v, f = patch_mesh
    center = v[f].mean(1)
    f = f[np.linalg.norm(center[:, :2], axis=1) > .3]
    query = np.array([[0., 0., 0.], [.8, .8, 0.], [1.1, 0., 0.]])
    assert geometry.projected_containment(query, v, f).tolist() == [False, True, False]


def test_all_exports_are_offline_and_explicitly_classified(geometry, patch_mesh, tmp_path):
    maps = maps_for(geometry, patch_mesh)
    plan = geometry.make_plan(maps, "UNIFORM_OFFSET", dict(thickness=.3), .1)
    metadata = geometry.export_plan(tmp_path, maps, plan)
    assert not metadata["physical_print_authorized"]
    assert "NOT_IDENTIFIABLE" in metadata["classification"]
    assert metadata["units"] == "UNCERTIFIED_DATASET_UNITS"
    assert len(metadata["candidate_layers"]) == 3
    assert (tmp_path/"deposition_maps.npz").exists()
    assert "calibration_target_v2.py" not in geometry.embedded_sources
    for source in geometry.embedded_sources.values():
        ast.parse(source)


def test_report_generation_includes_failures_and_qualified_volume_claims(geometry, patch_mesh, tmp_path):
    maps = maps_for(geometry, patch_mesh)
    plan = geometry.make_plan(maps, "UNIFORM_OFFSET", dict(thickness=.3), .1)
    folder = tmp_path/"sequences"/"fixture"
    metadata = geometry.export_plan(folder/"uniform", maps, plan)
    metadata["configuration"] = "uniform"
    np.savez(folder/"observed_geometry_maps.npz", vertices=maps["vertices"],
             uncertainty=maps["uncertainty"], no_print=maps["no_print"])
    np.savez(folder/"reference_disagreement_maps.npz", disagreement=np.full(81, .02))
    summary = dict(sequences=[dict(id="fixture", width_px=200, width_dataset_units=2.,
        geometry_status=maps["status"], plans=[metadata], envelope=dict(
            median_disagreement=.02, maximum_disagreement=.02, maximum_disagreement_in_layers=5.,
            flagged_fraction=1.))], synthetic_validation=dict(
                configurations=24, max_target_error=0., max_layer_error=0.))
    scope = {}
    exec(compile(geometry.embedded_sources["observable_report.py"], str(NOTEBOOK), "exec"), scope)
    scope["write_report"](tmp_path, summary)
    report = (tmp_path/"OBSERVABLE_GEOMETRY_REPORT.md").read_text()
    assert "Predecessor sealed as FAIL" in report
    assert "not anatomical ground" in report
    assert "0.0618138053%" in report
    assert (tmp_path/"policy_results.csv").exists()
