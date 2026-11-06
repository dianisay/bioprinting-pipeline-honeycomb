"""Numerical checks of notebook functions; fixtures are not reconstruction results."""

import ast
import hashlib
import json
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
import open3d as o3d
import pytest
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "04_rgb_wound_3d_reconstruction.ipynb"


@pytest.fixture(scope="module")
def functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    scope = dict(np=np, cv2=cv2, o3d=o3d, Delaunay=Delaunay, cKDTree=cKDTree, Path=Path,
                 math=__import__("math"), hashlib=hashlib, json=json,
                 distance_transform_edt=distance_transform_edt, PureWindowsPath=PureWindowsPath,
                 Rotation=Rotation)
    def definitions_in(body):
        for node in body:
            if isinstance(node, ast.FunctionDef):
                yield node
            elif isinstance(node, ast.If):
                yield from definitions_in(node.body)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            tree = ast.parse(source)
            if cell.get("metadata", {}).get("strict_measurement"):
                exec(compile(tree, str(NOTEBOOK), "exec"), scope)
                continue
            if cell.get("metadata", {}).get("calibration_safety") and "def c_fault_points(" in source:
                exec(compile(tree, str(NOTEBOOK), "exec"), scope)
                continue
            definitions = list(definitions_in(tree.body))
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    return scope


def test_notebook_all_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_calibration_router_refuses_failed_warning_and_inconclusive(functions):
    online = dict(coverage=1., stable_fraction=1., cross_view_iqr_over_width=0.,
                  wound_width_px=200)
    for decision in ("CALIBRATION_WARNING", "RGB_INTRINSICS_FAILED", "DEPTH_SCALE_FAILED",
                     "RGB_DEPTH_EXTRINSICS_FAILED", "SYNCHRONIZATION_FAILED",
                     "TARGET_NOT_OBSERVED", "CALIBRATION_INCONCLUSIVE",
                     "REFUSE_WOUND_RECONSTRUCTION"):
        result = functions["c_calibration_router"](dict(decision=decision), online)
        assert result["decision"] == "REFUSE_WOUND_RECONSTRUCTION"
        assert not result["reconstruction_accepted"]
        assert not result["print_authorized"]


def test_full3d_calibration_fault_identity_and_translation(functions):
    K = np.array([[200., 0., 40.], [0., 200., 40.], [0., 0., 1.]])
    points = np.array([[.1, .2, 1.], [-.2, .1, 1.2]])
    actual = functions["c_fault_points"](points, {}, K, 1.)
    np.testing.assert_allclose(actual, points, atol=1e-12)
    moved = functions["c_fault_points"](points, dict(translation_fraction=[.01, -.02, .03]), K, 1.)
    np.testing.assert_allclose(moved, points + [.01, -.02, .03], atol=1e-12)


def test_full3d_rotation_is_not_an_image_plane_warp(functions):
    K = np.array([[200., 0., 40.], [0., 200., 40.], [0., 0., 1.]])
    points = np.array([[.1, .2, 1.], [-.2, .1, 1.2]])
    actual = functions["c_fault_points"](points, dict(rotation_deg=[5., 0., 0.]), K, 1.)
    expected = points @ Rotation.from_euler("x", 5., degrees=True).as_matrix().T
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert not np.allclose(actual[:, 2], points[:, 2])


def test_depth_scale_changes_xyz_not_only_reported_z(functions):
    K = np.array([[200., 0., 40.], [0., 200., 40.], [0., 0., 1.]])
    points = np.array([[.1, .2, 1.], [-.2, .1, 1.2]])
    actual = functions["c_fault_points"](points, dict(depth_scale=.02), K, 1.)
    np.testing.assert_allclose(actual, points * 1.02, atol=1e-12)


def test_calibration_fault_transform_preserves_unknown_depth(functions):
    K = np.array([[200., 0., 40.], [0., 200., 40.], [0., 0., 1.]])
    points = np.array([[.1, .2, 1.], [np.nan, 1., 1.], [0., 0., -.1]])
    actual = functions["c_fault_points"](points, {}, K, 1.)
    np.testing.assert_allclose(actual[0], points[0], atol=1e-12)
    assert np.isnan(actual[1:]).all()


def test_calibration_pass_requires_observable_geometry_and_never_prints(functions):
    online = dict(coverage=.8, stable_fraction=.8, cross_view_iqr_over_width=.02,
                  wound_width_px=120)
    result = functions["c_calibration_router"](dict(decision="CALIBRATION_PASS"), online)
    assert result["decision"] == "RECONSTRUCTION_ACCEPTED_FOR_OFFLINE_EVALUATION"
    assert not result["print_authorized"]
    for key, bad in (("coverage", .69), ("stable_fraction", .69),
                     ("cross_view_iqr_over_width", .031), ("wound_width_px", 99)):
        result = functions["c_calibration_router"](
            dict(decision="CALIBRATION_PASS"), {**online, key: bad})
        assert not result["reconstruction_accepted"]


def test_calibration_dangerous_confusion_keeps_missing_geometry_explicit(functions):
    samples = [
        dict(volume_relative_error=.3, calibration_blocked=True),
        dict(volume_relative_error=.4, calibration_blocked=False),
        dict(volume_relative_error=.02, calibration_blocked=True),
        dict(volume_relative_error=.01, calibration_blocked=False),
        dict(volume_relative_error=None, calibration_blocked=True),
    ]
    result = functions["c_confusion"](samples, .20)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["unavailable"] == 1
    assert result["dangerous_count"] == 2
    assert result["false_negative_rate"] == .5
    assert result["precision"] == .5


def test_calibration_confusion_has_no_success_shaped_empty_denominator(functions):
    result = functions["c_confusion"]([
        dict(volume_relative_error=None, calibration_blocked=True)], .20)
    assert result["unavailable"] == 1
    assert result["dangerous_count"] == 0
    assert result["recall"] is None
    assert result["false_negative_rate"] is None


def test_calibration_router_does_not_read_ground_truth(functions):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = [ast.parse("".join(c["source"])) for c in notebook["cells"] if c["cell_type"] == "code"]
    function = next(n for source in sources for n in ast.walk(source)
                    if isinstance(n, ast.FunctionDef) and n.name == "c_calibration_router")
    names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
    assert not names & {"gtm", "gtbed", "source", "width", "gtpose"}
    constants = {n.value for n in ast.walk(function) if isinstance(n, ast.Constant)
                 and isinstance(n.value, str)}
    assert "volume_relative_error" not in constants
    assert "surface_gate" not in constants


def test_independent_calibration_detector_excludes_wound_and_fault_labels():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    trees = [ast.parse("".join(c["source"])) for c in notebook["cells"] if c["cell_type"] == "code"]
    detector = next(n for tree in trees for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "c_detect")
    arguments = {a.arg for a in detector.args.args}
    assert not arguments & {"fault", "fault_id", "source", "mesh", "gtpose", "gtbed", "wound"}
    constants = {n.value for n in ast.walk(detector) if isinstance(n, ast.Constant)
                 and isinstance(n.value, str)}
    assert not constants & {"volume_relative_error", "wound_width_dataset_units", "true_pose",
                            "true_extrinsics", "gtpose"}


def test_generalization_subsets_are_disjoint(functions):
    names = [f"image_{i:03}.png" for i in range(60)]
    for count in (10, 20, 30):
        subsets = functions["g_subsets"](names, count)
        assert len(subsets["A"]) == len(subsets["B"]) == count
        assert not set(subsets["A"]) & set(subsets["B"])
    assert functions["g_subsets"](names, 60) == {"ALL": names}
    with pytest.raises(ValueError, match="disjoint"):
        functions["g_subsets"](names, 40)


def test_generalization_depth_corruption_preserves_unknowns(functions):
    depth = np.arange(100, dtype=np.float32).reshape(10, 10) + 1
    depth[3, 4] = np.nan
    np.testing.assert_allclose(functions["g_corruption"](depth, {}), depth, equal_nan=True)
    shifted = functions["g_corruption"](depth, dict(offset_px=2, depth_scale=1.02))
    assert np.isnan(shifted[:, :2]).all()
    np.testing.assert_allclose(shifted[:, 2:], depth[:, :-2]*1.02, equal_nan=True)


def test_online_consensus_uses_observed_scale_not_ground_truth(functions):
    mask = np.zeros((10, 10), bool); mask[2:8, 2:8] = True
    yy, xx = np.where(mask)
    consensus = np.column_stack([xx, yy, np.full(len(xx), 3.),
                                 np.full(len(xx), .01), np.full((len(xx), 2), 20)])
    K = np.array([[100, 0, 5], [0, 100, 5], [0, 0, 1]], float)
    a = functions["g_online_consensus"](consensus, mask, K)
    consensus[:, 2:4] *= 10
    b = functions["g_online_consensus"](consensus, mask, K)
    assert a["stable_fraction"] == b["stable_fraction"] == 1.
    assert a["cross_view_iqr_over_width"] == pytest.approx(b["cross_view_iqr_over_width"])
    empty = functions["g_online_consensus"](np.empty((0, 6)), mask, K)
    assert empty["stable_fraction"] == 0.
    assert empty["cross_view_iqr_over_width"] == 1.


def test_batched_depth_fusion_exactly_matches_sealed_estimator(functions):
    rng = np.random.default_rng(7)
    uv = rng.integers(-2, 22, (16000, 2))
    z = rng.uniform(.1, 10, len(uv))
    points = np.column_stack([uv*z[:, None], z])
    views = rng.integers(0, 20, len(uv))
    region = np.ones((20, 20), bool); region[3:5] = False
    args = (points, views, np.eye(3), np.zeros(3), np.eye(3), region)
    np.testing.assert_array_equal(functions["g_depth_consensus"](*args),
                                  functions["per_pixel_pair_consensus"](*args))
    assert functions["g_depth_consensus"](points, views, np.eye(3), np.zeros(3),
                                            np.eye(3), np.zeros_like(region)).shape == (0, 6)


def test_observable_calibration_detector_does_not_require_geometry(functions):
    yy, xx = np.mgrid[:101, :101]
    mask = (xx-50)**2+(yy-50)**2 < 15**2
    depth = np.ones(mask.shape, np.float32); depth[mask] += .1
    aligned = functions["g_observable"](depth, mask)
    shifted = functions["g_observable"](functions["g_corruption"](depth, dict(offset_px=8)), mask)
    assert aligned["calibration_observable"]
    assert shifted["boundary_disagreement"] > aligned["boundary_disagreement"]
    unavailable = functions["g_observable"](np.full(mask.shape, np.nan), mask)
    assert not unavailable["calibration_observable"]
    assert unavailable["coverage"] == 0.


def test_generalization_router_is_offline_only_and_conservative(functions):
    quality = dict(calibration_observable=True, boundary_disagreement=.01, coverage=.9,
                   wound_width_px=120, view_count=30, stable_fraction=.95,
                   cross_view_iqr_over_width=.01)
    route = functions["g_router"]
    assert route(quality, True)["decision"] == "RECONSTRUCTION_ACCEPTED_FOR_OFFLINE_EVALUATION"
    assert route(quality, False)["decision"] == "REFUSE_TO_PRINT"
    assert route({**quality, "view_count": 10}, False)["decision"] == "REQUEST_MORE_RGB_VIEWS"
    assert route({**quality, "coverage": .5}, True)["decision"] == "REQUEST_DEPTH_RESCAN"
    assert route({**quality, "boundary_disagreement": .1}, True)["decision"] == "RGB_DEPTH_CALIBRATION_FAILED"
    assert route({**quality, "calibration_observable": False}, True)["decision"] == "RGB_DEPTH_CALIBRATION_FAILED"
    assert route({**quality, "wound_width_px": 60}, False)["decision"] == "INSUFFICIENT_BOUNDARY_RESOLUTION"
    assert route({**quality, "stable_fraction": .2}, False)["decision"] == "INSUFFICIENT_GEOMETRY"
    for has_depth in (False, True):
        for scale_valid in (False, True):
            assert not route(quality, has_depth, scale_valid)["print_authorized"]
            assert not route(quality, has_depth, scale_valid)["common_depth_scale_identifiable"]


def test_generalization_mask_freeze_blocks_mutation(functions, tmp_path):
    (tmp_path / "masks").mkdir()
    mask = tmp_path / "masks" / "mask.bin"; mask.write_bytes(b"reviewed RGB only")
    freeze = dict(phase="MASK_REVIEW_COMPLETE_BEFORE_GEOMETRY_UNLOCK",
                  files=[dict(path="masks\\mask.bin", sha256=hashlib.sha256(mask.read_bytes()).hexdigest())])
    (tmp_path / "mask_freeze.json").write_text(json.dumps(freeze))
    assert len(functions["g_verify_masks"](tmp_path)) == 64
    mask.write_bytes(b"geometry-informed revision")
    with pytest.raises(ValueError, match="Frozen"):
        functions["g_verify_masks"](tmp_path)


def test_generalization_gate_has_no_evaluation_metric_fallback():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"]
                       if c["cell_type"] == "code" and c.get("metadata", {}).get("sensor_generalization"))
    assert "g_online_consensus(consensus, mask, K)" in source
    assert "SENSOR_GENERALIZATION" in source
    assert 'row.get("cross_view_iqr_over_width")' not in source


def test_calibration_rejects_invented_or_invalid_values(functions):
    base = dict(K=[[100, 0, 50], [0, 100, 40], [0, 0, 1]], dist=[0]*5, image_size=[100, 80])
    with pytest.raises(ValueError, match="undocumented"):
        functions["validate_calibration"](base)
    base["source"] = "unit fixture"
    functions["validate_calibration"](base)
    base["K"][0][0] = -1
    with pytest.raises(ValueError):
        functions["validate_calibration"](base)


def test_masks_reject_rgb_render_and_wrong_size(functions, tmp_path):
    render = np.arange(100, dtype=np.uint8).reshape(10, 10)
    path = tmp_path / "mask.png"
    cv2.imwrite(str(path), render)
    with pytest.raises(ValueError, match="binary"):
        functions["binary_mask"](path, (10, 10))
    cv2.imwrite(str(path), np.full((10, 10), 255, np.uint8))
    assert functions["binary_mask"](path, (10, 10)).all()
    with pytest.raises(ValueError):
        functions["binary_mask"](path, (20, 20))


def test_visibility_does_not_label_occluded_back_surface(functions):
    points = np.array([[0, 0, 1], [0, 0, 2]], float)
    K = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]], float)
    poses = {"a": (np.eye(3), np.zeros(3)), "b": (np.eye(3), np.zeros(3))}
    masks = {n: np.ones((10, 10), bool) for n in poses}
    rings = {n: np.zeros((10, 10), bool) for n in poses}
    wound, ring = functions["classify_surface"](points, poses, K, masks, rings, (10, 10))
    assert wound.tolist() == [True, False]
    assert not ring.any()


def test_triangulation_recovers_known_point(functions):
    K = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], float)
    expected = np.array([0.1, 0.2, 3.0])
    poses = [(np.eye(3), np.array([x, 0, 0])) for x in (-0.3, 0, 0.3)]
    pixels = []
    for R, t in poses:
        p = K @ (R @ expected + t)
        pixels.append(p[:2] / p[2])
    actual, error = functions["triangulate_multiview"](pixels, poses, K)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert error < 1e-10


def test_zero_parallax_is_not_metric_evidence(functions):
    poses = [(np.eye(3), np.zeros(3))]*3
    with pytest.raises(ValueError, match="parallax"):
        functions["triangulate_multiview"]([[0,0]]*3, poses, np.eye(3))


def test_metric_marker_reconstructs_known_printed_side(functions):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    pattern = cv2.aruco.generateImageMarker(dictionary, 0, 160)
    pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    marker = np.array([[-.01,-.01,.35],[.01,-.01,.35],[.01,.01,.35],[-.01,.01,.35]])
    K = np.array([[800.,0,320],[0,800,240],[0,0,1]])
    images, poses = {}, {}
    source = np.float32([[0,0],[159,0],[159,159],[0,159]])
    for i,x in enumerate((-.03, 0, .03)):
        t = np.array([x,0,0])
        p = (marker+t)@K.T
        uv = (p[:,:2]/p[:,2,None]).astype(np.float32)
        transform = cv2.getPerspectiveTransform(source, uv)
        images[str(i)] = cv2.warpPerspective(pattern, transform, (640,480), borderValue=(255,255,255))
        poses[str(i)] = np.eye(3), t
    scale, details = functions["marker_scale"](images, poses, K, dict(
        dictionary="DICT_4X4_50", id=0, side_mm=20, source="analytic fixture, not a physical scan"))
    assert scale == pytest.approx(1000, rel=.03)
    assert details["scale_accuracy_independently_validated"] is False
    with pytest.raises(ValueError, match=">=3"):
        functions["marker_scale"]({}, {}, K, dict(
            dictionary="DICT_4X4_50", id=0, side_mm=20, source="fixture"))


def test_mesh_does_not_span_large_unsampled_gap(functions):
    x,y = np.meshgrid(np.linspace(0,1,6), np.linspace(0,1,6))
    island = np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    points = np.vstack([island,island+[10,0,0]])
    tri, _ = functions["supported_mesh"](points)
    edges = points[tri]-np.roll(points[tri],1,axis=1)
    assert np.linalg.norm(edges,axis=2).max() < 1


def test_known_box_surface_measurements_and_units(functions):
    local = np.array([[0, 0, -2], [4, 0, -2], [4, 3, -2], [0, 3, -2]], float)
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    flat = lambda xy: np.zeros(len(xy))
    result, depth, boundary = functions["measure_surface"](local, triangles, flat)
    assert result["perimeter"] == pytest.approx(14)
    assert result["surface_area"] == pytest.approx(12)
    assert result["max_depth"] == pytest.approx(2)
    assert result["mean_depth"] == pytest.approx(2)
    assert result["volume"] == pytest.approx(24)
    assert len(boundary) == 4
    scaled, _, _ = functions["measure_surface"](local*10, triangles, flat)
    assert scaled["perimeter"] == pytest.approx(result["perimeter"]*10)
    assert scaled["surface_area"] == pytest.approx(result["surface_area"]*100)
    assert scaled["volume"] == pytest.approx(result["volume"]*1000)


def test_reference_plane_and_quadratic_on_known_surface(functions):
    x, y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    skin = np.column_stack([x.ravel(), y.ravel(), (0.02*x*x+0.03*y*y).ravel()])
    origin, basis, predict, scores = functions["fit_reference"](skin, np.array([[0,0,10]]))
    local = (skin-origin)@basis
    assert scores["quadratic"] < 1e-10
    assert scores["plane"] > scores["quadratic"]
    np.testing.assert_allclose(predict(local[:, :2]), local[:, 2], atol=1e-10)


def test_similarity_is_evaluation_not_scale_validation(functions):
    rng = np.random.default_rng(42)
    source = rng.normal(size=(20, 3))
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    target = 3.4*source@R.T+np.array([3,5,7])
    scale, rot, trans = functions["similarity_alignment"](source, target)
    assert scale == pytest.approx(3.4)
    np.testing.assert_allclose(scale*source@rot.T+trans, target, atol=1e-12)


def test_dense_pair_column_translation_and_positive_depth(functions):
    # Rectified synthetic texture tests stereo algebra only, not wound accuracy.
    rng = np.random.default_rng(7)
    left = rng.integers(0, 256, (120, 320, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-12] = left[:, 12:]
    K = np.array([[240.,0,160],[0,240,60],[0,0,1]])
    poses = [(np.eye(3), np.zeros(3)), (np.eye(3), np.array([-0.1,0,0]))]
    points, colors, disparity = functions["dense_pair"](left, right, *poses, K, 320, 64)
    assert len(points) > 1000
    assert np.median(points[:, 2]) == pytest.approx(2, rel=0.01)
    assert colors.shape == points.shape
    assert np.nanmedian(disparity) == pytest.approx(12, abs=0.1)


def test_dense_pair_vertical_epipolar_geometry(functions):
    rng = np.random.default_rng(17)
    left = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:-12] = left[12:]
    K = np.array([[240., 0, 160], [0, 240, 120], [0, 0, 1]])
    poses = [(np.eye(3), np.zeros(3)), (np.eye(3), np.array([0, -0.1, 0]))]
    points, colors, disparity = functions["dense_pair"](left, right, *poses, K, 320, 64)
    assert len(points) > 1000
    assert np.median(points[:, 2]) == pytest.approx(2, rel=0.01)
    assert colors.shape == points.shape
    assert np.nanmedian(disparity) == pytest.approx(12, abs=0.1)
    with pytest.raises(ValueError, match="Reversed"):
        functions["dense_pair"](right, left, *poses[::-1], K, 320, 64)


def test_independent_surface_metrics_and_invalid_inputs(functions):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        o3d.utility.Vector3iVector([[0, 1, 2]]))
    prediction = np.array([[0.2, 0.2, 0.1], [0.3, 0.3, 0.1]])
    reference = np.array([[0.2, 0.2, 0.]])
    result = functions["surface_metrics"](prediction, mesh, reference, 0.15)
    assert result["point_to_surface_mean"] == pytest.approx(0.1, abs=1e-7)
    assert result["completeness"] == 1
    assert result["predicted_points"] == 2
    assert "not certified" in result["units"]
    with pytest.raises(ValueError, match="nonempty"):
        functions["surface_metrics"](np.empty((0, 3)), mesh, reference, 0.15)
    with pytest.raises(ValueError, match="finite"):
        functions["surface_metrics"](prediction, mesh, reference, float("nan"))


def test_camera_evaluation_alignment_preserves_scale_caveat(functions):
    centers = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    source = {str(i): (np.eye(3), -v) for i, v in enumerate(centers)}
    target = {str(i): (np.eye(3), -(2*v+np.array([3, 4, 5]))) for i, v in enumerate(centers)}
    scale, rotation, translation, receipt = functions["similarity_from_cameras"](source, target)
    assert scale == pytest.approx(2)
    np.testing.assert_allclose(scale*centers@rotation.T+translation, 2*centers+[3, 4, 5])
    assert "NOT independent metric scale" in receipt["alignment"]
    coincident = {str(i): (np.eye(3), np.zeros(3)) for i in range(4)}
    with pytest.raises(ValueError, match="collinear/coincident"):
        functions["similarity_from_cameras"](coincident, target)


def test_reviewed_annotations_cannot_transfer_to_another_rgb(functions, tmp_path):
    path = tmp_path / "rgb.png"
    cv2.imwrite(str(path), np.full((20, 20, 3), 50, np.uint8))
    record = dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  polygon=[[4,4],[12,4],[12,12],[4,12]])
    image, polygon = functions["verify_annotation_rgb"](path, record)
    assert image.shape == (20, 20, 3)
    assert polygon.shape == (4, 2)
    cv2.imwrite(str(path), np.full((20, 20, 3), 51, np.uint8))
    with pytest.raises(ValueError, match="differs"):
        functions["verify_annotation_rgb"](path, record)


def test_correlated_samples_from_one_pair_do_not_establish_consensus(functions):
    points = np.tile([0., 0., 3.], (100, 1))
    K = np.array([[10.,0,5],[0,10,5],[0,0,1]])
    region = np.ones((10, 10), bool)
    result = functions["per_pixel_pair_consensus"](
        points, np.zeros(100, int), np.eye(3), np.zeros(3), K, region)
    assert result.shape == (0, 6)


def test_consensus_reports_disagreement_between_distinct_pairs(functions):
    points = np.array([[0,0,z] for z in [2.,2.,3.,3.,4.,4.]])
    K = np.array([[10.,0,5],[0,10,5],[0,0,1]])
    result = functions["per_pixel_pair_consensus"](
        points, np.array([0,0,1,1,2,2]), np.eye(3), np.zeros(3), K, np.ones((10,10),bool))
    assert result.shape == (1, 6)
    np.testing.assert_allclose(result[0, :5], [5,5,3,1,3])


def test_mesh_components_exposes_disconnected_wound_fragments(functions):
    assert functions["mesh_components"](np.array([[0,1,2],[2,1,3]])) == 1
    assert functions["mesh_components"](np.array([[0,1,2],[3,4,5]])) == 2


def test_camera_reference_fits_without_rotating_into_depth_noise(functions):
    x,y = np.meshgrid(np.linspace(-1,1,15),np.linspace(-1,1,15))
    points = np.column_stack([x.ravel(),y.ravel(),np.full(x.size,3.)])
    origin,basis,reference,scores,kind = functions["camera_frame_reference"](
        points,np.eye(3),np.zeros(3))
    assert kind == "plane"
    assert max(scores.values()) < 1e-10
    local = (points-origin)@basis
    np.testing.assert_allclose(reference(local[:,:2]),local[:,2],atol=1e-10)


def test_native_pair_reduction_uses_median_not_repeated_sample_votes(functions):
    K = np.array([[10., 0, 5], [0, 10, 5], [0, 0, 1]])
    points = np.array([[0., 0., z] for z in [2, 2, 4, 10]])
    reduced = functions["native_pair_pixel_medians"](
        points, np.eye(3), np.zeros(3), K, np.ones((10, 10), bool))
    np.testing.assert_allclose(reduced, [[0, 0, 3]])
    consensus = functions["per_pixel_pair_consensus"](
        reduced, np.array([0]), np.eye(3), np.zeros(3), K, np.ones((10, 10), bool))
    assert consensus.shape == (0, 6)


def test_native_pair_reduction_preserves_original_consensus(functions):
    K = np.array([[10., 0, 5], [0, 10, 5], [0, 0, 1]])
    region = np.ones((10, 10), bool)
    points = np.array([[0., 0., z] for z in [2., 6., 3., 4., 7., 9.]])
    pairs = np.repeat(np.arange(3), 2)
    original = functions["per_pixel_pair_consensus"](
        points, pairs, np.eye(3), np.zeros(3), K, region)
    reduced = np.concatenate([functions["native_pair_pixel_medians"](
        points[pairs == p], np.eye(3), np.zeros(3), K, region) for p in range(3)])
    actual = functions["per_pixel_pair_consensus"](
        reduced, np.arange(3), np.eye(3), np.zeros(3), K, region)
    np.testing.assert_allclose(original, actual)


def test_native_alignment_rigid_never_optimizes_scale(functions):
    points = np.array([[0., 0., 0.], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    target = 4*points + [3, 2, 1]
    rigid = functions["native_fit_correspondences"](points, target, False)
    similar = functions["native_fit_correspondences"](points, target, True)
    assert np.linalg.det(rigid[:3, :3]) == pytest.approx(1)
    assert np.cbrt(np.linalg.det(similar[:3, :3])) == pytest.approx(4)
    np.testing.assert_allclose(functions["native_transform"](points, similar), target)
    with pytest.raises(ValueError, match="non-collinear"):
        functions["native_fit_correspondences"](np.zeros((4, 3)), target, True)


def test_native_gate_missing_evidence_never_passes(functions):
    good = dict(registered_fraction=1., stable_mask_fraction_all_pixels=.75,
                surface_retained_fraction=.95, scanner_bed_completeness_1mm=.85,
                median_cross_pair_iqr_over_width=.1, surface_components=1,
                scanner_bed_mean_error_mm=.8)
    repeat = dict(mean_disagreement_over_width=.04)
    assert functions["native_gate"](good, repeat)["status"] == "PASS"
    assert functions["native_gate"]({**good, "scanner_bed_completeness_1mm": None}, repeat)["status"] != "PASS"
    assert functions["native_gate"](good)["status"] != "PASS"
    assert functions["native_gate"]({**good, "surface_components": 0}, repeat)["status"] == "FAILED"
    assert functions["native_gate"]({**good, "scanner_bed_mean_error_mm": 1.05}, repeat)["status"] == "PARTIAL"
    assert functions["native_gate"]({**good, "stable_mask_fraction_all_pixels": .01}, repeat)["status"] == "FAILED"


def test_sensor_router_uses_capabilities_and_never_authorizes_printing(functions):
    caps = dict(has_rgb=True, calibration_valid=True, has_depth=True,
                depth_aligned_to_rgb=True, metric_scale_valid=False,
                depth_scale_consistent=True, has_camera_poses=True,
                synthetic_depth=True, high_res_rgb=True)
    quality = dict(wound_width_px=200, sharpness=100, depth_coverage=.9,
                   depth_noise_over_width=.005, rgb_depth_boundary_disagreement=.01,
                   stable_fraction=.9, completeness=.95, components=1)
    decision = functions["sensor_route"](caps, quality)
    assert decision["route"] == "HIGH_RES_RGB_PLUS_DEPTH"
    assert decision["decision"] == "INSUFFICIENT_SCALE"
    assert not decision["print_authorized"]
    poor = functions["sensor_route"](caps, {**quality, "depth_coverage": .4})
    assert poor["route"] == "RGB_POSE_ASSISTED"
    assert poor["fallback"] == "FALLBACK_TO_RGB"
    shifted = functions["sensor_route"](caps, {**quality, "rgb_depth_boundary_disagreement": .2})
    assert shifted["decision"] == "REFUSE_TO_PRINT"
    assert not shifted["print_authorized"]
    synthetic = functions["sensor_route"]({**caps, "metric_scale_valid": True}, quality)
    assert synthetic["decision"] == "REFUSE_TO_PRINT"
    with pytest.raises(ValueError, match="Incomplete"):
        functions["sensor_route"]({}, quality)


def test_sensor_mesh_rendering_is_camera_z_not_euclidean_range(functions):
    vertices = np.array([[-2., -2., 2.], [2., -2., 2.], [2., 2., 2.], [-2., 2., 2.]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
    K = np.array([[10., 0, 5], [0, 10, 5], [0, 0, 1]])
    depth, valid, normals = functions["sensor_render"](mesh, K, (np.eye(3), np.zeros(3)), (10, 10))
    assert valid.all()
    np.testing.assert_allclose(depth, 2., atol=1e-6)
    points, uv = functions["sensor_unproject"](depth, K, (np.eye(3), np.zeros(3)))
    np.testing.assert_allclose(points[:, 2], 2., atol=1e-6)
    assert np.linalg.norm(points[0]) > depth[0, 0]
    assert normals.shape == (10, 10, 3)


def test_sensor_corruption_is_separate_reproducible_and_preserves_clean_reference(functions):
    depth = np.full((20, 20), 2., np.float32); depth[:2] = np.nan
    clean = depth.copy()
    rgb = np.full((20, 20, 3), 80, np.uint8)
    profile = dict(downsample=2, noise_over_width=.01, missing_fraction=.2)
    a, meta = functions["sensor_corrupt"](depth, rgb, .1, profile, 42)
    b, _ = functions["sensor_corrupt"](depth, rgb, .1, profile, 42)
    np.testing.assert_allclose(a, b, equal_nan=True)
    np.testing.assert_allclose(depth, clean, equal_nan=True)
    assert meta["synthetic"]
    assert "NOT measured D405" in meta["noise_calibration"]


def strict_observable_fixture():
    return dict(stable_fraction=1., retained_fraction=1., components=1, wound_width_px=150.,
        reference_rmse_over_observed_width=.001, depth_signal_to_uncertainty=10.,
        volume_uncertainty_fraction=.05, volume_positive=True, healthy_samples=100,
        subset_volume_disagreement=.01, subset_depth_disagreement=.01)


@pytest.mark.parametrize("field,value,expected", [
    ("stable_fraction", .79, "WOUND_SURFACE_INCOMPLETE"),
    ("retained_fraction", .89, "WOUND_SURFACE_INCOMPLETE"),
    ("reference_rmse_over_observed_width", .021, "REFERENCE_SURFACE_UNSTABLE"),
    ("depth_signal_to_uncertainty", 4.9, "SHALLOW_DEPTH_UNRELIABLE"),
    ("wound_width_px", 99., "SHALLOW_DEPTH_UNRELIABLE"),
    ("volume_uncertainty_fraction", .11, "VOLUME_ESTIMATE_UNSUPPORTED"),
    ("subset_volume_disagreement", .11, "MEASUREMENT_INCONCLUSIVE"),
    ("subset_depth_disagreement", .11, "MEASUREMENT_INCONCLUSIVE")])
def test_strict_calibration_pass_never_overrides_measurement_failure(functions, field, value, expected):
    gate = functions["s_measurement_gate"]({**strict_observable_fixture(), field: value})
    assert gate["decision"] == expected
    router = functions["s_two_gate"]("CALIBRATION_PASS", gate)
    assert not router["reconstruction_accepted"]
    assert not router["print_authorized"]


@pytest.mark.parametrize("decision", ["CALIBRATION_WARNING", "EXTRINSICS_FAILED",
    "INTRINSICS_FAILED", "DEPTH_SCALE_FAILED", "SYNCHRONIZATION_FAILED",
    "CALIBRATION_INCONCLUSIVE", "RGB_DEPTH_EXTRINSICS_FAILED", "RGB_INTRINSICS_FAILED",
    "TARGET_NOT_OBSERVED"])
def test_strict_two_gates_reject_nonpassing_calibration(functions, decision):
    gate = functions["s_measurement_gate"](strict_observable_fixture())
    assert gate["supported"]
    router = functions["s_two_gate"](decision, gate)
    assert not router["reconstruction_accepted"]
    assert not router["print_authorized"]


def test_strict_missing_or_nonfinite_measurement_is_not_success(functions):
    for value in (None, np.nan, np.inf):
        result = functions["s_measurement_gate"]({**strict_observable_fixture(), "stable_fraction": value})
        assert result["decision"] == "MEASUREMENT_INCONCLUSIVE"
    fixture = strict_observable_fixture()
    del fixture["subset_volume_disagreement"]
    assert not functions["s_measurement_gate"](fixture)["supported"]


def test_strict_error_decomposition_does_not_attribute_clean_bias_to_calibration(functions):
    error = functions["s_error_triplet"](1.2, 1.1, 1.)
    assert error["TOTAL_MEASUREMENT_ERROR"] == pytest.approx(.2)
    assert error["CLEAN_BASELINE_ERROR"] == pytest.approx(.1)
    assert error["INCREMENTAL_ERROR_DUE_TO_CALIBRATION_FAULT"] == pytest.approx(.1 / 1.1)
    unavailable = functions["s_error_triplet"](None, 1.1, 1.)
    assert unavailable["TOTAL_MEASUREMENT_ERROR"] is None
    assert unavailable["CLEAN_BASELINE_ERROR"] == pytest.approx(.1)
    assert unavailable["INCREMENTAL_ERROR_DUE_TO_CALIBRATION_FAULT"] is None
    assert functions["s_error_triplet"](1., 0., 1.)["INCREMENTAL_ERROR_DUE_TO_CALIBRATION_FAULT"] is None


def test_strict_clean_biased_measurement_fails_independently(functions):
    row = dict(surface_gate=True, volume_relative_error=.11, max_depth_relative_error=.01)
    gate = functions["s_research_measurement_validity"](row)
    assert gate["decision"] == "VOLUME_ESTIMATE_UNSUPPORTED"
    assert gate["evaluation_only_uses_source_reference"]
    assert not functions["s_two_gate"]("CALIBRATION_PASS", gate)["reconstruction_accepted"]


def test_strict_measurement_policy_is_immutable(functions, tmp_path):
    before = functions["s_freeze_measurement_policy"](tmp_path)
    path = tmp_path / "measurement_policy.json"
    timestamp = path.stat().st_mtime_ns
    assert functions["s_freeze_measurement_policy"](tmp_path) == before
    assert path.stat().st_mtime_ns == timestamp
    value = json.loads(path.read_text())
    value["volume_uncertainty_fraction_max"] = .11
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="policy changed"):
        functions["s_freeze_measurement_policy"](tmp_path)


def test_strict_online_gate_ignores_ground_truth_and_fault_labels(functions):
    fixture = strict_observable_fixture()
    expected = functions["s_measurement_gate"](fixture)
    actual = functions["s_measurement_gate"]({**fixture, "source_mesh": "FORBIDDEN",
        "fault_class": "SECRET", "true_volume_error": 1e6, "true_target_pose": "FORBIDDEN"})
    assert actual == expected
    assert not actual["systematic_bias_certified"]


def strict_embedded_sources():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("metadata", {}).get("strict_holdout_runtime")
                and c["cell_type"] == "code" and "STRICT_EXPERIMENT_SOURCES =" in "".join(c["source"]))
    source = "".join(cell["source"])
    assert hashlib.sha256(source.encode()).hexdigest() == cell["metadata"]["source_sha256"]
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "STRICT_EXPERIMENT_SOURCES"
                              for t in n.targets))
    return ast.literal_eval(assignment.value)


def test_strict_embeds_frozen_original_detector_and_guarded_label_unlock():
    sources = strict_embedded_sources()
    target = sources["strict_target.py"]
    assert "607ec00b09717810d0829a027253256f221388852fa2856ff7a57e379766d11c" in target
    assert "d704b1d05803f7ceb814786a01ba9517ad54d765bcca833edd6e845da2d75726" in target
    unlock = next(n for n in ast.parse(target).body if isinstance(n, ast.FunctionDef)
                  and n.name == "s_unlock_manifest")
    assert "_verify_decisions(base)" in ast.unparse(unlock)
    for source in sources.values():
        ast.parse(source)


def test_strict_recall_preserves_known_catastrophe_with_unavailable_baseline():
    from collections import defaultdict
    sources = strict_embedded_sources()
    function = next(n for n in ast.parse(sources["strict_report.py"]).body
                    if isinstance(n, ast.FunctionDef) and n.name == "s_confusion")
    scope = dict(defaultdict=defaultdict)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    row = dict(id="DEVELOPMENT_ONLY", is_calibration_fault=True, scope="FAULT", fault_detected=False,
        volume_error_decomposition=dict(TOTAL_MEASUREMENT_ERROR=.25,
            INCREMENTAL_ERROR_DUE_TO_CALIBRATION_FAULT=None))
    result = scope["s_confusion"]([row], "volume", .2, "total_or_incremental")
    assert result["dangerous_configurations"] == 1
    assert result["recall"] == 0.
    row["volume_error_decomposition"]["TOTAL_MEASUREMENT_ERROR"] = .15
    result = scope["s_confusion"]([row], "volume", .2, "total_or_incremental")
    assert result["measurement_unavailable_subsets"] == 1
    assert result["recall"] is None
