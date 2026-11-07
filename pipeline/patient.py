"""Patient-level preview: the unified pipeline on a public Syn3DWound leg.

This module carries the honeycomb planner from the synthetic five-case scenes
onto a rendered human leg (Syn3DWound subject_1,
``Syn3DWound_0415_leg_strong_lighting_phi``). The stages are:

1. OBSERVE: 40 rendered RGB views, their wound masks, the COLMAP cameras and
   the dataset's cropped leg mesh (mesh-rendered geometry, i.e. the aligned
   RGB-D route; no RGB-only reconstruction is used here).
2. IDENTIFY: transfer the 2D wound masks onto the 3D mesh by projecting every
   vertex into every view and taking a majority vote.
3. SURFACE: build a local skin frame around the wound and resample the mesh as
   a height field (a grid of heights above a tangent plane).
4. TARGET: fit a quadratic surface to the healthy ring around the wound. This
   is an illustrative operator-accepted target, not anatomical truth; the
   sealed architecture decision requires the operator to supply or accept it.
5. HONEYCOMB + STAGED FILL: the same lattice, supported-cell rule, per-cell
   normals, toolpath construction and stop rules as ``pipeline.unified``.
6. PLACE: map the scaffold and toolpath back to the leg's 3D coordinates and
   project them into the original photograph.

Coordinates are Syn3DWound dataset units (DU). The dataset does not certify a
physical scale, so 1 DU is not claimed to be 1 metre. Printing stays disabled.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.interpolate import griddata
from scipy.sparse.csgraph import connected_components

from modules.conformal_mapping import compute_nozzle_orientations
from modules.honeycomb import hexagon_perimeter
from pipeline.coverage import plan_patient_coverage
from pipeline.unified import _cell_normals, _greedy_order, build_honeycomb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "data" / "rgb_wound" / "syn3dwound_leg"
SEQUENCE_ID = "Syn3DWound/subject_1/Syn3DWound_0415_leg_strong_lighting_phi"

PATIENT_DEFAULTS = {
    "mask_threshold": 40,              # wound pixels in the shaded masks are brighter than 40 / 255
    "vote_fraction": 0.5,              # a vertex is wound if > 50 % of the views that see it say so
    "frame_radius_factor": 2.5,        # skin frame fitted within 2.5 wound radii
    "ring_width_factor": 0.4,          # healthy ring: band up to 0.4 wound radii outside the wound edge
    "grid_pixel_du": 0.0001,           # height-field resolution (0.1 mDU)
    "grid_half_width_factor": 1.5,     # grid covers +/- 1.5 wound radii
    "layer_thickness_du": 0.0002,
    "measurement_uncertainty_du": 0.00002,
    "max_uncertainty_fraction_of_layer": 0.5,
    "safety_margin_fraction": 0.1,
    "honeycomb_cells_across_narrow_axis": 6,  # hexagon size set by the wound's narrow width
    "perimeter_shrink": 0.9,
    "perimeter_points_per_edge": 4,
    "tool_to_surface_offset_du": 0.0004,
    "travel_rise_du": 0.0008,
    "max_stages": 10,
    "seed": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    face_normal = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normal)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def read_colmap_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        model, params = parts[1], [float(v) for v in parts[4:]]
        if model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        elif model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        else:
            raise ValueError(f"Unsupported COLMAP camera model {model}")
        cameras[int(parts[0])] = {"width": int(parts[2]), "height": int(parts[3]),
                                  "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])}
    return cameras


def read_colmap_images(path: Path) -> list[dict]:
    """Parse COLMAP ``images.txt``; the pose maps world points into the camera."""
    images = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 10 or line.startswith("#"):
            continue
        number = int(parts[9].split("_")[-1])
        images.append({
            "image_id": int(parts[0]),
            "R": quaternion_to_matrix([float(v) for v in parts[1:5]]),
            "t": np.array([float(v) for v in parts[5:8]]),
            "camera_id": int(parts[8]),
            "file_stem": f"{number:03d}",
        })
    return images


def load_scene(scene_dir: Path = DEFAULT_SCENE) -> dict:
    scene_dir = Path(scene_dir)
    mesh_path = scene_dir / "cropped_mesh.obj"
    colmap = sorted(scene_dir.glob("*_colmap"))
    if not mesh_path.is_file() or not colmap:
        raise FileNotFoundError(
            f"Syn3DWound leg scene not found under {scene_dir}. Download the public sequence "
            f"{SEQUENCE_ID} (see docs/dataset_provenance.md) and place cropped_mesh.obj, images/, "
            "mask/ and the *_colmap/ folder there."
        )
    images_txt = next(colmap[0].rglob("images.txt"))
    cameras_txt = next(colmap[0].rglob("cameras.txt"))
    raw_vertices, faces = read_obj(mesh_path)
    # Syn3DWound meshes are stored y-up; COLMAP poses use the (x, -z, y) frame.
    vertices = raw_vertices[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    views = []
    for record in read_colmap_images(images_txt):
        rgb = scene_dir / "images" / f"image_{record['file_stem']}.png"
        mask = scene_dir / "mask" / f"mask_{record['file_stem']}.png"
        if rgb.is_file() and mask.is_file():
            views.append({**record, "rgb_path": rgb, "mask_path": mask})
    if not views:
        raise FileNotFoundError(f"No image/mask pairs matching {images_txt} were found in {scene_dir}")
    return {
        "scene_dir": scene_dir,
        "mesh_path": mesh_path,
        "images_txt": images_txt,
        "cameras_txt": cameras_txt,
        "vertices": vertices,
        "faces": faces,
        "normals": vertex_normals(vertices, faces),
        "cameras": read_colmap_cameras(cameras_txt),
        "views": views,
    }


def project(points: np.ndarray, view: dict, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    cam = points @ view["R"].T + view["t"]
    uvw = cam @ camera["K"].T
    return uvw[:, :2] / uvw[:, 2:3], cam[:, 2]


def camera_centre(view: dict) -> np.ndarray:
    return -view["R"].T @ view["t"]


def label_wound_vertices(scene: dict, params: dict = PATIENT_DEFAULTS) -> dict:
    """Majority vote of the 2D masks over the views in which each vertex faces the camera."""
    vertices, normals = scene["vertices"], scene["normals"]
    votes = np.zeros(len(vertices))
    seen = np.zeros(len(vertices))
    for view in scene["views"]:
        camera = scene["cameras"][view["camera_id"]]
        mask = cv2.imread(str(view["mask_path"]), cv2.IMREAD_GRAYSCALE)
        uv, depth = project(vertices, view, camera)
        cols, rows = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
        inside = (depth > 0) & (cols >= 0) & (rows >= 0) & (cols < mask.shape[1]) & (rows < mask.shape[0])
        facing = inside & (np.einsum("ij,ij->i", normals, camera_centre(view) - vertices) > 0)
        is_wound = np.zeros(len(vertices), dtype=bool)
        is_wound[inside] = mask[rows[inside], cols[inside]] > params["mask_threshold"]
        votes += is_wound & facing
        seen += facing
    wound = (seen > 0) & (votes > params["vote_fraction"] * np.maximum(seen, 1))
    wound = largest_mesh_component(wound, scene["faces"])
    return {"wound": wound, "votes": votes, "seen": seen}


def largest_mesh_component(selected: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = edges[selected[edges[:, 0]] & selected[edges[:, 1]]]
    n = len(selected)
    graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    if not selected.any():
        return selected
    biggest = np.bincount(labels[selected]).argmax()
    return selected & (labels == biggest)


def local_frame(vertices: np.ndarray, wound: np.ndarray, toward: np.ndarray, radius_factor: float) -> dict:
    """Tangent plane (e1, e2) and outward normal n from PCA of the skin around the wound."""
    centre = vertices[wound].mean(axis=0)
    radius = float(np.linalg.norm(vertices[wound] - centre, axis=1).max())
    near = np.linalg.norm(vertices - centre, axis=1) < radius_factor * radius
    _, _, vt = np.linalg.svd(vertices[near] - vertices[near].mean(axis=0))
    normal = vt[2]
    if normal @ (toward - centre) < 0:
        normal = -normal
    # e1 follows the wound's long axis inside the tangent plane, so the grid is aligned with the wound.
    flat = vertices[wound] - centre
    flat -= np.outer(flat @ normal, normal)
    e1 = np.linalg.svd(flat, full_matrices=False)[2][0]
    e1 = e1 - (e1 @ normal) * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return {"origin": centre, "e1": e1, "e2": e2, "normal": normal, "wound_radius": radius}


def to_local(points: np.ndarray, frame: dict) -> np.ndarray:
    rel = points - frame["origin"]
    return np.column_stack((rel @ frame["e1"], rel @ frame["e2"], rel @ frame["normal"]))


def to_world(local: np.ndarray, frame: dict) -> np.ndarray:
    return (frame["origin"] + local[:, :1] * frame["e1"] + local[:, 1:2] * frame["e2"]
            + local[:, 2:3] * frame["normal"])


def height_field(local: np.ndarray, wound: np.ndarray, frame: dict, params: dict = PATIENT_DEFAULTS) -> dict:
    pixel = params["grid_pixel_du"]
    half = params["grid_half_width_factor"] * frame["wound_radius"]
    axis = np.arange(-half, half + pixel / 2, pixel)
    gx, gy = np.meshgrid(axis, axis)
    keep = np.linalg.norm(local[:, :2], axis=1) < 2 * half
    height = griddata(local[keep, :2], local[keep, 2], (gx, gy), method="linear")
    wound_fraction = griddata(local[keep, :2], wound[keep].astype(float), (gx, gy), method="linear")
    wound_mask = np.nan_to_num(wound_fraction) >= 0.5
    count, labels, stats, _ = cv2.connectedComponentsWithStats(wound_mask.astype(np.uint8))
    if count > 1:
        wound_mask = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    outside_distance_px = cv2.distanceTransform((~wound_mask).astype(np.uint8), cv2.DIST_L2, 5)
    ring_px = params["ring_width_factor"] * frame["wound_radius"] / pixel
    ring = ~wound_mask & np.isfinite(height) & (outside_distance_px <= ring_px)
    ys, xs = np.nonzero(wound_mask)
    narrow_px = float(min(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
    return {
        "axis": axis,
        "gx": gx,
        "gy": gy,
        "pixel_du": pixel,
        "height": height,
        "valid": np.isfinite(height),
        "wound_mask": wound_mask,
        "ring_mask": ring,
        "centroid_px": [float(xs.mean()), float(ys.mean())],
        "boundary_width_px": float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1),
        "narrow_width_px": narrow_px,
    }


def quadratic_design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))


def fit_quadratic_target(field: dict) -> dict:
    """Least-squares quadratic over the healthy ring: the illustrative operator target."""
    ring = field["ring_mask"]
    x, y, z = field["gx"][ring], field["gy"][ring], field["height"][ring]
    coef, *_ = np.linalg.lstsq(quadratic_design(x, y), z, rcond=None)
    residual = z - quadratic_design(x, y) @ coef
    target = (quadratic_design(field["gx"].ravel(), field["gy"].ravel()) @ coef).reshape(field["gx"].shape)
    return {"coefficients": coef, "target": target, "ring_rms_du": float(np.sqrt(np.mean(residual**2))),
            "ring_pixels": int(ring.sum())}


def _supported(field: dict, uncertainty: np.ndarray, params: dict) -> np.ndarray:
    mask = field["wound_mask"]
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    margin = params["safety_margin_fraction"] * field["boundary_width_px"] / 2.0
    supported = field["valid"] & (distance >= margin)
    return supported & (uncertainty <= params["max_uncertainty_fraction_of_layer"] * params["layer_thickness_du"])


def _toolpath(cells, honeycomb, planned_top, normals, field, params, frame) -> dict:
    pixel = field["pixel_du"]
    side_px = honeycomb["hex_side_m"] / pixel * params["perimeter_shrink"]
    order = _greedy_order(honeycomb["centres_px"][cells]) if cells else []
    local_pts, local_normals, cell_ids, flags = [], [], [], []
    for rank in order:
        cell = cells[rank]
        centre = honeycomb["centres_px"][cell]
        outline = hexagon_perimeter(centre, side_px, params["perimeter_points_per_edge"])
        pts_px = np.vstack((outline, outline[:1], centre[None, :]))
        cols = np.clip(np.rint(pts_px[:, 0]).astype(int), 0, planned_top.shape[1] - 1)
        rows = np.clip(np.rint(pts_px[:, 1]).astype(int), 0, planned_top.shape[0] - 1)
        xy = np.column_stack((field["axis"][cols], field["axis"][rows]))
        surface = np.column_stack((xy, planned_top[rows, cols]))
        n = normals[cell]
        tool = surface + n * params["tool_to_surface_offset_du"]
        rise = np.array([0.0, 0.0, params["travel_rise_du"]])
        sequence = [(tool[0] + rise, 0)] + [(p, 1) for p in tool] + [(tool[-1] + rise, 0)]
        for point, flag in sequence:
            local_pts.append(point)
            local_normals.append(n)
            cell_ids.append(cell)
            flags.append(bool(flag))
    if not local_pts:
        empty = np.empty((0, 3))
        return {"local": empty, "world": empty, "normals_world": empty, "cell": np.empty(0, int),
                "deposit_flag": np.empty(0, bool), "rotations": np.empty((3, 3, 0))}
    local = np.asarray(local_pts)
    normals_local = np.asarray(local_normals)
    basis = np.vstack((frame["e1"], frame["e2"], frame["normal"]))
    normals_world = normals_local @ basis
    return {
        "local": local,
        "world": to_world(local, frame),
        "normals_world": normals_world,
        "cell": np.asarray(cell_ids),
        "deposit_flag": np.asarray(flags),
        "rotations": compute_nozzle_orientations(normals_world.T),
    }


def staged_fill(field: dict, target: np.ndarray, frame: dict, params: dict = PATIENT_DEFAULTS) -> dict:
    """Same rules as ``pipeline.unified.staged_honeycomb_fill`` with a spatially varying target."""
    layer = params["layer_thickness_du"]
    rng = np.random.default_rng(params["seed"])
    sigma = params["measurement_uncertainty_du"] / 3.0
    uncertainty = np.where(field["valid"], params["measurement_uncertainty_du"], np.inf)
    supported = _supported(field, uncertainty, params)
    cells_across = params["honeycomb_cells_across_narrow_axis"] * field["boundary_width_px"] / field["narrow_width_px"]
    honeycomb = build_honeycomb(field["wound_mask"], field["centroid_px"], field["pixel_du"], cells_across)
    labels = honeycomb["labels"]
    n_cells = len(honeycomb["centres_px"])
    fully = np.ones(n_cells, dtype=bool)
    fully[np.unique(labels[~supported])] = False
    fully &= np.bincount(labels.ravel(), minlength=n_cells) > 0
    in_lattice = fully[labels]

    true_height = field["height"].copy()
    measured = field["height"].copy()
    deposited_total = np.zeros_like(true_height)
    stages, stop_reason = [], None
    for stage in range(1, params["max_stages"] + 1):
        free = np.where(supported, target - (measured + uncertainty), np.nan)
        max_remaining = float(np.nanmax(free)) if np.isfinite(free).any() else 0.0
        if max_remaining < layer:
            stop_reason = "SUPPORTED_DEPTH_BELOW_ONE_LAYER"
            break
        # A cell is printed as a whole: it receives one full layer only if every pixel in it has
        # at least one layer of room below the lower uncertainty envelope.
        room = np.where(in_lattice, np.nan_to_num(free, nan=-np.inf), np.inf)
        cell_min_room = np.full(n_cells, np.inf)
        np.minimum.at(cell_min_room, labels.ravel(), room.ravel())
        active = [int(c) for c in range(n_cells) if fully[c] and cell_min_room[c] >= layer]
        if not active:
            stop_reason = "NO_SAFE_SUPPORTED_REGION"
            break
        deposit = np.where(np.isin(labels, active) & in_lattice, layer, 0.0)
        planned_top = np.where(field["valid"], measured + deposit, np.nan)
        normals = _cell_normals(measured, labels, active, field["pixel_du"])
        path = _toolpath(active, honeycomb, planned_top, normals, field, params, frame)
        cols = np.clip(np.rint((path["local"][:, 0] - field["axis"][0]) / field["pixel_du"]).astype(int), 0, labels.shape[1] - 1)
        rows = np.clip(np.rint((path["local"][:, 1] - field["axis"][0]) / field["pixel_du"]).astype(int), 0, labels.shape[0] - 1)
        deposit_rc = (rows[path["deposit_flag"]], cols[path["deposit_flag"]])
        boundary_violations = int(np.count_nonzero(~supported[deposit_rc]))
        true_height = np.where(field["valid"], true_height + deposit, true_height)
        deposited_total += deposit
        measured = np.where(field["valid"], true_height + rng.normal(0.0, sigma, true_height.shape), np.nan)
        printed = deposited_total > 0
        overfill = float(np.max(true_height[printed] - target[printed]))
        stages.append({
            "stage": stage,
            "active_cells": len(active),
            "toolpath_points": int(len(path["local"])),
            "deposition_points": int(path["deposit_flag"].sum()),
            "max_supported_remaining_before_du": max_remaining,
            "deposited_volume_du3": float(deposit.sum() * field["pixel_du"] ** 2),
            "boundary_violations": boundary_violations,
            "unsupported_cells": int(sum(not fully[c] for c in active)),
            "max_overfill_vs_target_du": max(overfill, 0.0),
            "_toolpath": path,
            "_deposit": deposit,
            "_surface_after": true_height.copy(),
        })
    stop_reason = stop_reason or "STAGE_LIMIT_REACHED"

    wound = field["wound_mask"] & field["valid"]
    gap = np.where(wound, target - true_height, np.nan)
    initial_gap = np.where(wound, target - field["height"], np.nan)
    unresolved = wound & (gap >= layer)
    return {
        "honeycomb": honeycomb,
        "supported": supported,
        "in_lattice": in_lattice,
        "stages": stages,
        "final_height": true_height,
        "summary": {
            "hex_side_du": float(honeycomb["hex_side_m"]),
            "lattice_cells_inside_supported_region": int(fully.sum()),
            "stage_count": len(stages),
            "stop_reason": stop_reason,
            "boundary_violations": int(sum(s["boundary_violations"] for s in stages)),
            "unsupported_cells": int(sum(s["unsupported_cells"] for s in stages)),
            "max_overfill_vs_target_du": float(max([s["max_overfill_vs_target_du"] for s in stages], default=0.0)),
            "initial_max_gap_du": float(np.nanmax(initial_gap)),
            "initial_median_gap_du": float(np.nanmedian(initial_gap)),
            "final_max_gap_du": float(np.nanmax(np.clip(gap, 0, None))),
            "unresolved_area_fraction_of_wound": float(unresolved.sum() / max(wound.sum(), 1)),
            "lattice_area_fraction_of_wound": float((in_lattice & wound).sum() / max(wound.sum(), 1)),
            "deposited_volume_du3": float(sum(s["deposited_volume_du3"] for s in stages)),
        },
    }


def choose_view(scene: dict, wound: np.ndarray, frame: dict) -> dict:
    """The view looking most squarely at the wound (largest cosine with the skin normal)."""
    def score(view):
        direction = camera_centre(view) - frame["origin"]
        return float(direction @ frame["normal"] / np.linalg.norm(direction))
    return max(scene["views"], key=lambda v: (score(v), v["file_stem"]))


def scaffold_prisms(result: dict, field: dict, params: dict = PATIENT_DEFAULTS) -> list[dict]:
    """Each filled cell as a prism (hexagon or clipped outline) from the original bed to the final top."""
    total = sum(stage["_deposit"] for stage in result["stages"]) if result["stages"] else None
    if total is None:
        return []
    if "honeycomb" not in result:  # generic cells from pipeline.coverage
        prisms = []
        flat_total = total.ravel()
        flat_height = field["height"].ravel()
        for index, cell in enumerate(result["cells"]):
            if not (flat_total[cell["pixels"]] > 0).any():
                continue
            footprint = cell["footprint_px"]
            cols = np.clip(footprint[:, 0], 0, field["height"].shape[1] - 1)
            rows = np.clip(footprint[:, 1], 0, field["height"].shape[0] - 1)
            xy = np.column_stack((np.interp(cols, np.arange(len(field["axis"])), field["axis"]),
                                  np.interp(rows, np.arange(len(field["axis"])), field["axis"])))
            bed = float(np.nanmean(flat_height[cell["pixels"]]))
            prisms.append({"cell": index, "kind": cell["kind"], "outline_xy": xy, "bed": bed,
                           "top": bed + float(np.nanmean(flat_total[cell["pixels"]])),
                           "layers": int(round(np.nanmax(flat_total[cell["pixels"]]) / params["layer_thickness_du"]))})
        return prisms
    honeycomb = result["honeycomb"]
    labels = honeycomb["labels"]
    side_px = honeycomb["hex_side_m"] / field["pixel_du"]
    prisms = []
    for cell in np.unique(labels[total > 0]):
        centre = honeycomb["centres_px"][cell]
        outline = hexagon_perimeter(centre, side_px, 1)[:6]
        pixels = labels == cell
        cols = np.clip(np.rint(outline[:, 0]).astype(int), 0, labels.shape[1] - 1)
        rows = np.clip(np.rint(outline[:, 1]).astype(int), 0, labels.shape[0] - 1)
        xy = np.column_stack((field["axis"][cols], field["axis"][rows]))
        bed = float(np.nanmean(field["height"][pixels]))
        layers = float(np.nanmean(total[pixels]))
        prisms.append({"cell": int(cell), "outline_xy": xy, "bed": bed, "top": bed + layers,
                       "layers": int(round(np.nanmax(total[pixels]) / params["layer_thickness_du"]))})
    return prisms


def run_patient(scene_dir: Path = DEFAULT_SCENE, params: dict = PATIENT_DEFAULTS) -> dict:
    scene = load_scene(scene_dir)
    labels = label_wound_vertices(scene, params)
    toward = np.mean([camera_centre(v) for v in scene["views"]], axis=0)
    frame = local_frame(scene["vertices"], labels["wound"], toward, params["frame_radius_factor"])
    local = to_local(scene["vertices"], frame)
    field = height_field(local, labels["wound"], frame, params)
    target = fit_quadratic_target(field)
    fill_v1 = staged_fill(field, target["target"], frame, params)
    # The v1 whole-hexagon lattice covered only a central strip; the coverage module audits it and
    # searches revised planners under the same safety gates. ``fill`` is the selected planner.
    coverage = plan_patient_coverage(field, target["target"], frame, params, to_world, fill_v1["honeycomb"])
    fill = coverage["results"][coverage["selected"]]
    view = choose_view(scene, labels["wound"], frame)
    return {"scene": scene, "labels": labels, "frame": frame, "local": local, "field": field,
            "target": target, "fill": fill, "fill_v1": fill_v1, "coverage": coverage, "view": view,
            "params": dict(params)}
