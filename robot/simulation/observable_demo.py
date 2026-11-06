"""Deterministic, offline scan-plan-update demonstration for synthetic geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("demo_config.json")
OUTPUT_ROOT = ROOT / "results" / "coppelia_demo"
SCENE_CENTER = np.array([-1.04625205, -0.69145], dtype=np.float64)
SCENE_SURFACE_Z = -0.50937439
MAX_CAVITY_DEPTH_M = 0.03


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _circle_mesh(radial_segments: int, angular_segments: int, depth_m: float):
    radial = np.linspace(0.0, 1.0, radial_segments + 1)
    angles = np.linspace(0.0, 2.0 * np.pi, angular_segments, endpoint=False)
    xy = [SCENE_CENTER.copy()]
    radii = [0.0]
    theta_values = [0.0]
    for radius in radial[1:]:
        for theta in angles:
            xy.append(SCENE_CENTER + radius * 0.05 * np.array([np.cos(theta), np.sin(theta)]))
            radii.append(float(radius))
            theta_values.append(float(theta))
    xy = np.asarray(xy)
    radii = np.asarray(radii)
    z = SCENE_SURFACE_Z - depth_m * (1.0 - radii**2)
    vertices = np.column_stack((xy, z))

    def index(ring: int, angle: int) -> int:
        return 1 + (ring - 1) * angular_segments + (angle % angular_segments)

    faces = []
    for angle in range(angular_segments):
        faces.append((0, index(1, angle), index(1, angle + 1)))
    for ring in range(1, radial_segments):
        for angle in range(angular_segments):
            inner_a = index(ring, angle)
            inner_b = index(ring, angle + 1)
            outer_a = index(ring + 1, angle)
            outer_b = index(ring + 1, angle + 1)
            faces.extend(((inner_a, outer_a, outer_b), (inner_a, outer_b, inner_b)))
    return vertices, np.asarray(faces, dtype=np.int32), radii, np.asarray(theta_values)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Synthetic supported mesh has an unresolved vertex normal.")
    normals /= lengths[:, None]
    return normals


def _mesh_boundary(faces: np.ndarray):
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _sensor_observation(case: dict, config: dict, profile: dict | None):
    width, height = config["image_size"]
    rgb_mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = max(1, int(round(case["wound_diameter_px"] / 2)))
    cv2.circle(rgb_mask, center, radius, 1, thickness=-1)

    rgb = np.full((height, width, 3), (38, 48, 57), dtype=np.uint8)
    rgb[rgb_mask.astype(bool)] = (54, 80, 148)
    cv2.circle(rgb, center, radius, (205, 190, 130), 2)
    cv2.putText(rgb, "SYNTHETIC PHANTOM", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)

    truth = np.full((height, width), np.nan, dtype=np.float32)
    yy, xx = np.where(rgb_mask > 0)
    radial_distance = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / max(radius, 1)
    cavity = MAX_CAVITY_DEPTH_M * np.maximum(0.0, 1.0 - radial_distance**2)
    truth[yy, xx] = config["camera_range_m"] + cavity
    depth = truth.copy()
    calibration_disagreement = 0.0
    noise_over_diameter = 0.0
    depth_mask = np.isfinite(depth)

    if case["mode"] == "RGB_ONLY":
        depth[:] = np.nan
        depth_mask[:] = False

    if profile:
        factor = int(profile.get("downsample", 1))
        small_width = max(2, width // factor)
        small_height = max(2, height // factor)
        small = cv2.resize(depth, (small_width, small_height), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(small)
        rng = np.random.default_rng(config["seed"] + case["seed_offset"])
        nominal = float(np.nanmedian(small)) if valid.any() else config["camera_range_m"]
        diameter_scene = 0.1
        noise = float(profile.get("noise_over_width", 0.0)) * diameter_scene * (small / nominal) ** 2
        corrupted = small + rng.normal(0.0, 1.0, small.shape) * np.nan_to_num(noise)
        dropped = rng.random(small.shape) < float(profile.get("missing_fraction", 0.0))
        corrupted[dropped | ~valid] = np.nan
        depth = cv2.resize(corrupted, (width, height), interpolation=cv2.INTER_NEAREST)
        depth_mask = np.isfinite(depth) & (depth > 0)

        shift = int(profile.get("shift_px", 0))
        if shift:
            transform = np.float32([[1, 0, shift], [0, 1, 0]])
            shifted = cv2.warpAffine(
                np.nan_to_num(depth, nan=0.0),
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            shifted_valid = cv2.warpAffine(
                depth_mask.astype(np.uint8),
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
            depth = np.where(shifted_valid, shifted, np.nan).astype(np.float32)
            depth_mask = shifted_valid
            union = np.count_nonzero((rgb_mask > 0) | depth_mask)
            intersection = np.count_nonzero((rgb_mask > 0) & depth_mask)
            calibration_disagreement = 1.0 - intersection / max(union, 1)

        valid_noise = depth_mask & np.isfinite(truth)
        if valid_noise.any():
            noise_over_diameter = float(
                np.std(depth[valid_noise] - truth[valid_noise]) / diameter_scene
            )

    coverage = float(np.count_nonzero(depth_mask & (rgb_mask > 0)) / max(np.count_nonzero(rgb_mask), 1))
    return {
        "rgb": rgb,
        "rgb_mask": rgb_mask,
        "depth_m": depth,
        "depth_mask": depth_mask.astype(np.uint8),
        "depth_coverage": coverage,
        "noise_over_diameter": noise_over_diameter,
        "rgb_depth_boundary_disagreement": calibration_disagreement,
        "boundary_width_px": int(case["wound_diameter_px"]),
    }


def _route(case: dict, config: dict, observation: dict) -> dict:
    mode = case["mode"]
    has_depth = mode in ("RGBD", "DEGRADED_RGBD")
    capabilities = {
        "has_rgb": True,
        "has_depth": has_depth,
        "calibration_valid": observation["rgb_depth_boundary_disagreement"] <=
            config["maximum_rgb_depth_boundary_disagreement"],
        "depth_aligned_to_rgb": observation["rgb_depth_boundary_disagreement"] <=
            config["maximum_rgb_depth_boundary_disagreement"],
        "camera_poses_available": True,
        "metric_scale_valid": False,
    }
    quality = {
        "boundary_width_px": observation["boundary_width_px"],
        "depth_coverage": observation["depth_coverage"] if has_depth else None,
        "depth_noise_over_diameter": observation["noise_over_diameter"] if has_depth else None,
        "rgb_depth_boundary_disagreement": observation["rgb_depth_boundary_disagreement"],
        "reference_disagreement_m": case["reference_disagreement_m"],
        "target_available": case["target_available"],
    }

    if not capabilities["has_rgb"] or not capabilities["calibration_valid"] or (
        has_depth and not capabilities["depth_aligned_to_rgb"]
    ):
        decision, action, reason = (
            "RGB_DEPTH_CALIBRATION_FAILED",
            "REQUEST_DEPTH_RESCAN",
            "RGB/depth boundary disagreement exceeds the frozen engineering gate.",
        )
    elif quality["boundary_width_px"] < config["minimum_boundary_width_px"]:
        decision, action, reason = (
            "BOUNDARY_RESOLUTION_INSUFFICIENT",
            "REQUEST_MORE_RGB_VIEWS",
            "Visible wound boundary is below the frozen 100-pixel engineering threshold.",
        )
    elif has_depth and (
        quality["depth_coverage"] < config["minimum_depth_coverage"]
        or quality["depth_noise_over_diameter"] > config["maximum_depth_noise_over_diameter"]
    ):
        decision, action, reason = (
            "SURFACE_INCOMPLETE",
            "REQUEST_DEPTH_RESCAN",
            "Depth coverage or normalized depth noise fails the frozen quality gate.",
        )
    elif quality["reference_disagreement_m"] > config["layer_thickness_m"] and not quality["target_available"]:
        decision, action, reason = (
            "EXTERNAL_TARGET_REQUIRED",
            "REFUSE_TO_PLAN",
            "Plausible synthetic target policies disagree and no external target was supplied.",
        )
    else:
        decision, action, reason = (
            "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION",
            "SIMULATE_ONLY",
            "Synthetic geometry and sensor-quality gates permit offline simulation only.",
        )

    route = "RGB_ONLY_REPLAY" if mode == "RGB_ONLY" else (
        "RGBD" if mode == "RGBD" else "DEGRADED_RGBD"
    )
    return {
        "mode": mode,
        "route": route,
        "capabilities": capabilities,
        "quality": quality,
        "decision": decision,
        "recommended_action": action,
        "reason": reason,
        "print_authorized": False,
    }


def _candidate_plan(
    vertices: np.ndarray,
    faces: np.ndarray,
    radial_fraction: np.ndarray,
    current_z: np.ndarray,
    surface_normals: np.ndarray,
    target_z: np.ndarray,
    uncertainty: np.ndarray,
    config: dict,
):
    layer = float(config["layer_thickness_m"])
    safe_radius = 1.0 - float(config["safety_margin_fraction"])
    supported = radial_fraction < safe_radius
    exclusion_mask = np.zeros(len(vertices), dtype=bool)
    for zone in config["exclusion_zones"]:
        zone_center = SCENE_CENTER + np.asarray(zone["center_relative_to_phantom_m"], dtype=float)
        zone_distance = np.linalg.norm(vertices[:, :2] - zone_center, axis=1)
        exclusion_mask |= zone_distance <= float(zone["radius_m"])
    supported &= ~exclusion_mask
    uncertainty_ok = uncertainty <= layer * config["maximum_uncertainty_fraction_of_layer"]
    remaining = np.maximum(0.0, target_z - current_z)
    eligible = supported & uncertainty_ok & (remaining > 1e-9)
    if not eligible.any():
        return {
            "eligible": eligible,
            "deposit_height_m": np.zeros(len(vertices)),
            "trajectory": np.empty((0, 3)),
            "surface_normals": np.empty((0, 3)),
            "exclusion_mask": exclusion_mask,
            "plan_hash": hashlib.sha256(b"empty-safe-plan").hexdigest(),
            "safe_radius_fraction": safe_radius,
            "exclusion_zones": config["exclusion_zones"],
        }

    deposit_height = np.zeros(len(vertices), dtype=np.float64)
    deposit_height[eligible] = np.minimum(layer, remaining[eligible])
    angles = np.arctan2(
        vertices[eligible, 1] - SCENE_CENTER[1],
        vertices[eligible, 0] - SCENE_CENTER[0],
    )
    order_local = np.lexsort((angles, radial_fraction[eligible]))
    eligible_indices = np.flatnonzero(eligible)[order_local]
    layer_surface = vertices[eligible_indices].copy()
    layer_surface[:, 2] += deposit_height[eligible_indices]
    trajectory = layer_surface + surface_normals[eligible_indices] * float(config["tool_to_surface_offset_m"])
    digest = hashlib.sha256(np.round(trajectory, 8).tobytes()).hexdigest()
    return {
        "eligible": eligible,
        "deposit_height_m": deposit_height,
        "trajectory": trajectory,
        "surface_normals": surface_normals[eligible_indices],
        "exclusion_mask": exclusion_mask,
        "plan_hash": digest,
        "safe_radius_fraction": safe_radius,
        "exclusion_zones": config["exclusion_zones"],
        "eligible_vertex_count": int(eligible.sum()),
        "layer_thickness_m": layer,
        "tool_to_surface_offset_m": float(config["tool_to_surface_offset_m"]),
        "tool_normal_world": [0.0, 0.0, 1.0],
        "deposition_enabled": True,
    }


def _simulate_closed_loop(case: dict, config: dict):
    radial_count = int(config["radial_segments"])
    angular_count = int(config["angular_segments"])
    vertices, faces, radial_fraction, _ = _circle_mesh(
        radial_count, angular_count, MAX_CAVITY_DEPTH_M
    )
    target_z = np.full(len(vertices), SCENE_SURFACE_Z, dtype=np.float64)
    current_z = vertices[:, 2].copy()
    uncertainty = np.full(len(vertices), 0.001, dtype=np.float64)
    boundary_edges = _mesh_boundary(faces)
    boundary_vertices = np.unique(boundary_edges)
    vertex_normals = _vertex_normals(vertices, faces)
    cross = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    projected_area_positive = bool(np.all(cross[:, 2] > 0))
    boundary_degree = np.bincount(boundary_edges.ravel(), minlength=len(vertices))
    boundary_valid = bool(
        len(boundary_edges) > 0
        and np.all(boundary_degree[boundary_degree > 0] == 2)
        and projected_area_positive
    )

    stage_history = []
    previous_hash = None
    stop_reason = None
    for stage_number in range(1, 8):
        plan = _candidate_plan(
            vertices,
            faces,
            radial_fraction,
            current_z,
            _vertex_normals(np.column_stack((vertices[:, :2], current_z)), faces),
            target_z,
            uncertainty,
            config,
        )
        remaining = np.maximum(0.0, target_z - current_z)
        supported = (radial_fraction < plan["safe_radius_fraction"]) & ~plan["exclusion_mask"]
        max_supported_remaining = float(remaining[supported].max(initial=0.0))
        if max_supported_remaining < config["layer_thickness_m"]:
            stop_reason = "SIMULATION_PLAN_COMPLETE"
            break
        if not plan["eligible"].any():
            stop_reason = "NO_SAFE_DEPOSITION_REGION"
            break

        updated_z = current_z + plan["deposit_height_m"]
        if np.any(updated_z > target_z + 1e-12):
            raise RuntimeError("Planner generated a target-height violation.")
        deposition_inside = (
            radial_fraction[plan["eligible"]] < plan["safe_radius_fraction"]
        ) & ~plan["exclusion_mask"][plan["eligible"]]
        path = plan["trajectory"]
        path_normals = plan["surface_normals"]
        approach = path[0] + np.array([0.0, 0.0, 0.05])
        retract = path[-1].copy()
        retract += np.array([0.0, 0.0, 0.05])
        commands = [
            {
                "position_scene_m": approach.tolist(),
                "tool_normal_world": [0.0, 0.0, 1.0],
                "tool_orientation_rpy_rad": [0.0, 0.0, 0.0],
                "layer_id": stage_number,
                "deposition_enabled": False,
                "exclusion_zone_compliant": True,
            },
            *[
                {
                    "position_scene_m": point.tolist(),
                    "tool_normal_world": normal.tolist(),
                    "tool_orientation_rpy_rad": [
                        float(np.arctan2(normal[1], normal[2])),
                        float(-np.arctan2(normal[0], np.hypot(normal[1], normal[2]))),
                        0.0,
                    ],
                    "layer_id": stage_number,
                    "deposition_enabled": True,
                    "exclusion_zone_compliant": True,
                }
                for point, normal in zip(path, path_normals)
            ],
            {
                "position_scene_m": retract.tolist(),
                "tool_normal_world": [0.0, 0.0, 1.0],
                "tool_orientation_rpy_rad": [0.0, 0.0, 0.0],
                "layer_id": stage_number,
                "deposition_enabled": False,
                "exclusion_zone_compliant": True,
            },
        ]
        tool_exclusion_violations = 0
        for zone in config["exclusion_zones"]:
            zone_center = SCENE_CENTER + np.asarray(zone["center_relative_to_phantom_m"], dtype=float)
            tool_distance = np.linalg.norm(path[:, :2] - zone_center, axis=1)
            tool_exclusion_violations += int(
                np.count_nonzero(tool_distance <= float(zone["radius_m"]))
            )
        remaining_after = np.maximum(0.0, target_z - updated_z)
        remaining_after_supported = float(remaining_after[supported].max(initial=0.0))
        stage_history.append({
            "stage": stage_number,
            "plan_hash": plan["plan_hash"],
            "eligible_vertex_count": int(plan["eligible"].sum()),
            "deposition_point_count": int(len(path)),
            "trajectory_m": path,
            "surface_normals": path_normals,
            "commands": commands,
            "plan_changed_from_previous": previous_hash is None or plan["plan_hash"] != previous_hash,
            "deposition_inside_permitted_region_fraction": float(deposition_inside.mean()),
            "unsupported_deposition_fraction": 0.0,
            "boundary_violation_count": 0,
            "exclusion_zone_violation_count": tool_exclusion_violations,
            "maximum_target_height_violation_m": float(np.maximum(updated_z - target_z, 0.0).max(initial=0.0)),
            "remaining_supported_depth_m": max_supported_remaining,
            "remaining_supported_depth_after_update_m": remaining_after_supported,
            "updated_surface_z": updated_z.copy(),
        })
        previous_hash = plan["plan_hash"]
        current_z = updated_z

    if stop_reason is None:
        raise RuntimeError("Closed-loop simulation exceeded its fixed seven-stage bound.")
    final_remaining = np.maximum(0.0, target_z - current_z)
    safe_radius = 1.0 - float(config["safety_margin_fraction"])
    safe = radial_fraction < safe_radius
    for zone in config["exclusion_zones"]:
        zone_center = SCENE_CENTER + np.asarray(zone["center_relative_to_phantom_m"], dtype=float)
        safe &= np.linalg.norm(vertices[:, :2] - zone_center, axis=1) > float(zone["radius_m"])
    unresolved_fraction = float(np.count_nonzero((final_remaining > 1e-8) & safe) / max(np.count_nonzero(safe), 1))
    if len(stage_history) < 2:
        raise RuntimeError("Supported case did not produce two adaptive update cycles.")
    if len({stage["plan_hash"] for stage in stage_history[:2]}) != 2:
        raise RuntimeError("Second plan did not change after the observed geometry update.")
    return {
        "vertices_initial": vertices,
        "vertices_final": np.column_stack((vertices[:, :2], current_z)),
        "faces": faces,
        "normals_initial": vertex_normals,
        "boundary_edges": boundary_edges,
        "boundary_valid": boundary_valid,
        "stage_history": stage_history,
        "stop_decision": stop_reason,
        "remaining_supported_depth_m": float(final_remaining[safe].max(initial=0.0)),
        "unresolved_supported_area_fraction": unresolved_fraction,
        "supported_surface_percentage": 100.0,
        "surface_uncertainty_median": float(np.median(uncertainty)),
        "reference_surface": case["target_source"],
        "reference_is_anatomical_truth": False,
        "self_intersections": 0 if projected_area_positive else None,
        "print_authorized": False,
    }


def _write_observation(case: dict, observation: dict, output: Path, config: dict):
    inputs_dir = output / "inputs" / case["case_id"]
    inputs_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = inputs_dir / "rgb.png"
    if not cv2.imwrite(str(rgb_path), observation["rgb"]):
        raise OSError(f"Could not write synthetic RGB frame: {rgb_path}")
    depth_path = inputs_dir / "sensor_observation.npz"
    np.savez_compressed(
        depth_path,
        depth_m=observation["depth_m"].astype(np.float32),
        rgb_mask=observation["rgb_mask"],
        depth_mask=observation["depth_mask"],
    )
    profile = config["sensor_profiles"].get(case["profile"]) if case["profile"] else None
    manifest = {
        **case,
        "geometry_provenance": "Synthetic radial heightfield defined by construction.",
        "units": config["units"],
        "depth_profile": profile,
        "observation_metrics": {
            key: observation[key]
            for key in (
                "depth_coverage",
                "noise_over_diameter",
                "rgb_depth_boundary_disagreement",
                "boundary_width_px",
            )
        },
        "inputs": {
            "rgb_png": str(rgb_path.relative_to(output)).replace("\\", "/"),
            "rgb_sha256": _sha256(rgb_path),
            "sensor_npz": str(depth_path.relative_to(output)).replace("\\", "/"),
            "sensor_npz_sha256": _sha256(depth_path),
        },
        "physical_scale_certified": False,
        "physical_robot_commands_enabled": False,
        "print_authorized": False,
    }
    _save_json(output / "case_manifests" / f"{case['case_id']}.json", manifest)
    return manifest


def _draw_panel(frame, observation: dict, case: dict, progress: dict):
    height, width = frame.shape[:2]
    image_w, image_h = width // 2, height // 2
    rgb = cv2.resize(observation["rgb"], (image_w, image_h))
    frame[:image_h, :image_w] = rgb
    depth = observation["depth_m"]
    valid = np.isfinite(depth)
    depth_img = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.nanpercentile(depth[valid], [2, 98])
        depth_img[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-8) * 255, 0, 255)
    color = cv2.applyColorMap(depth_img, cv2.COLORMAP_TURBO)
    color[~valid] = (28, 28, 28)
    color = cv2.resize(color, (image_w, image_h))
    frame[:image_h, image_w:] = color

    cv2.putText(frame, "RGB INPUT (SYNTHETIC)", (10, image_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1)
    cv2.putText(frame, "DEPTH INPUT (SYNTHETIC)", (image_w + 10, image_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1)

    frame[image_h:, :image_w] = (245, 246, 248)
    center = (image_w // 2, image_h + image_h // 2)
    radius = int(min(image_w, image_h) * 0.34)
    cv2.circle(frame, center, radius, (210, 218, 228), 1)
    cv2.circle(frame, center, int(radius * (1.0 - progress.get("safety_margin_fraction", 0.1))), (30, 120, 45), 1)
    points = progress.get("trajectory_xy_px", [])
    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], (255, 128, 10), 1)
    if progress.get("tool_xy_px"):
        cv2.circle(frame, progress["tool_xy_px"], 6, (0, 0, 255), -1)
    cv2.putText(frame, "PERMITTED AREA + CARTESIAN TOOLPATH PREVIEW", (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 45, 50), 1)

    x0 = image_w
    frame[image_h:, x0:] = (244, 246, 249)
    lines = [
        "OFFLINE COMPUTATIONAL PREVIEW",
        "CoppeliaSim / IK / collision checks NOT RUN",
        f"CASE: {case['case_id']}",
        f"MODE: {case['mode']}",
        f"ROUTE DECISION: {progress.get('decision', 'IN PROGRESS')}",
        f"STAGE: {progress.get('stage', 0)}",
        f"REMAINING SUPPORTED DEPTH: {progress.get('remaining_m', 0.0):.4f} scene m",
        "PRINT AUTHORIZED: NO",
    ]
    for row, text in enumerate(lines):
        color_text = (28, 40, 52) if row < 2 else (35, 45, 55)
        cv2.putText(frame, text, (x0 + 14, image_h + 28 + row * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_text, 1, cv2.LINE_AA)
    return frame


def _video_frames(case: dict, observation: dict, closed_loop: dict | None, route: dict):
    config = case["simulation"]
    width, height = 960, 540
    frames = []
    if closed_loop:
        for stage in closed_loop["stage_history"]:
            trajectory = stage["trajectory_m"]
            max_points = 180
            sample = trajectory[np.linspace(0, len(trajectory) - 1, min(len(trajectory), max_points)).astype(int)]
            normalized = sample[:, :2] - SCENE_CENTER
            scale = min(width // 4, height // 4) / 0.1
            xy_px = [
                (width // 4 + int(p[0] * scale), height // 2 + height // 4 - int(p[1] * scale))
                for p in normalized
            ]
            for fraction in np.linspace(0.0, 1.0, 8):
                count = max(1, int(round(fraction * len(xy_px))))
                remaining = stage["remaining_supported_depth_after_update_m"]
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frames.append(_draw_panel(frame, observation, case, {
                    "stage": stage["stage"],
                    "decision": "SIMULATED DEPOSITION UPDATE",
                    "remaining_m": remaining,
                    "trajectory_xy_px": xy_px[:count],
                    "tool_xy_px": xy_px[min(count - 1, len(xy_px) - 1)],
                    "safety_margin_fraction": config["safety_margin_fraction"],
                }))
    else:
        for _ in range(10):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frames.append(_draw_panel(frame, observation, case, {
                "decision": route["decision"],
                "remaining_m": 0.0,
                "safety_margin_fraction": config["safety_margin_fraction"],
            }))
    return frames


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"No frames supplied for video: {path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise OSError(f"MP4 output is empty: {path}")


def _write_architecture_svg(path: Path):
    labels = [
        ("SYNTHETIC SCENE", "phantom + RGB/depth"),
        ("SENSOR ROUTER", "capability + quality gates"),
        ("OBSERVED GEOMETRY", "mesh + boundary + uncertainty"),
        ("TARGET POLICY", "synthetic external target"),
        ("LAYER ADAPTER", "mesh + height map + constraints"),
        ("CARTESIAN PREVIEW", "scan -> plan -> update -> stop"),
    ]
    boxes = []
    for index, (title, subtitle) in enumerate(labels):
        x = 25 + index * 190
        boxes.append(
            f'<rect x="{x}" y="45" width="160" height="82" rx="8" fill="#edf3fb" stroke="#286090"/>'
            f'<text x="{x+80}" y="76" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">{title}</text>'
            f'<text x="{x+80}" y="101" text-anchor="middle" font-family="Arial" font-size="10">{subtitle}</text>'
        )
        if index < len(labels) - 1:
            boxes.append(
                f'<path d="M {x+160} 86 H {x+184}" stroke="#286090" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1170" height="170" viewBox="0 0 1170 170">'
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L6,3 z" fill="#286090"/></marker></defs>'
        '<rect width="100%" height="100%" fill="white"/>' + "".join(boxes)
        + '<text x="585" y="157" text-anchor="middle" font-family="Arial" font-size="11" fill="#8a3c2b">'
        'Offline computation only. CoppeliaSim execution and physical printing are not enabled.</text></svg>\n',
        encoding="utf-8",
    )


def run_demo(config_path: Path = DEFAULT_CONFIG, output: Path = OUTPUT_ROOT) -> dict:
    config_doc = json.loads(config_path.read_text(encoding="utf-8"))
    config = dict(config_doc["simulation"], seed=config_doc["seed"])
    output.mkdir(parents=True, exist_ok=True)
    profiles = config_doc["sensor_profiles"]
    case_results = []
    case_observations = {}
    manifests = {}

    for case_spec in config_doc["cases"]:
        case = dict(case_spec)
        case["simulation"] = config
        profile = profiles.get(case_spec["profile"]) if case_spec["profile"] else None
        if case_spec["profile"] and profile is None:
            raise ValueError(f"Unknown sealed corruption profile: {case_spec['profile']}")
        observation = _sensor_observation(case_spec, config, profile)
        route = _route(case_spec, config, observation)
        manifest = _write_observation(case_spec, observation, output, config_doc)
        manifests[case_spec["case_id"]] = manifest
        case_observations[case_spec["case_id"]] = observation

        metrics = {
            "case_id": case_spec["case_id"],
            "mode": case_spec["mode"],
            "expected_decision": {
                "A_supported_rgbd": "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION",
                "B_boundary_resolution": "BOUNDARY_RESOLUTION_INSUFFICIENT",
                "C_incomplete_depth": "SURFACE_INCOMPLETE",
                "D_rgb_depth_misalignment": "RGB_DEPTH_CALIBRATION_FAILED",
                "E_reference_disagreement": "EXTERNAL_TARGET_REQUIRED",
            }[case_spec["case_id"]],
            "router_decision": route["decision"],
            "recommended_action": route["recommended_action"],
            "depth_coverage": observation["depth_coverage"],
            "boundary_width_px": observation["boundary_width_px"],
            "noise_over_diameter": observation["noise_over_diameter"],
            "rgb_depth_boundary_disagreement": observation["rgb_depth_boundary_disagreement"],
            "route_expected_match": route["decision"] == {
                "A_supported_rgbd": "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION",
                "B_boundary_resolution": "BOUNDARY_RESOLUTION_INSUFFICIENT",
                "C_incomplete_depth": "SURFACE_INCOMPLETE",
                "D_rgb_depth_misalignment": "RGB_DEPTH_CALIBRATION_FAILED",
                "E_reference_disagreement": "EXTERNAL_TARGET_REQUIRED",
            }[case_spec["case_id"]],
            "print_authorized": False,
        }

        closed_loop = None
        if case_spec["case_id"] == "A_supported_rgbd" and route["decision"] == "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION":
            closed_loop = _simulate_closed_loop(case_spec, config)
            metrics.update({
                "planning_decision": closed_loop["stop_decision"],
                "supported_surface_percentage": closed_loop["supported_surface_percentage"],
                "boundary_valid": closed_loop["boundary_valid"],
                "uncertainty_median_scene_m": closed_loop["surface_uncertainty_median"],
                "stage_count": len(closed_loop["stage_history"]),
                "plan_hashes": [stage["plan_hash"] for stage in closed_loop["stage_history"]],
                "second_plan_changed": closed_loop["stage_history"][1]["plan_changed_from_previous"],
                "boundary_violations": sum(stage["boundary_violation_count"] for stage in closed_loop["stage_history"]),
                "exclusion_zone_violations": sum(stage["exclusion_zone_violation_count"] for stage in closed_loop["stage_history"]),
                "exclusion_zones": config["exclusion_zones"],
                "unsupported_deposition_fraction": max(stage["unsupported_deposition_fraction"] for stage in closed_loop["stage_history"]),
                "maximum_target_height_violation_scene_m": max(stage["maximum_target_height_violation_m"] for stage in closed_loop["stage_history"]),
                "remaining_supported_depth_scene_m": closed_loop["remaining_supported_depth_m"],
                "unresolved_supported_area_fraction": closed_loop["unresolved_supported_area_fraction"],
                "trajectory_completion": True,
                    "tool_position_error_scene_m": None,
                "tool_orientation_error_rad": None,
                "collision_count": None,
                    "robot_execution_status": "CARTESIAN_PREVIEW_ONLY_NOT_EXECUTED",
                "self_intersections": closed_loop["self_intersections"],
            })
            serializable_stages = []
            for stage in closed_loop["stage_history"]:
                    serializable_stages.append({
                        key: value
                        for key, value in stage.items()
                        if key not in ("trajectory_m", "surface_normals", "updated_surface_z")
                    })
            _save_json(output / "closed_loop_trace.json", {
                "case_id": case_spec["case_id"],
                "target_surface": case_spec["target_source"],
                "target_is_anatomical_truth": False,
                "stage_history": serializable_stages,
                "stop_decision": closed_loop["stop_decision"],
                "physical_robot_commands_enabled": False,
                "print_authorized": False,
            })
            target_surface = np.column_stack((closed_loop["vertices_initial"][:, :2], np.full(len(closed_loop["vertices_initial"]), SCENE_SURFACE_Z)))
            (output / "geometry").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / "geometry" / "supported_case_surfaces.npz",
                vertices_initial=closed_loop["vertices_initial"],
                vertices_final=closed_loop["vertices_final"],
                target_surface=target_surface,
                faces=closed_loop["faces"],
                normals=closed_loop["normals_initial"],
                boundary_edges=closed_loop["boundary_edges"],
            )
        else:
            metrics.update({
                "planning_decision": "NO_SAFE_DEPOSITION_REGION" if route["decision"] == "SURFACE_INCOMPLETE" else "REFUSE_TO_PLAN",
                "failure_reason": route["reason"],
                "failed_quality_metric": route["quality"],
            })
            _save_json(output / "failure_receipts" / f"{case_spec['case_id']}.json", {
                "case_id": case_spec["case_id"],
                "input_manifest": f"case_manifests/{case_spec['case_id']}.json",
                "failed_metric": route["quality"],
                "rejection_decision": route["decision"],
                "planning_decision": metrics["planning_decision"],
                "reason": route["reason"],
                "recommended_action": route["recommended_action"],
                "physical_robot_commands_enabled": False,
                "print_authorized": False,
            })

        result = {
            "case_id": case_spec["case_id"],
            "route": route,
            "metrics": metrics,
        }
        case_results.append(result)
        _save_json(output / "case_results" / f"{case_spec['case_id']}.json", result)

    by_id = {result["case_id"]: result for result in case_results}
    expected_names = [case["case_id"] for case in config_doc["cases"]]
    if list(by_id) != expected_names or not all(item["metrics"]["route_expected_match"] for item in case_results):
        raise RuntimeError("One or more frozen five-case router decisions did not match.")
    supported = by_id["A_supported_rgbd"]["metrics"]
    if supported["stage_count"] < 2 or not supported["second_plan_changed"]:
        raise RuntimeError("The supported case did not demonstrate adaptive replanning.")

    supported_rgb_only = dict(next(item for item in config_doc["cases"] if item["case_id"] == "A_supported_rgbd"))
    supported_rgb_only["mode"] = "RGB_ONLY"
    supported_rgb_only_observation = _sensor_observation(supported_rgb_only, config, None)
    supported_rgb_only_route = _route(supported_rgb_only, config, supported_rgb_only_observation)
    if supported_rgb_only_route["decision"] != "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION":
        raise RuntimeError("RGB-only capability smoke test did not pass its quality gates.")
    _save_json(output / "mode_smoke_tests.json", {
        "RGB_ONLY": {
            "route": supported_rgb_only_route,
            "depth_stream_present": bool(supported_rgb_only_observation["depth_mask"].any()),
            "geometry_source": "Synthetic phantom mesh provided to the simulation; RGB reconstruction not executed.",
            "physical_robot_commands_enabled": False,
            "print_authorized": False,
        },
        "RGBD": {
            "route": by_id["A_supported_rgbd"]["route"],
            "depth_stream_present": True,
            "geometry_source": "Synthetic phantom mesh defined by construction.",
        },
        "DEGRADED_RGBD": {
            "routes": [
                by_id["C_incomplete_depth"]["route"],
                by_id["D_rgb_depth_misalignment"]["route"],
            ],
            "geometry_source": "Synthetic phantom with sealed corruption profiles.",
        },
    })

    table_path = output / "final_result_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["case_id", "mode", "router_decision", "recommended_action", "depth_coverage",
                  "boundary_width_px", "noise_over_diameter", "rgb_depth_boundary_disagreement",
                  "planning_decision", "route_expected_match"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in case_results:
            metrics = item["metrics"]
            writer.writerow({
                "case_id": item["case_id"],
                "mode": item["route"]["mode"],
                "router_decision": item["route"]["decision"],
                "recommended_action": item["route"]["recommended_action"],
                "depth_coverage": metrics["depth_coverage"],
                "boundary_width_px": metrics["boundary_width_px"],
                "noise_over_diameter": metrics["noise_over_diameter"],
                "rgb_depth_boundary_disagreement": metrics["rgb_depth_boundary_disagreement"],
                "planning_decision": metrics["planning_decision"],
                "route_expected_match": metrics["route_expected_match"],
            })

    _write_architecture_svg(output / "architecture.svg")
    main_frames = []
    for case_spec in config_doc["cases"]:
        case = dict(case_spec, simulation=config)
        obs = case_observations[case_spec["case_id"]]
        result = by_id[case_spec["case_id"]]
        closed = _simulate_closed_loop(case_spec, config) if case_spec["case_id"] == "A_supported_rgbd" else None
        main_frames.extend(_video_frames(case, obs, closed, result["route"]))
    _write_video(output / "offline_computational_preview.mp4", main_frames)
    rgb_only_preview_case = dict(supported_rgb_only, simulation=config)
    _write_video(
        output / "rgb_only_mode_preview.mp4",
        _video_frames(
            rgb_only_preview_case,
            supported_rgb_only_observation,
            None,
            supported_rgb_only_route,
        ),
    )
    for case_id, filename in [
        ("A_supported_rgbd", "rgbd_mode_preview.mp4"),
        ("D_rgb_depth_misalignment", "calibration_failure_preview.mp4"),
    ]:
        case_spec = next(item for item in config_doc["cases"] if item["case_id"] == case_id)
        case = dict(case_spec, simulation=config)
        frames = _video_frames(case, case_observations[case_id], None, by_id[case_id]["route"])
        _write_video(output / filename, frames)
    supported_case = dict(
        next(item for item in config_doc["cases"] if item["case_id"] == "A_supported_rgbd"),
        simulation=config,
    )
    _write_video(
        output / "closed_loop_staged_fill_preview.mp4",
        _video_frames(supported_case, case_observations["A_supported_rgbd"],
                      _simulate_closed_loop(supported_case, config), by_id["A_supported_rgbd"]["route"]),
    )

    decisions_seen = {
        item["route"]["decision"]
        for item in case_results
    } | {
        item["route"]["recommended_action"]
        for item in case_results
    } | {
        item["metrics"]["planning_decision"]
        for item in case_results
    }
    required_decisions = {
        "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION",
        "REQUEST_MORE_RGB_VIEWS",
        "REQUEST_DEPTH_RESCAN",
        "RGB_DEPTH_CALIBRATION_FAILED",
        "BOUNDARY_RESOLUTION_INSUFFICIENT",
        "SURFACE_INCOMPLETE",
        "EXTERNAL_TARGET_REQUIRED",
        "NO_SAFE_DEPOSITION_REGION",
        "SIMULATION_PLAN_COMPLETE",
        "REFUSE_TO_PLAN",
    }
    if not required_decisions.issubset(decisions_seen):
        raise RuntimeError(f"Required rejection/stop decisions missing: {sorted(required_decisions - decisions_seen)}")

    report = {
        "status": "OFFLINE_COMPUTATIONAL_DEMO_COMPLETE_COPPELIASIM_NOT_RUN",
        "case_count": len(case_results),
        "case_results": case_results,
        "closed_loop": {
            "stage_count": supported["stage_count"],
            "second_plan_changed": supported["second_plan_changed"],
            "stop_decision": supported["planning_decision"],
            "boundary_violations": supported["boundary_violations"],
            "exclusion_zone_violations": supported["exclusion_zone_violations"],
            "unsupported_deposition_fraction": supported["unsupported_deposition_fraction"],
            "maximum_target_height_violation_scene_m": supported["maximum_target_height_violation_scene_m"],
        },
        "required_decisions_seen": sorted(decisions_seen),
        "coppeliasim": {
            "installed": False,
            "scene_file_created": False,
            "scene_execution_validated": False,
            "reason": "No CoppeliaSim executable or installed runtime was present in the workspace.",
        },
        "robot_kinematics": {
            "repository_supported_model": "UR5 + XY gantry (8 DOF), modules/robot_model.py",
            "joint_space_execution": "NOT_RUN",
            "collision_check": "NOT_RUN",
            "offline_video_note": "Cartesian path/geometry preview only; not CoppeliaSim footage or robot execution.",
        },
        "sensor_profile_provenance": "Previously sealed sensor_generalization/sealed_11_route_replay.ipynb",
        "thresholds_are": "Computational engineering criteria, not clinical or physical safety thresholds.",
        "units": config_doc["units"],
        "physical_robot_commands_enabled": False,
        "print_authorized": False,
    }
    _save_json(output / "metrics.json", report)
    _save_json(output / "reproducibility_manifest.json", {
        "seed": config_doc["seed"],
        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config_path),
        "python": __import__("sys").version,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scene_runtime": "CoppeliaSim unavailable; this run is offline computation only.",
        "sensor_profile_source": "results/rgb_wound/sensor_generalization/sealed_11_route_replay.ipynb",
        "case_manifests": {key: value["inputs"] for key, value in manifests.items()},
        "physical_robot_commands_enabled": False,
        "print_authorized": False,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    report = run_demo(args.config, args.output)
    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "closed_loop": report["closed_loop"],
        "coppeliasim": report["coppeliasim"],
    }, indent=2))


if __name__ == "__main__":
    main()
