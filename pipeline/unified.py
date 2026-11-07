"""One offline pipeline: RGB-D observation to staged honeycomb toolpath preview.

The stages are run in a fixed order, and each one can stop the run:

1. OBSERVE: synthetic RGB(-D) capture from the frozen five-case configuration
   (``robot/simulation/demo_config.json``).
2. IDENTIFY: segment the wound in the RGB image to get a pixel mask.
3. SURFACE: turn the depth image inside that mask into a wound-bed height field.
4. ROUTE: apply the frozen quality gates (``observable_demo._route``).
5. TARGET: require an operator-supplied final target surface. The sealed
   architecture decision (STAGED_FILL_WITH_OPERATOR_FINAL_TARGET) rules out
   inferring the final surface automatically.
6. HONEYCOMB: tile the visible wound with the hexagonal lattice from
   ``modules.honeycomb`` and keep only the cells that lie fully inside the
   supported region.
7. STAGED FILL: deposit at most one layer per stage below the lower uncertainty
   envelope, re-measure the surface, re-plan, and stop using the observable
   rule.

This is a computational preview only. Units are synthetic scene metres at a
declared renderer scale, and metric scale is not independently recovered.
Robot commands and printing stay disabled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modules.conformal_mapping import compute_nozzle_orientations
from modules.honeycomb import create_hex_grid, hexagon_perimeter
from robot.simulation.observable_demo import (
    DEFAULT_CONFIG,
    _route,
    _sensor_observation,
)

ACCEPTED = "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION"
SCENE_WOUND_DIAMETER_M = 0.1  # renderer scale used by observable_demo
PIPELINE_DEFAULTS = {
    "honeycomb_cells_across": 12,
    "surface_uncertainty_base_m": 0.001,
    "travel_rise_m": 0.02,
    "perimeter_shrink": 0.9,
    "perimeter_points_per_edge": 4,
    "max_stages": 10,
}
STAGE_NAMES = ("OBSERVE", "IDENTIFY", "SURFACE", "ROUTE", "TARGET", "HONEYCOMB", "STAGED_FILL")


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    doc = json.loads(Path(config_path).read_text(encoding="utf-8"))
    doc["pipeline"] = {**PIPELINE_DEFAULTS, **doc.get("pipeline", {})}
    return doc


def identify_wound(rgb: np.ndarray) -> dict:
    """Colour-threshold segmenter for the synthetic phantom image.

    This stands in for the wound masks used by the reconstruction notebook. It
    keeps the largest bright-red connected region and fills its holes.
    """
    candidate = (rgb[:, :, 2].astype(np.int16) > 100).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count <= 1:
        return {"mask": np.zeros(rgb.shape[:2], dtype=bool), "found": False}
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    ys, xs = np.nonzero(filled)
    width_px = int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    return {
        "mask": filled.astype(bool),
        "found": True,
        "centroid_px": [float(xs.mean()), float(ys.mean())],
        "boundary_width_px": width_px,
        "area_px": int(filled.sum()),
    }


def surface_from_depth(depth_m: np.ndarray, mask: np.ndarray, camera_range_m: float) -> dict:
    """Convert depth along the camera axis into height relative to the skin plane.

    Positive height is toward the camera, so a cavity has negative height.
    Pixels with no valid depth stay NaN and are treated as unsupported.
    """
    valid = mask & np.isfinite(depth_m) & (depth_m > 0)
    height = np.full(depth_m.shape, np.nan, dtype=np.float64)
    height[valid] = camera_range_m - depth_m[valid]
    return {
        "height_m": height,
        "valid": valid,
        "coverage": float(valid.sum() / max(mask.sum(), 1)),
    }


def _cell_normals(height: np.ndarray, labels: np.ndarray, cells: list[int], pixel_m: float) -> dict[int, np.ndarray]:
    """One nozzle normal per cell, from a least-squares plane over the cell's pixels.

    A per-cell fit stays stable at the height steps between neighbouring cells
    and the unfilled rim, where pixel gradients would tilt the tool sideways.
    """
    normals = {}
    for cell in cells:
        rows, cols = np.nonzero(labels == cell)
        z = height[rows, cols]
        keep = np.isfinite(z)
        design = np.column_stack((cols[keep] * pixel_m, rows[keep] * pixel_m, np.ones(keep.sum())))
        (a, b, _), *_ = np.linalg.lstsq(design, z[keep], rcond=None)
        n = np.array([-a, -b, 1.0])
        normals[cell] = n / np.linalg.norm(n)
    return normals


def build_honeycomb(mask: np.ndarray, centroid_px: list[float], pixel_m: float, cells_across: int) -> dict:
    """Lay the ``modules.honeycomb`` lattice over the identified wound.

    Every pixel is assigned to the nearest hexagon centre. For this lattice,
    that nearest-centre partition is exactly the flat-top hexagonal tiling.
    """
    ys, xs = np.nonzero(mask)
    width_m = (max(xs.max() - xs.min(), ys.max() - ys.min()) + 1) * pixel_m
    hex_side = width_m / cells_across
    nx = int(np.ceil(width_m / (1.5 * hex_side))) + 3
    ny = int(np.ceil(width_m / (np.sqrt(3) * hex_side))) + 3
    grid = create_hex_grid(nx, ny, hex_side).reshape(-1, 2)
    grid -= grid.mean(axis=0)
    centres_px = grid / pixel_m + np.asarray(centroid_px)

    height, width = mask.shape
    yy, xx = np.mgrid[0:height, 0:width]
    pixels = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    labels = np.full(height * width, -1, dtype=np.int32)
    best = np.full(height * width, np.inf)
    for index, centre in enumerate(centres_px):
        distance = np.sum((pixels - centre) ** 2, axis=1)
        closer = distance < best
        best[closer] = distance[closer]
        labels[closer] = index
    return {
        "hex_side_m": float(hex_side),
        "centres_m": grid,
        "centres_px": centres_px,
        "labels": labels.reshape(height, width),
    }


def _greedy_order(points: np.ndarray) -> list[int]:
    if len(points) == 0:
        return []
    remaining = list(range(len(points)))
    start = min(remaining, key=lambda i: (points[i, 1], points[i, 0]))
    order = [start]
    remaining.remove(start)
    while remaining:
        last = points[order[-1]]
        nxt = min(remaining, key=lambda i: (np.sum((points[i] - last) ** 2), i))
        order.append(nxt)
        remaining.remove(nxt)
    return order


def _supported_region(identified: dict, surface: dict, observation: dict, config: dict, pixel_m: float) -> np.ndarray:
    sim = config["simulation"]
    mask = identified["mask"]
    distance_px = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    margin_px = sim["safety_margin_fraction"] * identified["boundary_width_px"] / 2.0
    supported = surface["valid"] & (distance_px >= margin_px)
    height, width = mask.shape
    yy, xx = np.mgrid[0:height, 0:width]
    centre = np.asarray(identified["centroid_px"])
    for zone in sim["exclusion_zones"]:
        offset_px = np.asarray(zone["center_relative_to_phantom_m"], dtype=float) / pixel_m
        radius_px = float(zone["radius_m"]) / pixel_m
        supported &= np.hypot(xx - centre[0] - offset_px[0], yy - centre[1] - offset_px[1]) > radius_px
    return supported


def _stage_toolpath(cells: list[int], honeycomb: dict, planned_top: np.ndarray, normals: dict[int, np.ndarray],
                    pixel_m: float, centroid_px: list[float], config: dict, stage: int) -> dict:
    pipe, sim = config["pipeline"], config["simulation"]
    side_px = honeycomb["hex_side_m"] / pixel_m * pipe["perimeter_shrink"]
    order = _greedy_order(honeycomb["centres_px"][cells]) if cells else []
    rows = []
    for rank in order:
        cell = cells[rank]
        centre = honeycomb["centres_px"][cell]
        outline = hexagon_perimeter(centre, side_px, pipe["perimeter_points_per_edge"])
        points_px = np.vstack((outline, outline[:1], centre[None, :]))
        cols = np.clip(np.rint(points_px[:, 0]).astype(int), 0, planned_top.shape[1] - 1)
        rws = np.clip(np.rint(points_px[:, 1]).astype(int), 0, planned_top.shape[0] - 1)
        z = planned_top[rws, cols]
        n = np.tile(normals[cell], (len(points_px), 1))
        xy = (points_px - np.asarray(centroid_px)) * pixel_m
        surface_pts = np.column_stack((xy, z))
        tool = surface_pts + n * sim["tool_to_surface_offset_m"]
        rise = np.array([0.0, 0.0, pipe["travel_rise_m"]])
        rows.append((tool[0] + rise, n[0], cell, 0, rws[0], cols[0]))
        for k in range(len(tool)):
            rows.append((tool[k], n[k], cell, 1, rws[k], cols[k]))
        rows.append((tool[-1] + rise, n[-1], cell, 0, rws[-1], cols[-1]))
    if not rows:
        return {"points_m": np.empty((0, 3)), "normals": np.empty((0, 3)), "cell": np.empty(0, int),
                "deposit_flag": np.empty(0, bool), "pixel_rc": np.empty((0, 2), int), "rotations": np.empty((3, 3, 0)),
                "stage": stage, "plan_hash": hashlib.sha256(b"empty").hexdigest()}
    points = np.array([r[0] for r in rows])
    normal_arr = np.array([r[1] for r in rows])
    return {
        "points_m": points,
        "normals": normal_arr,
        "cell": np.array([r[2] for r in rows]),
        "deposit_flag": np.array([bool(r[3]) for r in rows]),
        "pixel_rc": np.array([(r[4], r[5]) for r in rows]),
        "rotations": compute_nozzle_orientations(normal_arr.T),
        "stage": stage,
        "plan_hash": hashlib.sha256(np.round(points, 9).tobytes()).hexdigest(),
    }


def staged_honeycomb_fill(identified: dict, surface: dict, observation: dict, target_height_m: float,
                          config: dict, pixel_m: float, seed: int) -> dict:
    """Plan and simulate staged fill using only re-measured geometry.

    The hidden synthetic truth (``true_height``) is used only for the
    evaluation metrics at the end of each stage and never for planning.
    """
    sim, pipe = config["simulation"], config["pipeline"]
    layer = float(sim["layer_thickness_m"])
    rng = np.random.default_rng(seed)
    noise_sigma = observation["noise_over_diameter"] * SCENE_WOUND_DIAMETER_M
    uncertainty = np.where(surface["valid"], pipe["surface_uncertainty_base_m"] + noise_sigma, np.inf)
    supported = _supported_region(identified, surface, observation, config, pixel_m)
    supported &= uncertainty <= sim["maximum_uncertainty_fraction_of_layer"] * layer
    honeycomb = build_honeycomb(identified["mask"], identified["centroid_px"], pixel_m,
                                pipe["honeycomb_cells_across"])
    labels = honeycomb["labels"]
    n_cells = len(honeycomb["centres_px"])
    cell_fully_supported = np.ones(n_cells, dtype=bool)
    unsupported_labels = np.unique(labels[~supported])
    cell_fully_supported[unsupported_labels] = False
    cell_has_pixels = np.bincount(labels.ravel(), minlength=n_cells) > 0
    cell_fully_supported &= cell_has_pixels
    in_lattice = cell_fully_supported[labels]

    true_height = surface["height_m"].copy()
    measured = surface["height_m"].copy()
    stages, stop_reason = [], None
    for stage in range(1, pipe["max_stages"] + 1):
        lower_free = target_height_m - (measured + uncertainty)
        lower_free = np.where(supported, lower_free, np.nan)
        max_remaining = float(np.nanmax(lower_free)) if np.isfinite(lower_free).any() else 0.0
        if max_remaining < layer:
            stop_reason = "SUPPORTED_DEPTH_BELOW_ONE_LAYER"
            break
        deposit = np.where(in_lattice, np.clip(np.minimum(layer, np.nan_to_num(lower_free, nan=0.0)), 0.0, None), 0.0)
        active = np.unique(labels[deposit > 0])
        active = [int(c) for c in active if cell_fully_supported[c]]
        if not active:
            stop_reason = "NO_SAFE_SUPPORTED_REGION"
            break
        planned_top = np.where(surface["valid"], measured + deposit, np.nan)
        normals = _cell_normals(measured, labels, active, pixel_m)
        path = _stage_toolpath(active, honeycomb, planned_top, normals, pixel_m,
                               identified["centroid_px"], config, stage)

        rc = path["pixel_rc"][path["deposit_flag"]]
        boundary_violations = int(np.count_nonzero(~supported[rc[:, 0], rc[:, 1]]))
        unsupported_cells = int(sum(not cell_fully_supported[c] for c in active))
        true_height = np.where(surface["valid"], true_height + deposit, true_height)
        measured = np.where(surface["valid"], true_height + rng.normal(0.0, 1.0, true_height.shape) * noise_sigma, np.nan)
        overfill = float(np.nanmax(np.where(surface["valid"], true_height - target_height_m, np.nan)))
        stages.append({
            "stage": stage,
            "active_cells": len(active),
            "toolpath_points": int(len(path["points_m"])),
            "deposition_points": int(path["deposit_flag"].sum()),
            "plan_hash": path["plan_hash"],
            "max_supported_remaining_before_m": max_remaining,
            "deposited_volume_m3": float(deposit.sum() * pixel_m**2),
            "boundary_violations": boundary_violations,
            "unsupported_cells": unsupported_cells,
            "max_overfill_vs_target_m_evaluation_only": max(overfill, 0.0),
            "_toolpath": path,
            "_surface_after": true_height.copy(),
        })
    if stop_reason is None:
        stop_reason = "STAGE_LIMIT_REACHED"

    wound = identified["mask"] & surface["valid"]
    final_gap = np.where(wound, target_height_m - true_height, np.nan)
    unresolved = wound & (final_gap >= layer)
    return {
        "honeycomb": honeycomb,
        "supported": supported,
        "in_lattice": in_lattice,
        "stages": stages,
        "stop_reason": stop_reason,
        "initial_height_m": surface["height_m"],
        "final_height_m": true_height,
        "summary": {
            "hex_side_m": honeycomb["hex_side_m"],
            "lattice_cells_inside_supported_region": int(cell_fully_supported.sum()),
            "stage_count": len(stages),
            "stop_reason": stop_reason,
            "boundary_violations": int(sum(s["boundary_violations"] for s in stages)),
            "unsupported_cells": int(sum(s["unsupported_cells"] for s in stages)),
            "max_overfill_m_evaluation_only": float(max([s["max_overfill_vs_target_m_evaluation_only"] for s in stages], default=0.0)),
            "max_remaining_underfill_m_evaluation_only": float(np.nanmax(np.clip(final_gap, 0, None))) if wound.any() else 0.0,
            "unresolved_area_fraction_of_wound": float(unresolved.sum() / max(wound.sum(), 1)),
            "lattice_area_fraction_of_wound": float((in_lattice & wound).sum() / max(wound.sum(), 1)),
        },
    }


def run_case(case_spec: dict, config_doc: dict | None = None, operator_target_height_m: float | None = None) -> dict:
    """Run one case through all stages; stop at the first stage that refuses."""
    config_doc = config_doc or load_config()
    sim = dict(config_doc["simulation"], seed=config_doc["seed"])
    config = {"simulation": sim, "pipeline": config_doc["pipeline"]}
    profile_name = case_spec.get("profile")
    profile = config_doc["sensor_profiles"].get(profile_name) if profile_name else None
    pixel_m = SCENE_WOUND_DIAMETER_M / case_spec["wound_diameter_px"]
    trace: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "case_id": case_spec["case_id"],
        "units": "synthetic scene metres at a declared renderer scale; metric scale not independently recovered",
        "pixel_size_m_declared": pixel_m,
        "print_authorized": False,
        "physical_robot_commands_enabled": False,
        "stage_trace": trace,
    }

    def stop(stage: str, decision: str, reason: str) -> dict:
        trace.append({"stage": stage, "status": "STOP", "decision": decision, "reason": reason})
        result.update({"final_stage": stage, "final_decision": decision, "plan": None})
        return result

    observation = _sensor_observation(case_spec, sim, profile)
    result["_observation"] = observation
    trace.append({"stage": "OBSERVE", "status": "PASS", "mode": case_spec["mode"], "profile": profile_name})

    identified = identify_wound(observation["rgb"])
    result["_identified"] = identified
    if not identified["found"]:
        return stop("IDENTIFY", "WOUND_NOT_FOUND", "No wound-coloured region in the RGB image.")
    truth = observation["rgb_mask"] > 0
    iou = float((identified["mask"] & truth).sum() / max((identified["mask"] | truth).sum(), 1))
    trace.append({"stage": "IDENTIFY", "status": "PASS", "boundary_width_px": identified["boundary_width_px"],
                  "mask_iou_vs_renderer_mask_evaluation_only": iou})

    surface = surface_from_depth(observation["depth_m"], identified["mask"], sim["camera_range_m"])
    result["_surface"] = surface
    trace.append({"stage": "SURFACE", "status": "PASS" if surface["valid"].any() else "NO_DEPTH",
                  "depth_coverage_of_identified_mask": surface["coverage"]})

    routed_observation = dict(observation, boundary_width_px=identified["boundary_width_px"])
    route = _route(case_spec, sim, routed_observation)
    result["route"] = {k: route[k] for k in ("route", "decision", "recommended_action", "reason")}
    if route["decision"] != ACCEPTED:
        return stop("ROUTE", route["decision"], route["reason"])
    trace.append({"stage": "ROUTE", "status": "PASS", "decision": route["decision"]})

    if operator_target_height_m is None and case_spec.get("target_available"):
        operator_target_height_m = 0.0
    if operator_target_height_m is None:
        return stop("TARGET", "OPERATOR_FINAL_TARGET_REQUIRED",
                    "Healthy-skin closure is not identifiable from observations; an operator target is required.")
    trace.append({"stage": "TARGET", "status": "PASS", "target_source": case_spec.get("target_source") or "OPERATOR_ARGUMENT",
                  "target_height_m": operator_target_height_m, "target_is_anatomical_truth": False})

    plan = staged_honeycomb_fill(identified, surface, observation, operator_target_height_m, config,
                                 pixel_m, config_doc["seed"] + case_spec["seed_offset"])
    trace.append({"stage": "HONEYCOMB", "status": "PASS", "hex_side_m": plan["summary"]["hex_side_m"],
                  "cells": plan["summary"]["lattice_cells_inside_supported_region"]})
    trace.append({"stage": "STAGED_FILL", "status": "PASS", **{k: plan["summary"][k] for k in ("stage_count", "stop_reason")}})
    result.update({
        "final_stage": "STAGED_FILL",
        "final_decision": "PLAN_READY_FOR_OPERATOR_REVIEW_PRINT_DISABLED",
        "plan": plan,
    })
    return result


def run_all_cases(config_path: Path = DEFAULT_CONFIG) -> list[dict]:
    config_doc = load_config(config_path)
    return [run_case(case, config_doc) for case in config_doc["cases"]]


def public_summary(result: dict) -> dict:
    """JSON-serialisable summary without image arrays or private keys."""
    summary = {k: v for k, v in result.items() if not k.startswith("_") and k != "plan"}
    plan = result.get("plan")
    if plan:
        summary["plan_summary"] = plan["summary"]
        summary["stages"] = [{k: v for k, v in s.items() if not k.startswith("_")} for s in plan["stages"]]
    return summary
