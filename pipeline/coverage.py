"""Coverage audit and revised honeycomb planner for the patient-level case.

The first patient planner (``pipeline.patient.staged_fill``) kept only lattice
cells whose whole hexagon lay inside the supported region. On the Syn3DWound
leg that left eight cells in one central row and most of the eligible wound
uncovered. This module reopens only that planning component:

1. ``eligibility`` classifies every height-field pixel with the *unchanged*
   safety rules (wound boundary margin, surface support, measurement
   uncertainty gate, at least one layer of room below the lower envelope).
2. ``audit_lattice`` explains why each cell of an existing lattice was
   rejected.
3. Three planners choose a fixed set of cells inside the eligible region:
   ``WHOLE_CELL`` (one regular lattice; side, rotation and staggered offset are
   searched), ``ADAPTIVE_WHOLE_CELL`` (a regular lattice plus smaller whole
   hexagons in the left-over boundary region) and ``CLIPPED_BOUNDARY`` (a
   regular lattice whose boundary cells are clipped to the eligible region,
   provided the clipped piece is wide and large enough to print).
4. ``staged_fill_cells`` applies the same staged rule as before (one whole
   layer per cell whenever every pixel of the cell has one layer of room below
   the lower envelope; re-measure; repeat) and counts every safety violation.

No gate is weakened: a pixel can be deposited on only if it passes all of the
v1 rules. Printing stays disabled.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

from modules.conformal_mapping import compute_nozzle_orientations
from pipeline.unified import _cell_normals, _greedy_order

OUTSIDE_WOUND, BOUNDARY_MARGIN, NO_SUPPORT, UNCERTAINTY_GATE, BELOW_ONE_LAYER, ELIGIBLE, COVERED = range(7)
CATEGORY_NAMES = {
    OUTSIDE_WOUND: "outside the wound",
    BOUNDARY_MARGIN: "excluded boundary margin",
    NO_SUPPORT: "ineligible: no measured surface support",
    UNCERTAINTY_GATE: "ineligible: measurement uncertainty above gate",
    BELOW_ONE_LAYER: "ineligible: < 1 layer below lower uncertainty envelope",
    ELIGIBLE: "eligible but uncovered",
    COVERED: "covered by accepted honeycomb",
}
CATEGORY_COLOURS = {
    OUTSIDE_WOUND: "#ffffff",
    BOUNDARY_MARGIN: "#f4c7c3",
    NO_SUPPORT: "#adb5bd",
    UNCERTAINTY_GATE: "#9d4edd",
    BELOW_ONE_LAYER: "#ffd166",
    ELIGIBLE: "#e63946",
    COVERED: "#2a9d8f",
}

COVERAGE_DEFAULTS = {
    # A deposited bead is assumed to be as wide as one layer is thick. A hexagon side shorter
    # than two bead widths would let opposite walls merge and close the pore.
    "strand_width_du": 0.0002,
    "min_side_strand_widths": 2.0,
    "side_steps": 5,                     # sides evenly spaced from the v1 side down to the minimum
    "rotations_deg": (0, 10, 20, 30, 40, 50),   # a hexagonal lattice repeats every 60 degrees
    "offset_steps": 4,                   # staggered offsets u, v in {0, 1/4, 1/2, 3/4} lattice vectors
    "clip_min_area_fraction": 0.25,      # a clipped cell keeps at least a quarter of a full hexagon
    "clip_min_width_strands": 2.0,       # and survives an opening by a disk two bead widths wide
    "coverage_gate": 0.90,               # eligible-safe-area coverage needed for COMPLETE
    # Diagnostic only (never planned): whole-cell coverage with cells below the printable minimum,
    # used to show which constraint limits the whole-cell planner.
    "diagnostic_side_strand_widths": (1.5, 1.0),
}
PLANNER_PREFERENCE = ("WHOLE_CELL", "ADAPTIVE_WHOLE_CELL", "CLIPPED_BOUNDARY")


# ----------------------------------------------------------------------------- eligibility

def eligibility(field: dict, target: np.ndarray, params: dict) -> dict:
    """Classify pixels with the unchanged v1 rules (see ``pipeline.patient._supported``)."""
    layer = params["layer_thickness_du"]
    wound, valid = field["wound_mask"], field["valid"]
    uncertainty = np.where(valid, params["measurement_uncertainty_du"], np.inf)
    distance = cv2.distanceTransform(wound.astype(np.uint8), cv2.DIST_L2, 5)
    margin_px = params["safety_margin_fraction"] * field["boundary_width_px"] / 2.0
    beyond = wound & (distance >= margin_px)
    uncertainty_ok = uncertainty <= params["max_uncertainty_fraction_of_layer"] * layer
    supported = beyond & valid & uncertainty_ok
    room = np.where(valid, target - (np.nan_to_num(field["height"]) + np.where(valid, uncertainty, 0.0)), -np.inf)
    eligible = supported & (room >= layer)
    category = np.full(wound.shape, OUTSIDE_WOUND, dtype=np.int8)
    category[wound & ~beyond] = BOUNDARY_MARGIN
    category[beyond & ~valid] = NO_SUPPORT
    category[beyond & valid & ~uncertainty_ok] = UNCERTAINTY_GATE
    category[supported & ~eligible] = BELOW_ONE_LAYER
    category[eligible] = ELIGIBLE
    return {
        "wound": wound,
        "margin_band": wound & ~beyond,
        "uncertainty": uncertainty,
        "uncertainty_ok": uncertainty_ok,
        "supported": supported,
        "room_du": room,
        "eligible": eligible,
        "category": category,
        "margin_px": float(margin_px),
    }


# ----------------------------------------------------------------------------- geometry helpers

def hex_vertices(centre: np.ndarray, radius: float, theta: float) -> np.ndarray:
    angles = theta + np.radians(np.arange(0, 360, 60))
    return np.column_stack((centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)))


def hex_outline(centre: np.ndarray, radius: float, theta: float, per_edge: int) -> np.ndarray:
    """Perimeter points of a rotated flat-top hexagon (theta = 0 matches ``modules.honeycomb``)."""
    vertices = hex_vertices(centre, radius, theta)
    closed = np.vstack((vertices, vertices[:1]))
    points = [np.column_stack((np.linspace(closed[i, 0], closed[i + 1, 0], per_edge, endpoint=False),
                               np.linspace(closed[i, 1], closed[i + 1, 1], per_edge, endpoint=False)))
              for i in range(6)]
    return np.vstack(points)


def _round_px(points: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    cols = np.clip(np.rint(points[:, 0]).astype(int), 0, shape[1] - 1)
    rows = np.clip(np.rint(points[:, 1]).astype(int), 0, shape[0] - 1)
    return rows, cols


def segment_crossings(segments: np.ndarray, chunk: int = 512) -> int:
    """Number of pairs of segments that cross properly (touching end points are not counted)."""
    if len(segments) < 2:
        return 0
    a, b = segments[:, 0], segments[:, 1]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    eps = 1e-9
    total = 0
    for start in range(0, len(segments), chunk):
        sl = slice(start, start + chunk)
        overlap = ((lo[sl, None, 0] <= hi[None, :, 0]) & (lo[None, :, 0] <= hi[sl, None, 0])
                   & (lo[sl, None, 1] <= hi[None, :, 1]) & (lo[None, :, 1] <= hi[sl, None, 1]))
        i, j = np.nonzero(overlap)
        i = i + start
        keep = j > i
        i, j = i[keep], j[keep]
        if not len(i):
            continue

        def orient(p, q, r):
            return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

        o1 = orient(a[i], b[i], a[j])
        o2 = orient(a[i], b[i], b[j])
        o3 = orient(a[j], b[j], a[i])
        o4 = orient(a[j], b[j], b[i])
        total += int(np.count_nonzero((o1 * o2 < -eps) & (o3 * o4 < -eps)))
    return total


def _path_segments(cells: list[dict], indices) -> np.ndarray:
    segments = []
    for k in indices:
        path = cells[k]["path_px"]
        segments.append(np.stack((path[:-1], path[1:]), axis=1))
    return np.concatenate(segments) if segments else np.empty((0, 2, 2))


# ----------------------------------------------------------------------------- lattices

def _roi(field: dict, pad: int) -> dict:
    height, width = field["wound_mask"].shape
    ys, xs = np.nonzero(field["wound_mask"])
    r0, r1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, height)
    c0, c1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, width)
    rows, cols = np.mgrid[r0:r1, c0:c1]
    return {"shape": (r1 - r0, c1 - c0), "origin": (r0, c0), "rows": rows.ravel(), "cols": cols.ravel(),
            "flat": (rows * width + cols).ravel(), "xy": np.column_stack((cols.ravel(), rows.ravel())).astype(float)}


def lattice_centres(origin_px, side_px: float, theta: float, offset, roi: dict) -> np.ndarray:
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    a1 = rot @ (side_px * np.array([1.5, math.sqrt(3) / 2]))
    a2 = rot @ (side_px * np.array([0.0, math.sqrt(3)]))
    reach = float(np.max(np.linalg.norm(roi["xy"] - np.asarray(origin_px), axis=1))) + 2 * side_px
    n = int(math.ceil(reach / (1.5 * side_px))) + 2
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    centres = (np.asarray(origin_px) + (i.ravel() + offset[0])[:, None] * a1
               + (j.ravel() + offset[1])[:, None] * a2)
    lo, hi = roi["xy"].min(axis=0) - 2 * side_px, roi["xy"].max(axis=0) + 2 * side_px
    return centres[np.all((centres >= lo) & (centres <= hi), axis=1)]


def _assign(roi: dict, centres: np.ndarray) -> np.ndarray:
    return cKDTree(centres).query(roi["xy"])[1]


def _group(labels: np.ndarray, n: int) -> list[np.ndarray]:
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(n + 1))
    return [order[bounds[k]:bounds[k + 1]] for k in range(n)]


def _whole_cell(centre, side_px, theta, pixels_flat, params, kind="whole") -> dict:
    outline = hex_outline(centre, side_px * params["perimeter_shrink"], theta, params["perimeter_points_per_edge"])
    path = np.vstack((outline, outline[:1], np.asarray(centre)[None, :]))
    return {"kind": kind, "centre_px": np.asarray(centre, float), "side_px": float(side_px), "theta": float(theta),
            "pixels": np.asarray(pixels_flat), "path_px": path,
            "footprint_px": hex_vertices(centre, side_px, theta), "radius_px": float(side_px)}


def whole_cells(allowed: np.ndarray, roi: dict, centres: np.ndarray, side_px: float, theta: float,
                params: dict, labels: np.ndarray | None = None) -> tuple[list[dict], np.ndarray, list[np.ndarray]]:
    """Accept every lattice hexagon whose pixels and deposition points all lie in ``allowed``."""
    labels = _assign(roi, centres) if labels is None else labels
    allowed_roi = allowed.ravel()[roi["flat"]]
    total = np.bincount(labels, minlength=len(centres))
    good = np.bincount(labels, weights=allowed_roi.astype(float), minlength=len(centres))
    groups = _group(labels, len(centres))
    cells = []
    for k in np.flatnonzero((total > 0) & (good == total)):
        cell = _whole_cell(centres[k], side_px, theta, roi["flat"][groups[k]], params)
        rows, cols = _round_px(cell["path_px"], allowed.shape)
        if allowed[rows, cols].all():
            cell["lattice_index"] = int(k)
            cells.append(cell)
    return cells, labels, groups


def _clipped_cells(allowed: np.ndarray, roi: dict, centres: np.ndarray, side_px: float, theta: float,
                   labels: np.ndarray, groups: list[np.ndarray], taken: set[int], params: dict,
                   cov: dict) -> list[dict]:
    """Clip partially eligible lattice cells to the eligible region when the piece is printable."""
    width = allowed.shape[1]
    allowed_roi = allowed.ravel()[roi["flat"]]
    hex_area = 1.5 * math.sqrt(3) * side_px ** 2
    strand_px = cov["strand_width_du"] / params["grid_pixel_du"]
    k_open = 2 * int(math.floor(cov["clip_min_width_strands"] * strand_px / 2)) + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cells = []
    partial = np.bincount(labels, weights=allowed_roi.astype(float), minlength=len(centres))
    for k in np.flatnonzero(partial > 0):
        if int(k) in taken:
            continue
        members = groups[k][allowed_roi[groups[k]]]
        rows, cols = roi["rows"][members], roi["cols"][members]
        r0, c0 = rows.min() - 3, cols.min() - 3
        crop = np.zeros((rows.max() - r0 + 4, cols.max() - c0 + 4), np.uint8)
        crop[rows - r0, cols - c0] = 1
        opened = cv2.morphologyEx(crop, cv2.MORPH_OPEN, disk)
        count, comp = cv2.connectedComponents(opened, connectivity=4)
        for c in range(1, count):
            piece = (comp == c).astype(np.uint8)
            if piece.sum() < cov["clip_min_area_fraction"] * hex_area:
                continue
            inner = cv2.erode(piece, cross)
            if cv2.connectedComponents(inner, connectivity=8)[0] != 2:
                continue
            contours, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=len)[:, 0, :].astype(float)
            if len(contour) < 3 or len(np.unique(contour, axis=0)) != len(contour):
                continue
            outline = np.vstack((contour, contour[:1])) + [c0, r0]
            pr, pc = np.nonzero(piece)
            boundary, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            footprint = max(boundary, key=len)[:, 0, :].astype(float) + [c0, r0]
            centre = np.array([pc.mean() + c0, pr.mean() + r0])
            cells.append({"kind": "clipped", "centre_px": centre, "side_px": float(side_px), "theta": float(theta),
                          "pixels": (pr + r0) * width + (pc + c0), "path_px": outline, "footprint_px": footprint,
                          "radius_px": float(np.max(np.linalg.norm(outline - centre, axis=1))),
                          "lattice_index": int(k)})
    return cells


def _no_new_crossings(candidate: dict, existing: list[dict], tree: cKDTree | None, max_radius: float) -> bool:
    if tree is None:
        return True
    near = tree.query_ball_point(candidate["centre_px"], candidate["radius_px"] + max_radius + 1.0)
    if not near:
        return True
    segments = np.concatenate([_path_segments([candidate], [0])] + [_path_segments(existing, [k]) for k in near])
    own = len(candidate["path_px"]) - 1
    return segment_crossings(segments) == segment_crossings(segments[:own])


def search_planners(field: dict, elig: dict, params: dict, cov: dict) -> dict:
    """Compact search over hexagon side, rotation and staggered offset for the three planners."""
    pixel = params["grid_pixel_du"]
    # The v1 side is the wound's narrow width / cells across (build_honeycomb reduces to this).
    v1_side = field["narrow_width_px"] / params["honeycomb_cells_across_narrow_axis"]
    min_side = cov["min_side_strand_widths"] * cov["strand_width_du"] / pixel
    sides = np.linspace(v1_side, min_side, cov["side_steps"])
    steps = cov["offset_steps"]
    offsets = [(u / steps, v / steps) for u in range(steps) for v in range(steps)]
    eligible = elig["eligible"]
    eligible_px = int(eligible.sum())
    if eligible_px == 0:
        empty = {"score": (0.0, float(v1_side)), "config": None, "cells": []}
        return {"sides_px": sides.tolist(), "v1_side_px": float(v1_side), "min_side_px": float(min_side),
                "offsets": offsets, "records": [], "adaptive_records": [],
                "best": {name: dict(empty) for name in PLANNER_PREFERENCE}, "eligible_px": 0,
                "diagnostic_below_printable_minimum": []}
    roi = _roi(field, int(math.ceil(2 * sides.max())) + 4)
    ys, xs = np.nonzero(eligible)
    origin = (float(xs.mean()), float(ys.mean()))

    records, best = [], {name: None for name in PLANNER_PREFERENCE}
    best_whole_per_side = {}

    def better(score, incumbent):
        return incumbent is None or score > incumbent["score"]

    for side in sides:
        for theta_deg in cov["rotations_deg"]:
            theta = math.radians(theta_deg)
            for offset in offsets:
                centres = lattice_centres(origin, side, theta, offset, roi)
                cells, labels, groups = whole_cells(eligible, roi, centres, side, theta, params)
                covered = sum(len(c["pixels"]) for c in cells)
                taken = {c["lattice_index"] for c in cells}
                clipped = _clipped_cells(eligible, roi, centres, side, theta, labels, groups, taken, params, cov)
                clipped_cover = covered + sum(len(c["pixels"]) for c in clipped)
                config = {"side_px": float(side), "theta_deg": theta_deg, "offset_u": offset[0], "offset_v": offset[1]}
                records.append({**config, "whole_cells": len(cells), "whole_coverage": covered / eligible_px,
                                "clipped_cells": len(clipped), "clipped_coverage": clipped_cover / eligible_px})
                # Ties go to the larger side (fewer, more regular pores), then to the first configuration.
                score = (round(covered / eligible_px, 9), side)
                if better(score, best["WHOLE_CELL"]):
                    best["WHOLE_CELL"] = {"score": score, "config": config, "cells": cells}
                key = float(side)
                if key not in best_whole_per_side or score > best_whole_per_side[key]["score"]:
                    best_whole_per_side[key] = {"score": score, "config": config, "cells": cells}
                score = (round(clipped_cover / eligible_px, 9), side)
                if better(score, best["CLIPPED_BOUNDARY"]):
                    best["CLIPPED_BOUNDARY"] = {"score": score, "config": config, "cells": cells + clipped}

    adaptive_records = []
    for base_side, base in sorted(best_whole_per_side.items(), reverse=True):
        cells = list(base["cells"])
        covered_mask = np.zeros(eligible.size, bool)
        for cell in cells:
            covered_mask[cell["pixels"]] = True
        levels = [{"side_px": base_side, **{k: base["config"][k] for k in ("theta_deg", "offset_u", "offset_v")},
                   "cells": len(cells)}]
        for side in [s for s in sides if s < base_side - 1e-9]:
            remaining = eligible & ~covered_mask.reshape(eligible.shape)
            tree = cKDTree([c["centre_px"] for c in cells]) if cells else None
            max_radius = max((c["radius_px"] for c in cells), default=0.0)
            level_best = None
            for theta_deg in cov["rotations_deg"]:
                for offset in offsets:
                    centres = lattice_centres(origin, side, math.radians(theta_deg), offset, roi)
                    found, _, _ = whole_cells(remaining, roi, centres, side, math.radians(theta_deg), params)
                    found = [c for c in found if _no_new_crossings(c, cells, tree, max_radius)]
                    gain = sum(len(c["pixels"]) for c in found)
                    if level_best is None or gain > level_best[0]:
                        level_best = (gain, theta_deg, offset, found)
            gain, theta_deg, offset, found = level_best
            if gain:
                for cell in found:
                    cell["kind"] = "adaptive"
                    covered_mask[cell["pixels"]] = True
                cells += found
                levels.append({"side_px": float(side), "theta_deg": theta_deg, "offset_u": offset[0],
                               "offset_v": offset[1], "cells": len(found)})
        coverage = covered_mask.sum() / eligible_px
        adaptive_records.append({"base_side_px": base_side, "levels": levels, "cells": len(cells),
                                 "coverage": coverage})
        score = (round(coverage, 9), base_side)
        if better(score, best["ADAPTIVE_WHOLE_CELL"]):
            best["ADAPTIVE_WHOLE_CELL"] = {"score": score, "config": {"levels": levels}, "cells": cells}

    diagnostic = []
    for widths in cov["diagnostic_side_strand_widths"]:
        side = widths * cov["strand_width_du"] / pixel
        coverage = 0.0
        for theta_deg in cov["rotations_deg"]:
            for offset in offsets:
                centres = lattice_centres(origin, side, math.radians(theta_deg), offset, roi)
                cells, _, _ = whole_cells(eligible, roi, centres, side, math.radians(theta_deg), params)
                coverage = max(coverage, sum(len(c["pixels"]) for c in cells) / eligible_px)
        diagnostic.append({"side_strand_widths": widths, "side_px": side, "best_whole_coverage": coverage,
                           "printable": False})

    return {"sides_px": sides.tolist(), "v1_side_px": float(v1_side), "min_side_px": float(min_side),
            "offsets": offsets, "records": records, "adaptive_records": adaptive_records, "best": best,
            "eligible_px": eligible_px, "diagnostic_below_printable_minimum": diagnostic}


# ----------------------------------------------------------------------------- v1 lattice audit

def audit_lattice(honeycomb: dict, elig: dict, field: dict) -> dict:
    """Why each v1 lattice cell that touches the wound was or was not used."""
    labels = honeycomb["labels"]
    wound, valid = elig["wound"], field["valid"]
    reasons = []
    for cell in np.unique(labels[wound]):
        pix = labels == cell
        flags = {
            "complete_hexagon_outside_wound": bool((pix & ~wound).any()),
            "boundary_margin": bool((pix & elig["margin_band"]).any()),
            "missing_support": bool((pix & wound & ~valid).any()),
            "uncertainty_gate": bool((pix & wound & valid & ~elig["uncertainty_ok"]).any()),
            "below_one_layer": bool((pix & elig["supported"] & ~elig["eligible"]).any()),
        }
        flags["partially_eligible"] = bool((pix & elig["eligible"]).any()) and not bool((pix & ~elig["eligible"]).all())
        primary = next((name for name, hit in flags.items() if hit and name != "partially_eligible"), "accepted")
        reasons.append({"cell": int(cell), "primary": primary, **flags,
                        "eligible_pixels": int((pix & elig["eligible"]).sum()), "pixels": int(pix.sum())})
    rejected = [r for r in reasons if r["primary"] != "accepted"]
    any_counts = {name: sum(r[name] for r in rejected) for name in (
        "complete_hexagon_outside_wound", "boundary_margin", "missing_support", "uncertainty_gate", "below_one_layer")}
    primary_counts = {name: sum(r["primary"] == name for r in rejected) for name in any_counts}
    return {
        "cells_touching_wound": len(reasons),
        "cells_accepted": len(reasons) - len(rejected),
        "cells_rejected": len(rejected),
        # v1 rejects a cell exactly when its complete hexagon does not fit inside the supported region.
        "rejected_complete_hexagon_did_not_fit_supported_region": len(rejected),
        "rejected_by_reason_any": any_counts,
        "rejected_by_primary_reason": primary_counts,
        "primary_reason_order": ["complete_hexagon_outside_wound", "boundary_margin", "missing_support",
                                 "uncertainty_gate", "below_one_layer"],
        "rejected_cells_with_eligible_pixels": sum(r["eligible_pixels"] > 0 for r in rejected),
        "eligible_pixels_in_rejected_cells": int(sum(r["eligible_pixels"] for r in rejected)),
        "cells": reasons,
    }


def v1_cells(honeycomb: dict, supported: np.ndarray, params: dict, field: dict) -> list[dict]:
    """The cells the v1 planner accepted, expressed in the generic cell format."""
    labels = honeycomb["labels"]
    n = len(honeycomb["centres_px"])
    fully = np.ones(n, bool)
    fully[np.unique(labels[~supported])] = False
    fully &= np.bincount(labels.ravel(), minlength=n) > 0
    side_px = honeycomb["hex_side_m"] / field["pixel_du"]
    cells = []
    for k in np.flatnonzero(fully):
        cell = _whole_cell(honeycomb["centres_px"][k], side_px, 0.0, np.flatnonzero(labels.ravel() == k), params,
                           kind="v1")
        cell["lattice_index"] = int(k)
        cells.append(cell)
    return cells


# ----------------------------------------------------------------------------- ordering and toolpath

def order_cells(cells: list[dict], indices: list[int]) -> dict:
    """Greedy nearest-neighbour order, then 2-opt; travel measured between real path end points."""
    if not indices:
        return {"order": [], "travel_greedy_px": 0.0, "travel_two_opt_px": 0.0}
    starts = np.array([cells[k]["path_px"][0] for k in indices])
    ends = np.array([cells[k]["path_px"][-1] for k in indices])

    def travel(route):
        return float(sum(np.linalg.norm(starts[b] - ends[a]) for a, b in zip(route[:-1], route[1:])))

    centres = np.array([cells[k]["centre_px"] for k in indices])
    greedy = _greedy_order(centres)
    route = list(greedy)
    dist = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2).tolist()
    n = len(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b, c = route[i - 1], route[i], route[k]
                d = route[k + 1] if k + 1 < n else None
                delta = dist[a][c] - dist[a][b] + ((dist[b][d] - dist[c][d]) if d is not None else 0.0)
                if delta < -1e-9:
                    route[i:k + 1] = route[i:k + 1][::-1]
                    improved = True
    if travel(route) > travel(greedy):
        route = list(greedy)
    return {"order": [indices[r] for r in route], "travel_greedy_px": travel(greedy),
            "travel_two_opt_px": travel(route)}


def cells_toolpath(cells, order, planned_top, normals, field, params, frame, to_world) -> dict:
    local_pts, local_normals, cell_ids, flags = [], [], [], []
    rise = np.array([0.0, 0.0, params["travel_rise_du"]])
    for k in order:
        rows, cols = _round_px(cells[k]["path_px"], planned_top.shape)
        xy = np.column_stack((field["axis"][cols], field["axis"][rows]))
        tool = np.column_stack((xy, planned_top[rows, cols])) + normals[k] * params["tool_to_surface_offset_du"]
        for point, flag in [(tool[0] + rise, 0)] + [(p, 1) for p in tool] + [(tool[-1] + rise, 0)]:
            local_pts.append(point)
            local_normals.append(normals[k])
            cell_ids.append(k)
            flags.append(bool(flag))
    if not local_pts:
        empty = np.empty((0, 3))
        return {"local": empty, "world": empty, "normals_world": empty, "cell": np.empty(0, int),
                "deposit_flag": np.empty(0, bool), "rotations": np.empty((3, 3, 0))}
    local = np.asarray(local_pts)
    normals_world = np.asarray(local_normals) @ np.vstack((frame["e1"], frame["e2"], frame["normal"]))
    return {"local": local, "world": to_world(local, frame), "normals_world": normals_world,
            "cell": np.asarray(cell_ids), "deposit_flag": np.asarray(flags),
            "rotations": compute_nozzle_orientations(normals_world.T)}


# ----------------------------------------------------------------------------- staged fill

def staged_fill_cells(field: dict, target: np.ndarray, frame: dict, cells: list[dict], elig: dict,
                      params: dict, to_world, planner: str) -> dict:
    """The v1 staged rule on a fixed set of cells, with every safety invariant counted."""
    layer = params["layer_thickness_du"]
    pixel = field["pixel_du"]
    rng = np.random.default_rng(params["seed"])
    sigma = params["measurement_uncertainty_du"] / 3.0
    uncertainty, supported, wound = elig["uncertainty"], elig["supported"], elig["wound"]
    unsupported_pixel = wound & (~field["valid"] | ~elig["uncertainty_ok"])
    shape = wound.shape
    label_map = np.full(wound.size, -1, dtype=np.int32)
    overlaps = 0
    for k, cell in enumerate(cells):
        overlaps += int(np.count_nonzero(label_map[cell["pixels"]] >= 0))
        label_map[cell["pixels"]] = k
    label_map = label_map.reshape(shape)

    true_height = field["height"].copy()
    measured = field["height"].copy()
    deposited_total = np.zeros_like(true_height)
    ever_active = np.zeros(len(cells), bool)
    eligible0 = elig["eligible"]
    area = pixel ** 2

    def remaining(surface):
        gap = np.where(wound & field["valid"], target - surface, -np.inf)
        fillable = np.where(supported, target - (surface + uncertainty), -np.inf)
        return float((gap >= layer).sum() * area), float((fillable >= layer).sum() * area)

    cavity0, fillable0 = remaining(true_height)
    stages, stop_reason, stop_detail = [], None, {}
    for stage in range(1, params["max_stages"] + 1):
        free = np.where(supported, target - (measured + uncertainty), np.nan)
        max_remaining = float(np.nanmax(free)) if np.isfinite(free).any() else 0.0
        if max_remaining < layer:
            stop_reason = "SUPPORTED_DEPTH_BELOW_ONE_LAYER"
            break
        room = np.nan_to_num(free, nan=-np.inf).ravel()
        cell_min = np.array([room[c["pixels"]].min() for c in cells]) if cells else np.empty(0)
        active = [k for k in range(len(cells)) if cell_min[k] >= layer]
        if not active:
            stop_reason = "NO_SAFE_SUPPORTED_REGION"
            room_px = np.isfinite(free) & (free >= layer)
            in_cell = room_px & (label_map >= 0)
            stop_detail = {
                "supported_pixels_with_one_layer_room": int(room_px.sum()),
                "of_which_inside_accepted_cells": int(in_cell.sum()),
                "of_which_outside_every_accepted_cell": int((room_px & (label_map < 0)).sum()),
                "cells_with_some_room_but_min_below_one_layer": int(len(np.unique(label_map[in_cell]))),
                "largest_cell_min_room_du": float(cell_min.max()) if len(cell_min) else None,
                "max_room_du": max_remaining,
            }
            break
        ever_active[active] = True
        deposit = np.isin(label_map, active) * layer
        planned_top = np.where(field["valid"], measured + deposit, np.nan)
        normals = _cell_normals(measured, label_map, active, pixel)
        ordering = order_cells(cells, active)
        path = cells_toolpath(cells, ordering["order"], planned_top, normals, field, params, frame, to_world)
        rows = np.clip(np.rint((path["local"][:, 1] - field["axis"][0]) / pixel).astype(int), 0, shape[0] - 1)
        cols = np.clip(np.rint((path["local"][:, 0] - field["axis"][0]) / pixel).astype(int), 0, shape[1] - 1)
        rc = (rows[path["deposit_flag"]], cols[path["deposit_flag"]])
        dep = deposit > 0
        checks = {
            "deposit_pixels_outside_wound": int((dep & ~wound).sum()),
            "deposit_pixels_in_exclusion_margin": int((dep & elig["margin_band"]).sum()),
            "deposit_pixels_unsupported": int((dep & unsupported_pixel).sum()),
            "deposit_pixels_above_lower_envelope": int((dep & ~(np.nan_to_num(free, nan=-np.inf) >= layer)).sum()),
            "path_points_outside_wound": int(np.count_nonzero(~wound[rc])),
            "path_points_in_exclusion_margin": int(np.count_nonzero(elig["margin_band"][rc])),
            "path_points_unsupported": int(np.count_nonzero(~supported[rc] & wound[rc] & ~elig["margin_band"][rc])),
            "path_self_intersections": segment_crossings(_path_segments(cells, active)),
        }
        true_height = np.where(field["valid"], true_height + deposit, true_height)
        deposited_total += deposit
        measured = np.where(field["valid"], true_height + rng.normal(0.0, sigma, true_height.shape), np.nan)
        printed = deposited_total > 0
        overfill = float(np.max(true_height[printed] - target[printed]))
        cavity, fillable = remaining(true_height)
        stages.append({
            "stage": stage,
            "active_cells": len(active),
            "toolpath_points": int(len(path["local"])),
            "deposition_points": int(path["deposit_flag"].sum()),
            "max_supported_remaining_before_du": max_remaining,
            "deposited_volume_du3": float(deposit.sum() * area),
            "travel_greedy_du": ordering["travel_greedy_px"] * pixel,
            "travel_two_opt_du": ordering["travel_two_opt_px"] * pixel,
            "remaining_cavity_area_du2": cavity,
            "remaining_fillable_supported_area_du2": fillable,
            "max_overfill_vs_target_du": max(overfill, 0.0),
            **checks,
            "boundary_violations": checks["path_points_outside_wound"] + checks["path_points_in_exclusion_margin"],
            "unsupported_cells": int(sum(not supported.ravel()[cells[k]["pixels"]].all() for k in active)),
            "_toolpath": path,
            "_deposit": deposit,
            "_surface_after": true_height.copy(),
        })
    stop_reason = stop_reason or "STAGE_LIMIT_REACHED"

    footprint = np.zeros(wound.size, bool)
    for k in np.flatnonzero(ever_active):
        footprint[cells[k]["pixels"]] = True
    footprint = footprint.reshape(shape)
    category = elig["category"].copy()
    category[footprint & eligible0] = COVERED
    wound_px, eligible_px = int(wound.sum()), int(eligible0.sum())
    covered_eligible = int((footprint & eligible0).sum())
    violation_keys = ("deposit_pixels_outside_wound", "deposit_pixels_in_exclusion_margin", "deposit_pixels_unsupported",
                      "deposit_pixels_above_lower_envelope", "path_points_outside_wound",
                      "path_points_in_exclusion_margin", "path_points_unsupported", "path_self_intersections")
    violations = {key: int(sum(s[key] for s in stages)) for key in violation_keys}
    violations["overlapping_cell_pixels"] = overlaps
    violations["deposit_pixels_outside_initial_eligible_region"] = int((footprint & ~eligible0).sum())
    gap = np.where(wound & field["valid"], target - true_height, np.nan)
    fill_gap = np.where(eligible0, target - (true_height + uncertainty), np.nan)
    eligible_coverage = covered_eligible / max(eligible_px, 1)
    any_violation = any(v > 0 for v in violations.values())
    gate = params.get("coverage_gate", COVERAGE_DEFAULTS["coverage_gate"])
    status = "REJECTED" if (any_violation or not ever_active.any()) else (
        "COMPLETE" if eligible_coverage >= gate else "PARTIAL")
    summary = {
        "planner": planner,
        "status": status,
        "coverage_gate": gate,
        "cells": len(cells),
        "cells_by_kind": {kind: sum(c["kind"] == kind for c in cells) for kind in sorted({c["kind"] for c in cells})},
        "wound_area_du2": wound_px * area,
        "supported_area_du2": float(supported.sum() * area),
        "eligible_area_du2": eligible_px * area,
        "covered_area_du2": float(footprint.sum() * area),
        "covered_eligible_area_du2": covered_eligible * area,
        "total_wound_coverage": float((footprint & wound).sum() / max(wound_px, 1)),
        "eligible_coverage": eligible_coverage,
        "uncovered_eligible_area_du2": (eligible_px - covered_eligible) * area,
        "category_area_du2": {CATEGORY_NAMES[c]: float((category == c).sum() * area) for c in CATEGORY_NAMES},
        "remaining_cavity_area_du2_by_stage": [cavity0] + [s["remaining_cavity_area_du2"] for s in stages],
        "remaining_fillable_supported_area_du2_by_stage": [fillable0] + [s["remaining_fillable_supported_area_du2"]
                                                                         for s in stages],
        "stage_count": len(stages),
        "active_cells_by_stage": [s["active_cells"] for s in stages],
        "stop_reason": stop_reason,
        "stop_detail": stop_detail,
        "violations": violations,
        "boundary_violations": violations["path_points_outside_wound"] + violations["path_points_in_exclusion_margin"]
        + violations["deposit_pixels_outside_wound"] + violations["deposit_pixels_in_exclusion_margin"],
        "unsupported_cells": int(sum(s["unsupported_cells"] for s in stages)),
        "max_overfill_vs_target_du": float(max([s["max_overfill_vs_target_du"] for s in stages], default=0.0)),
        "final_max_gap_du": float(np.nanmax(np.clip(gap, 0, None))),
        "final_max_gap_in_eligible_du": float(np.nanmax(np.clip(target - true_height, 0, None)[eligible0]))
        if eligible_px else 0.0,
        "final_eligible_area_still_one_layer_below_envelope_du2": float((fill_gap >= layer).sum() * area),
        "unresolved_area_fraction_of_wound": float(((gap >= layer) & wound).sum() / max(wound_px, 1)),
        "deposited_volume_du3": float(sum(s["deposited_volume_du3"] for s in stages)),
        "travel_greedy_du": float(sum(s["travel_greedy_du"] for s in stages)),
        "travel_two_opt_du": float(sum(s["travel_two_opt_du"] for s in stages)),
    }
    return {"planner": planner, "cells": cells, "label_map": label_map, "supported": supported,
            "eligible": eligible0, "category": category, "footprint": footprint, "in_lattice": label_map >= 0,
            "ever_active": ever_active, "stages": stages, "final_height": true_height, "summary": summary}


def binding_constraint(results: dict, elig: dict, field: dict, params: dict) -> str:
    """Plain-language statement of what limits coverage (used only when no planner passes)."""
    best = max(results.values(), key=lambda r: r["summary"]["eligible_coverage"])
    s = best["summary"]
    return (f"Best eligible coverage {s['eligible_coverage']:.1%} ({best['planner']}). Uncovered eligible pixels lie "
            f"in strips narrower than one printable cell between accepted cells and the eligible boundary; the "
            f"binding constraints are the minimum printable cell size ({COVERAGE_DEFAULTS['min_side_strand_widths']:g} "
            f"strand widths) and the requirement that every deposited pixel be eligible.")


def plan_patient_coverage(field: dict, target: np.ndarray, frame: dict, params: dict, to_world,
                          v1_honeycomb: dict | None = None, cov: dict | None = None) -> dict:
    """Run the audit, the planner search and the staged fill for every planner; select one."""
    cov = dict(COVERAGE_DEFAULTS, **(cov or {}))
    fill_params = dict(params, coverage_gate=cov["coverage_gate"])
    elig = eligibility(field, target, params)
    search = search_planners(field, elig, params, cov)
    results = {}
    if v1_honeycomb is not None:
        results["V1_LATTICE"] = staged_fill_cells(field, target, frame, v1_cells(v1_honeycomb, elig["supported"],
                                                  params, field), elig, fill_params, to_world, "V1_LATTICE")
    for name in PLANNER_PREFERENCE:
        results[name] = staged_fill_cells(field, target, frame, search["best"][name]["cells"], elig, fill_params,
                                          to_world, name)
    passing = [name for name in PLANNER_PREFERENCE if results[name]["summary"]["status"] == "COMPLETE"]
    selected = passing[0] if passing else max(PLANNER_PREFERENCE,
                                              key=lambda n: results[n]["summary"]["eligible_coverage"])
    whole_best = max(results[n]["summary"]["eligible_coverage"] for n in ("WHOLE_CELL", "ADAPTIVE_WHOLE_CELL"))
    diag = ", ".join(f"{d['side_strand_widths']:g} strand widths -> {d['best_whole_coverage']:.1%}"
                     for d in search["diagnostic_below_printable_minimum"])
    whole_limit = (f"Whole-cell planners reach at most {whole_best:.1%} of the eligible area. Their best lattice already "
                   f"uses the smallest printable side ({cov['min_side_strand_widths']:g} strand widths); the uncovered "
                   f"eligible pixels lie along the curved eligible boundary where no complete hexagon fits. Diagnostic "
                   f"(not plannable, below the printable minimum): {diag}. The binding constraint for whole cells is "
                   f"the minimum printable cell size combined with the requirement that every deposited pixel be "
                   f"eligible.")
    return {
        "eligibility": elig,
        "search": search,
        "results": results,
        "selected": selected,
        "selection_rule": ("first planner in the order WHOLE_CELL, ADAPTIVE_WHOLE_CELL, CLIPPED_BOUNDARY whose staged "
                           "fill has zero violations and eligible coverage >= the gate; otherwise the highest coverage"),
        "binding_constraint": None if passing else binding_constraint(
            {n: results[n] for n in PLANNER_PREFERENCE}, elig, field, params),
        "whole_cell_limit": whole_limit,
        "audit_v1": audit_lattice(v1_honeycomb, elig, field) if v1_honeycomb is not None else None,
        "coverage_parameters": cov,
    }
