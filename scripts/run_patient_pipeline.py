"""Full-pipeline figure on a rendered patient leg: from photographs to a scaffold placed on the body.

Usage:
    python scripts/run_patient_pipeline.py [--scene data/rgb_wound/syn3dwound_leg]
                                           [--output results/patient_pipeline]

Inputs (not stored in Git; public, CC BY-NC-SA 4.0):
    Syn3DWound subject_1 / Syn3DWound_0415_leg_strong_lighting_phi
    cropped_mesh.obj, images/image_XXX.png, mask/mask_XXX.png, *_colmap/{cameras,images}.txt

Outputs:
    patient_pipeline.png / .pdf   eight-panel figure (300 dpi PNG)
    coverage_audit.png / .pdf     coverage audit of the honeycomb planners (300 dpi PNG)
    coverage_audit.json           previous-planner audit, planner search and staged-fill summaries
    coverage_search.csv           every searched lattice (side, rotation, offset, coverage)
    coverage_planners.csv         one row per planner (coverage, stages, violations, status)
    summary.json                  every number shown in the figure
    scaffold_toolpath_world.csv   staged toolpath in leg (dataset-unit) coordinates
    provenance.json               input paths, SHA-256 hashes, licence and steps
    caption.md                    caption draft
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.coverage import CATEGORY_COLOURS, CATEGORY_NAMES, ELIGIBLE  # noqa: E402
from pipeline.patient import (  # noqa: E402
    DEFAULT_SCENE,
    SEQUENCE_ID,
    camera_centre,
    project,
    run_patient,
    scaffold_prisms,
    sha256_file,
    to_world,
)

DEFAULT_OUTPUT = ROOT / "results" / "patient_pipeline"
MDU = 1e3  # plot lengths in milli-dataset-units
STAGE_COLOURS = ["#f4a261", "#2a9d8f", "#264653", "#8338ec"]
LICENCE = {
    "dataset": "Syn3DWound - A synthetic dataset for 3D wound bed analysis",
    "dataset_doi": "10.25919/5rwz-ts17",
    "collection_url": "https://data.csiro.au/collection/csiro:73258",
    "license": "Creative Commons Attribution Noncommercial-Share Alike 4.0 (CC BY-NC-SA 4.0)",
    "license_url": "https://data.csiro.au/licence/1202",
    "required_attribution": (
        "Chierchia, Remi; Lebrat, Leo; Santa Cruz, Rodrigo; Arzhaeva, Yulia; Armin, Ali; Joshua, Goldsmith; "
        "Oorloff, Jeremy; Reddy, Prithvi; Nguyen, Chuong; Petersson, Lars; Barakat-Johnson, Michelle; "
        "Luscombe, Georgina; Fookes, Clinton; Salvado, Olivier; & Ahmedt Aristizabal, David (2024): "
        "Syn3DWound - A synthetic dataset for 3D wound bed analysis. v1. CSIRO. Data Collection. "
        "https://doi.org/10.25919/5rwz-ts17"
    ),
}


def vertex_colours(result: dict, rgb: np.ndarray) -> np.ndarray:
    scene, view = result["scene"], result["view"]
    camera = scene["cameras"][view["camera_id"]]
    uv, _ = project(scene["vertices"], view, camera)
    cols = np.clip(np.rint(uv[:, 0]).astype(int), 0, rgb.shape[1] - 1)
    rows = np.clip(np.rint(uv[:, 1]).astype(int), 0, rgb.shape[0] - 1)
    return rgb[rows, cols] / 255.0


def shaded_faces(local: np.ndarray, faces: np.ndarray, colours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tris = local[faces]
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    light = np.array([0.3, -0.4, 0.87])
    shade = 0.55 + 0.45 * np.abs(normal @ (light / np.linalg.norm(light)))
    face_colour = colours[faces].mean(axis=1) * shade[:, None]
    return tris, np.clip(face_colour, 0, 1)


def mesh_axes(ax, local, faces, colours, keep_radius=None, title="", exaggeration=1.0, skip_faces=None):
    if keep_radius is not None:
        centre_distance = np.linalg.norm(local[faces].mean(axis=1)[:, :2], axis=1)
        faces = faces[centre_distance < keep_radius]
    if skip_faces is not None:
        faces = faces[~skip_faces(local[faces].mean(axis=1))]
    scaled = local * MDU * np.array([1.0, 1.0, exaggeration])
    tris, face_colour = shaded_faces(scaled, faces, colours ** 0.6)
    ax.computed_zorder = False
    ax.add_collection3d(Poly3DCollection(tris, facecolors=face_colour, edgecolors="none", rasterized=True))
    lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], max(hi[2] - lo[2], 0.15 * (hi[0] - lo[0]))))
    ax.view_init(elev=38, azim=-62)
    ax.set_xlabel("along wound (mDU)", fontsize=6, labelpad=-6)
    ax.set_ylabel("across (mDU)", fontsize=6, labelpad=-6)
    ax.set_zticks([])
    ax.tick_params(labelsize=5, pad=-3)
    ax.set_title(title, fontsize=8, loc="left")


def prism_polys(prisms: list[dict], exaggeration: float = 1.0) -> tuple[list, list, list]:
    scale = np.array([MDU, MDU, MDU * exaggeration])
    tops, walls, colours = [], [], []
    for prism in prisms:
        xy = prism["outline_xy"]
        bottom = np.column_stack((xy, np.full(len(xy), prism["bed"])))
        top = np.column_stack((xy, np.full(len(xy), prism["top"])))
        tops.append(top * scale)
        colours.append(STAGE_COLOURS[min(prism["layers"], len(STAGE_COLOURS)) - 1])
        for k in range(len(xy)):
            j = (k + 1) % len(xy)
            walls.append(np.array([bottom[k], bottom[j], top[j], top[k]]) * scale)
    return tops, walls, colours


PLANNER_LABELS = {
    "V1_LATTICE": "previous planner (one lattice, whole cells)",
    "WHOLE_CELL": "whole-cell search",
    "ADAPTIVE_WHOLE_CELL": "adaptive whole-cell (smaller cells at edge)",
    "CLIPPED_BOUNDARY": "boundary cells clipped to safe region",
}
STOP_EXPLANATIONS = {
    "NO_SAFE_SUPPORTED_REGION": "no accepted cell still has one full layer of room in every one of its pixels",
    "SUPPORTED_DEPTH_BELOW_ONE_LAYER": "no safe pixel has one layer of room left",
    "STAGE_LIMIT_REACHED": "the stage limit was reached",
}


def category_image(category: np.ndarray) -> np.ndarray:
    lut = np.array([matplotlib.colors.to_rgb(CATEGORY_COLOURS[k]) for k in sorted(CATEGORY_COLOURS)])
    return lut[category]


def px_to_local(points_px: np.ndarray, field: dict) -> np.ndarray:
    index = np.arange(len(field["axis"]))
    return np.column_stack((np.interp(points_px[:, 0], index, field["axis"]),
                            np.interp(points_px[:, 1], index, field["axis"])))


def mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :].astype(float) for c in contours if len(c) >= 3]


def draw_toolpath_2d(ax, toolpath: dict, colour: str = "black", lw: float = 0.3) -> None:
    for cell in np.unique(toolpath["cell"]):
        keep = (toolpath["cell"] == cell) & toolpath["deposit_flag"]
        ax.plot(toolpath["local"][keep, 0] * MDU, toolpath["local"][keep, 1] * MDU, color=colour, lw=lw)


def draw_coverage_map(ax, field: dict, plan: dict, frame: dict) -> None:
    """Pixel categories (covered, uncovered eligible, margin, ...) with accepted cell outlines."""
    p = field["pixel_du"] / 2
    a0, a1 = field["axis"][0], field["axis"][-1]
    ax.imshow(category_image(plan["category"]), extent=[(a0 - p) * MDU, (a1 + p) * MDU, (a1 + p) * MDU, (a0 - p) * MDU],
              interpolation="nearest")
    for index in np.flatnonzero(plan["ever_active"]):
        xy = px_to_local(plan["cells"][index]["footprint_px"], field) * MDU
        ax.add_patch(Polygon(xy, closed=True, fill=False, ec="#1b4332", lw=0.35))
    for mask, colour, style in ((field["wound_mask"], "black", "-"), (plan["eligible"], "#0077b6", "--")):
        for contour in mask_contours(mask):
            xy = px_to_local(np.vstack((contour, contour[:1])), field) * MDU
            ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=0.7, ls=style)
    span = 1.15 * frame["wound_radius"] * MDU
    ax.set_xlim(-span, span)
    ax.set_ylim(span * 0.6, -span * 0.6)
    ax.set_aspect("equal")


def make_coverage_figure(result: dict, audit: dict, output: Path) -> None:
    """Coverage audit: category maps for every planner, the v1 rejection reasons, the search and the stages."""
    coverage, field, frame = result["coverage"], result["field"], result["frame"]
    results = coverage["results"]
    fig = plt.figure(figsize=(16, 7.6))
    grid = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.3, height_ratios=[0.72, 1.0])
    for column, name in enumerate(("V1_LATTICE", "WHOLE_CELL", "ADAPTIVE_WHOLE_CELL", "CLIPPED_BOUNDARY")):
        plan = results[name]
        s = plan["summary"]
        ax = fig.add_subplot(grid[0, column])
        draw_coverage_map(ax, field, plan, frame)
        selected = "  [SELECTED]" if name == coverage["selected"] else ""
        ax.set_title(f"{chr(65 + column)}  {PLANNER_LABELS[name]}{selected}\n{s['cells']} cells; safe-area coverage "
                     f"{s['eligible_coverage']:.1%}; wound {s['total_wound_coverage']:.1%}\nstatus {s['status']}; "
                     f"violations {sum(s['violations'].values())}; stop {s['stop_reason']}", fontsize=7, loc="left")
        ax.set_xlabel("along wound (mDU)", fontsize=6)
        ax.set_ylabel("across wound (mDU)", fontsize=6)
        ax.tick_params(labelsize=5)
    row_bottom = ax.get_position().y0
    present = sorted(set(np.unique(np.concatenate([results[n]["category"].ravel() for n in results])).tolist()))
    handles = [Patch(fc=CATEGORY_COLOURS[k], ec="#555555", lw=0.3, label=CATEGORY_NAMES[k]) for k in present]
    handles += [Line2D([], [], color="black", lw=0.8, label="wound boundary"),
                Line2D([], [], color="#0077b6", lw=0.8, ls="--", label="safe (eligible) region boundary"),
                Line2D([], [], color="#1b4332", lw=0.6, label="accepted cell outline")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, row_bottom - 0.045), ncol=5, fontsize=6.5,
               frameon=False)

    a = audit["v1_lattice_audit"]
    ax = fig.add_subplot(grid[1, 0])
    reasons = list(a["rejected_by_reason_any"])
    y = np.arange(len(reasons))
    ax.barh(y + 0.2, [a["rejected_by_reason_any"][r] for r in reasons], 0.4, color="#adb5bd",
            label="cell touches this reason")
    ax.barh(y - 0.2, [a["rejected_by_primary_reason"][r] for r in reasons], 0.4, color="#e76f51",
            label="first reason (order shown)")
    ax.set_yticks(y, [r.replace("_", " ") for r in reasons], fontsize=6)
    ax.invert_yaxis()
    ax.tick_params(labelsize=6)
    ax.set_xlabel("rejected cells of the previous lattice", fontsize=6.5)
    ax.legend(fontsize=5.5, loc="lower right")
    ax.set_title(f"E  Previous planner: {a['cells_rejected']} of {a['cells_touching_wound']} cells rejected\n"
                 f"({a['rejected_cells_with_eligible_pixels']} of them held "
                 f"{a['eligible_pixels_in_rejected_cells'] * field['pixel_du'] ** 2 * 1e6:.1f} mDU² of safe area)",
                 fontsize=7, loc="left")

    ax = fig.add_subplot(grid[1, 1])
    records = coverage["search"]["records"]
    sides = sorted({r["side_px"] for r in records}, reverse=True)
    to_mdu = field["pixel_du"] * MDU
    whole = [max(r["whole_coverage"] for r in records if r["side_px"] == s) for s in sides]
    clipped = [max(r["clipped_coverage"] for r in records if r["side_px"] == s) for s in sides]
    adaptive = {r["base_side_px"]: r["coverage"] for r in coverage["search"]["adaptive_records"]}
    xs = np.array(sides) * to_mdu
    ax.plot(xs, np.array(whole) * 100, "o-", color="#264653", label="whole cells (best rotation/offset)")
    ax.plot(xs, [adaptive.get(s, np.nan) * 100 for s in sides], "s--", color="#8338ec",
            label="adaptive whole cells (base side)")
    ax.plot(xs, np.array(clipped) * 100, "^-", color="#2a9d8f", label="whole + clipped boundary cells")
    diag = coverage["search"]["diagnostic_below_printable_minimum"]
    ax.plot([d["side_px"] * to_mdu for d in diag], [d["best_whole_coverage"] * 100 for d in diag], "o", mfc="none",
            color="#264653", label="whole cells below printable size (not allowed)")
    v1 = results["V1_LATTICE"]["summary"]["eligible_coverage"]
    ax.plot([coverage["search"]["v1_side_px"] * to_mdu], [v1 * 100], "X", color="#e63946", ms=7,
            label="previous planner (one fixed lattice)")
    ax.axhline(100 * results["CLIPPED_BOUNDARY"]["summary"]["coverage_gate"], color="black", lw=0.8, ls=":",
               label="coverage gate")
    ax.axvline(coverage["search"]["min_side_px"] * to_mdu, color="#6c757d", lw=0.8, ls="-.",
               label="smallest printable side")
    ax.set_xlabel("hexagon side (mDU)", fontsize=6.5)
    ax.set_ylabel("planned safe-area coverage (%)", fontsize=6.5)
    ax.set_ylim(0, 105)
    ax.invert_xaxis()
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="lower left")
    ax.set_title(f"F  Planner search: {audit['search']['configurations_per_planner']} lattices\n"
                 f"({len(sides)} sizes x {len(audit['search']['rotations_deg'])} rotations x "
                 f"{len(audit['search']['offsets'])} offsets)", fontsize=7, loc="left")

    for column, key, label in ((2, "remaining_cavity_area_du2_by_stage", "G  Wound area still >= 1 layer below target"),
                               (3, "remaining_fillable_supported_area_du2_by_stage",
                                "H  Safe area still with >= 1 layer of room\n(below the lower uncertainty envelope)")):
        ax = fig.add_subplot(grid[1, column])
        for name, colour in zip(results, ("#e63946", "#264653", "#8338ec", "#2a9d8f")):
            values = np.array(results[name]["summary"][key]) * 1e6
            style = "--" if name == "ADAPTIVE_WHOLE_CELL" else "-"
            ax.plot(np.arange(len(values)), values, style, marker="o", color=colour, ms=3, label=PLANNER_LABELS[name])
        ax.set_xlabel("after stage (0 = before printing)", fontsize=6.5)
        ax.set_ylabel("area (mDU²)", fontsize=6.5)
        ax.set_xticks(range(max(len(results[n]["summary"][key]) for n in results)))
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="lower left")
        ax.set_title(label, fontsize=7, loc="left")

    fig.suptitle("Honeycomb coverage audit on the rendered patient leg: same safety gates, four planners "
                 "(plan only; printing disabled)", fontsize=10, y=0.995)
    fig.savefig(output / "coverage_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "coverage_audit.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary(result: dict) -> dict:
    field, fill, frame, labels = result["field"], result["fill"], result["frame"], result["labels"]
    coverage, fill_v1 = result["coverage"], result["fill_v1"]
    scene = result["scene"]
    edges = np.vstack((scene["faces"][:, [0, 1]], scene["faces"][:, [1, 2]]))
    edge_len = np.linalg.norm(scene["vertices"][edges[:, 0]] - scene["vertices"][edges[:, 1]], axis=1)
    stages = [{k: v for k, v in s.items() if not k.startswith("_")} for s in fill["stages"]]
    wound = field["wound_mask"] & field["valid"]
    gap = (result["target"]["target"] - field["height"])[wound]
    return {
        "sequence": SEQUENCE_ID,
        "units": "Syn3DWound dataset units (DU); 1 mDU = 0.001 DU; physical scale not certified",
        "data_route": "mesh-rendered geometry (aligned RGB-D route); not an RGB-only reconstruction",
        "target_status": "ILLUSTRATIVE_OPERATOR_ACCEPTED_TARGET_NOT_ANATOMICAL_TRUTH",
        "print_status": "PLAN_FOR_OPERATOR_REVIEW_PRINT_DISABLED",
        "views_used": len(scene["views"]),
        "display_view": f"image_{result['view']['file_stem']}.png",
        "mesh_vertices": int(len(scene["vertices"])),
        "mesh_faces": int(len(scene["faces"])),
        "mesh_median_edge_du": float(np.median(edge_len)),
        "wound_vertices": int(labels["wound"].sum()),
        "wound_vertex_min_views_seen": int(labels["seen"][labels["wound"]].min()),
        "wound_radius_du": frame["wound_radius"],
        "frame_normal_world": frame["normal"].tolist(),
        "height_field_pixel_du": field["pixel_du"],
        "height_field_shape": list(field["height"].shape),
        "wound_long_axis_du": field["boundary_width_px"] * field["pixel_du"],
        "wound_narrow_axis_du": field["narrow_width_px"] * field["pixel_du"],
        "wound_area_du2": float(field["wound_mask"].sum() * field["pixel_du"] ** 2),
        "initial_median_gap_du": float(np.median(gap)),
        "initial_max_gap_du": float(np.max(gap)),
        "healthy_ring_pixels": result["target"]["ring_pixels"],
        "target_model": "quadratic z = c0 + c1 x + c2 y + c3 x^2 + c4 xy + c5 y^2 fitted to the healthy ring",
        "target_coefficients": result["target"]["coefficients"].tolist(),
        "target_ring_rms_du": result["target"]["ring_rms_du"],
        "parameters": result["params"],
        "coverage_parameters": {k: list(v) if isinstance(v, tuple) else v
                                for k, v in coverage["coverage_parameters"].items()},
        "selected_planner": coverage["selected"],
        "selection_rule": coverage["selection_rule"],
        "planner_status": fill["summary"]["status"],
        "fill": fill["summary"],
        "fill_v1_superseded": {k: fill_v1["summary"][k] for k in (
            "hex_side_du", "lattice_cells_inside_supported_region", "lattice_area_fraction_of_wound", "stage_count",
            "stop_reason", "boundary_violations", "unsupported_cells")},
        "stages": stages,
    }


def coverage_audit(result: dict) -> dict:
    """Every number of the coverage correction (old-planner audit, planner search, staged fills)."""
    coverage, field = result["coverage"], result["field"]
    elig, search = coverage["eligibility"], coverage["search"]
    area = field["pixel_du"] ** 2
    room = elig["room_du"][elig["supported"]]
    best = {}
    for name, entry in search["best"].items():
        best[name] = {"config": entry["config"], "planned_cells": len(entry["cells"]),
                      "planned_eligible_coverage": entry["score"][0]}
    return {
        "question": ("Can the planner cover the eligible wound region adequately while keeping zero deposition outside "
                     "the wound, zero exclusion-zone violations, zero unsupported deposition and the unchanged "
                     "uncertainty gates?"),
        "units": "areas in mDU^2 = 1e-6 DU^2; lengths in DU unless stated; grid pixel = 0.1 mDU",
        "regions": {
            "wound_area_mdu2": float(elig["wound"].sum() * area * 1e6),
            "boundary_margin_area_mdu2": float(elig["margin_band"].sum() * area * 1e6),
            "boundary_margin_width_px": elig["margin_px"],
            "supported_area_mdu2": float(elig["supported"].sum() * area * 1e6),
            "eligible_safe_area_mdu2": float(elig["eligible"].sum() * area * 1e6),
            "maximum_possible_total_wound_coverage": float(elig["eligible"].sum() / elig["wound"].sum()),
            "supported_room_du": {"min": float(room.min()), "median": float(np.median(room)), "max": float(room.max())},
            "pixels_failing_uncertainty_gate": int((elig["wound"] & ~elig["uncertainty_ok"]).sum()),
            "pixels_without_surface_support": int((elig["wound"] & ~field["valid"]).sum()),
        },
        "v1_lattice_audit": coverage["audit_v1"],
        "planners": {name: res["summary"] for name, res in coverage["results"].items()},
        "search": {
            "sides_px": search["sides_px"], "v1_side_px": search["v1_side_px"], "min_side_px": search["min_side_px"],
            "rotations_deg": list(coverage["coverage_parameters"]["rotations_deg"]),
            "offsets": search["offsets"], "configurations_per_planner": len(search["records"]),
            "best": best, "adaptive_records": search["adaptive_records"],
            "diagnostic_below_printable_minimum": search["diagnostic_below_printable_minimum"],
        },
        "selected_planner": coverage["selected"],
        "selection_rule": coverage["selection_rule"],
        "binding_constraint_if_no_planner_passes": coverage["binding_constraint"],
        "whole_cell_limit": coverage["whole_cell_limit"],
        "gates_unchanged": ("boundary margin, surface support, uncertainty <= 0.5 layer and >= 1 layer of room below "
                            "the lower envelope are identical to the v1 planner (pipeline.patient._supported)"),
    }


def write_coverage_tables(audit: dict, search_records: list[dict], output: Path) -> None:
    with (output / "coverage_search.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(search_records[0]))
        writer.writeheader()
        writer.writerows(search_records)
    fields = ["planner", "status", "cells", "cells_by_kind", "eligible_coverage", "total_wound_coverage",
              "uncovered_eligible_area_mdu2", "stage_count", "active_cells_by_stage", "stop_reason",
              "violations_total", "max_overfill_vs_target_du", "final_eligible_area_still_one_layer_below_envelope_mdu2"]
    with (output / "coverage_planners.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for name, s in audit["planners"].items():
            writer.writerow([name, s["status"], s["cells"], json.dumps(s["cells_by_kind"]),
                             round(s["eligible_coverage"], 4), round(s["total_wound_coverage"], 4),
                             round(s["uncovered_eligible_area_du2"] * 1e6, 3), s["stage_count"],
                             json.dumps(s["active_cells_by_stage"]), s["stop_reason"], sum(s["violations"].values()),
                             s["max_overfill_vs_target_du"],
                             round(s["final_eligible_area_still_one_layer_below_envelope_du2"] * 1e6, 3)])


def write_toolpath(path: Path, fill: dict) -> int:
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "cell", "deposition_planned", "x_du", "y_du", "z_du", "nx", "ny", "nz"])
        for stage in fill["stages"]:
            tp = stage["_toolpath"]
            for point, normal, cell, flag in zip(tp["world"], tp["normals_world"], tp["cell"], tp["deposit_flag"]):
                writer.writerow([stage["stage"], int(cell), int(flag), *np.round(point, 9), *np.round(normal, 6)])
                rows += 1
    return rows


def make_figure(result: dict, summary: dict, output: Path) -> None:
    scene, view, field, fill = result["scene"], result["view"], result["field"], result["fill"]
    frame, local, target = result["frame"], result["local"], result["target"]["target"]
    camera = scene["cameras"][view["camera_id"]]
    rgb = cv2.cvtColor(cv2.imread(str(view["rgb_path"])), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(view["mask_path"]), cv2.IMREAD_GRAYSCALE)
    colours = vertex_colours(result, rgb)
    wound_uv, _ = project(scene["vertices"][result["labels"]["wound"]], view, camera)
    cu, cv_ = wound_uv.mean(axis=0)
    half = 32
    x0, y0 = int(cu - half), int(cv_ - half)
    crop = rgb[y0:y0 + 2 * half, x0:x0 + 2 * half]
    crop_bright = (mask[y0:y0 + 2 * half, x0:x0 + 2 * half] > result["params"]["mask_threshold"]).astype(np.uint8)
    _, components = cv2.connectedComponents(crop_bright)
    crop_mask = components == components[half, half] if components[half, half] else crop_bright.astype(bool)
    prisms = scaffold_prisms(fill, field, result["params"])
    extent = [field["axis"][0] * MDU, field["axis"][-1] * MDU, field["axis"][-1] * MDU, field["axis"][0] * MDU]
    exaggeration = 4.0

    fig = plt.figure(figsize=(16, 8.6))
    grid = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.18)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(rgb[::4, ::4], extent=[0, rgb.shape[1], rgb.shape[0], 0])
    ax.add_patch(Rectangle((x0 - 90, y0 - 90), 2 * half + 180, 2 * half + 180, fill=False, ec="yellow", lw=1.2))
    ax.set_title(f"1  OBSERVE: one of {summary['views_used']} photographs\n(rendered leg, {view['rgb_path'].name})",
                 fontsize=8, loc="left")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(crop, interpolation="bicubic")
    ax.contour(crop_mask, levels=[0.5], colors="yellow", linewidths=1.0)
    ax.set_title("2  IDENTIFY (2D): the dataset's wound mask\n(yellow outline; zoom into the yellow box)", fontsize=8, loc="left")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 2], projection="3d")
    tinted = colours.copy()
    tinted[result["labels"]["wound"]] = [0.85, 0.1, 0.1]
    mesh_axes(ax, local, scene["faces"], tinted, title=(
        f"2  IDENTIFY (3D): masks from {summary['views_used']} views voted\nonto the leg mesh "
        f"({summary['wound_vertices']} wound vertices, red)"))

    ax = fig.add_subplot(grid[0, 3])
    gap = np.where(field["wound_mask"], (target - field["height"]) * MDU, np.nan)
    image = ax.imshow(gap, extent=extent, cmap="magma_r", vmin=0)
    ax.contour(field["gx"] * MDU, field["gy"] * MDU, field["ring_mask"], levels=[0.5], colors="#2a9d8f", linewidths=0.8)
    ax.contour(field["gx"] * MDU, field["gy"] * MDU, field["wound_mask"], levels=[0.5], colors="black", linewidths=0.8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("depth below target (mDU)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("3-4  SURFACE + TARGET: how far the wound bed lies\nbelow the operator-accepted target (green = healthy ring)",
                 fontsize=8, loc="left")
    ax.set_xlabel("along wound (mDU)", fontsize=7)
    ax.set_ylabel("across wound (mDU)", fontsize=7)
    ax.tick_params(labelsize=6)

    ax = fig.add_subplot(grid[1, 0])
    plan = fill
    s = plan["summary"]
    draw_coverage_map(ax, field, plan, frame)
    if fill["stages"]:
        draw_toolpath_2d(ax, fill["stages"][0]["_toolpath"])
    ax.set_title(f"5  HONEYCOMB COVERAGE: {s['cells']} cells cover\n"
                 f"{s['eligible_coverage']:.1%} of safe area, {s['total_wound_coverage']:.1%} of wound;\n"
                 f"red = safe area left uncovered, pink = edge margin", fontsize=8, loc="left")
    ax.set_xlabel("along wound (mDU)", fontsize=7)
    ax.set_ylabel("across wound (mDU)", fontsize=7)
    ax.tick_params(labelsize=6)

    ax = fig.add_subplot(grid[1, 1])
    row = int(np.argmax(fill["in_lattice"].sum(axis=1))) if fill["in_lattice"].any() else int(round(field["centroid_px"][1]))
    x = field["axis"] * MDU
    wound_cols = np.nonzero(field["wound_mask"][row])[0]
    if wound_cols.size:
        ax.axvspan(x[wound_cols.min()], x[wound_cols.max()], color="#f8d7da", alpha=0.5, lw=0, label="wound")
    ax.plot(x, target[row] * MDU, "k--", lw=1.0, label="operator-accepted target")
    ax.plot(x, (target[row] - result["params"]["measurement_uncertainty_du"]) * MDU, ":", color="#e76f51", lw=1.0,
            label="lower envelope (target - uncertainty)")
    for stage in fill["stages"]:
        after = np.where(fill["in_lattice"][row], stage["_surface_after"][row], np.nan)
        ax.plot(x, after * MDU, color=STAGE_COLOURS[stage["stage"] - 1], lw=1.4,
                label=f"top after stage {stage['stage']} ({stage['active_cells']} cells)")
    ax.plot(x, field["height"][row] * MDU, color="#495057", lw=1.6, label="measured skin / wound bed (before)")
    ax.set_title(f"6  STAGED FILL: slice through the printed cells\nstop rule fired: {summary['fill']['stop_reason']}",
                 fontsize=8, loc="left")
    ax.set_xlabel("along wound (mDU)", fontsize=7)
    ax.set_ylabel("height above skin plane (mDU)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, loc="lower left")

    ax = fig.add_subplot(grid[1, 2], projection="3d")
    mesh_axes(ax, local, scene["faces"], colours, keep_radius=1.4 * frame["wound_radius"], exaggeration=exaggeration,
              title=f"7  PLACE (3D): honeycomb scaffold sitting in the\nwound on the leg (heights x{exaggeration:g} for visibility)")
    tops, walls, top_colours = prism_polys(prisms, exaggeration)
    ax.add_collection3d(Poly3DCollection(walls, facecolors="#1d3557", edgecolors="none", alpha=0.9))
    ax.add_collection3d(Poly3DCollection(tops, facecolors=top_colours, edgecolors="black", linewidths=0.3))
    for stage in fill["stages"]:
        tp = stage["_toolpath"]
        pts = tp["local"] * MDU * np.array([1, 1, exaggeration])
        for cell in np.unique(tp["cell"]):
            keep = (tp["cell"] == cell) & tp["deposit_flag"]
            ax.plot(pts[keep, 0], pts[keep, 1], pts[keep, 2], color="white", lw=0.35)

    ax = fig.add_subplot(grid[1, 3])
    ax.imshow(rgb, interpolation="bicubic", extent=[-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5])
    surface = np.where(field["valid"], field["height"], target)

    def projected(mask):
        out = []
        for contour in mask_contours(mask):
            xy = px_to_local(contour, field)
            rows, cols = contour[:, 1].astype(int), contour[:, 0].astype(int)
            uv, _ = project(to_world(np.column_stack((xy, surface[rows, cols])), frame), view, camera)
            out.append(uv)
        return out

    s = fill["summary"]
    uncovered = fill["eligible"] & ~fill["footprint"]
    for uv in projected(uncovered):
        ax.add_patch(Polygon(uv, closed=True, fc=CATEGORY_COLOURS[ELIGIBLE], ec="none", alpha=0.75))
    for prism, colour in zip(prisms, top_colours):
        local_top = np.column_stack((prism["outline_xy"], np.full(len(prism["outline_xy"]), prism["top"])))
        uv, _ = project(to_world(local_top, frame), view, camera)
        ax.add_patch(Polygon(uv, closed=True, fc=colour, ec="black", lw=0.3, alpha=0.85))
    for stage in fill["stages"][:1]:
        tp = stage["_toolpath"]
        uv, _ = project(tp["world"], view, camera)
        for cell in np.unique(tp["cell"]):
            keep = (tp["cell"] == cell) & tp["deposit_flag"]
            ax.plot(uv[keep, 0], uv[keep, 1], color="white", lw=0.25)
    for uv in projected(fill["eligible"]):
        ax.plot(*np.vstack((uv, uv[:1])).T, color="#00e5ff", lw=0.8, ls="--")
    for uv in projected(field["wound_mask"]):
        ax.plot(*np.vstack((uv, uv[:1])).T, color="yellow", lw=0.9)
    gate = s["coverage_gate"]
    wound_uv = np.vstack(projected(field["wound_mask"]))
    centre_uv = (wound_uv.min(axis=0) + wound_uv.max(axis=0)) / 2
    zoom = 0.85 * float(np.ptp(wound_uv, axis=0).max())
    max_total = fill["eligible"].sum() / max(field["wound_mask"].sum(), 1)
    ax.text(0.02, 0.02, f"STATUS: {s['status']} (coverage gate)\n"
            f"safe-area coverage {s['eligible_coverage']:.1%} (gate {gate:.0%})\n"
            f"total wound coverage {s['total_wound_coverage']:.1%} (max {max_total:.1%})\n"
            f"depth not finished: {s['final_eligible_area_still_one_layer_below_envelope_du2'] * 1e6:.1f} mDU² of safe\n"
            f"area still >= 1 layer below envelope\n"
            f"printing disabled", transform=ax.transAxes, fontsize=6, va="bottom", family="monospace",
            bbox={"fc": "white", "ec": "black", "lw": 0.5, "alpha": 0.9})
    handles = [Line2D([], [], color="yellow", lw=1.2, label="wound boundary"),
               Line2D([], [], color="#00e5ff", lw=1.0, ls="--", label="safe (eligible) region boundary")]
    for layers in sorted({p["layers"] for p in prisms}):
        handles.append(Patch(fc=STAGE_COLOURS[min(layers, len(STAGE_COLOURS)) - 1], ec="black", lw=0.3,
                             label=f"accepted honeycomb cell, {layers} layer{'s' if layers > 1 else ''}"))
    handles.append(Patch(fc=CATEGORY_COLOURS[ELIGIBLE], alpha=0.75, label="uncovered safe area"))
    ax.legend(handles=handles, fontsize=5.5, loc="upper right", framealpha=0.85)
    ax.set_xlim(centre_uv[0] - zoom, centre_uv[0] + zoom)
    ax.set_ylim(centre_uv[1] + zoom, centre_uv[1] - zoom)
    ax.set_title("OUTPUT: the planned scaffold projected back onto\nthe patient photograph (plan only; printing disabled)",
                 fontsize=8, loc="left", fontweight="bold")
    ax.axis("off")

    fig.suptitle("Full pipeline on a rendered patient leg (Syn3DWound, CC BY-NC-SA 4.0): photographs → wound → surface → "
                 "target → honeycomb → staged fill → scaffold on the body", fontsize=10, y=0.995)
    fig.savefig(output / "patient_pipeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "patient_pipeline.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_provenance(result: dict, output: Path) -> None:
    scene, view = result["scene"], result["view"]
    rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix() if Path(p).resolve().is_relative_to(ROOT) else str(p)  # noqa: E731
    inputs = {
        "mesh": {"path": rel(scene["mesh_path"]), "sha256": sha256_file(scene["mesh_path"])},
        "colmap_images_txt": {"path": rel(scene["images_txt"]), "sha256": sha256_file(scene["images_txt"])},
        "colmap_cameras_txt": {"path": rel(scene["cameras_txt"]), "sha256": sha256_file(scene["cameras_txt"])},
        "display_rgb": {"path": rel(view["rgb_path"]), "sha256": sha256_file(view["rgb_path"])},
        "masks": [{"path": rel(v["mask_path"]), "sha256": sha256_file(v["mask_path"])} for v in scene["views"]],
    }
    provenance = {
        **LICENCE,
        "source_identifier": SEQUENCE_ID,
        "inputs": inputs,
        "transformation_steps": [
            "Read cropped_mesh.obj and convert its y-up axes to the COLMAP world frame: (x, y, z) -> (x, -z, y).",
            "Project every mesh vertex into every view with the COLMAP pose and pinhole intrinsics.",
            "Label a vertex as wound when, among views where it faces the camera, more than half of the masks are > 40/255 "
            "at its pixel; keep the largest connected wound patch.",
            "Build a tangent frame by PCA of the skin within 2.5 wound radii; align e1 with the wound's long axis.",
            "Resample the mesh as a 0.1 mDU height field by linear interpolation.",
            "Fit a quadratic surface to the healthy ring (band up to 0.4 wound radii outside the wound) and use it as the "
            "illustrative operator-accepted target.",
            "Tile the wound with the modules.honeycomb lattice and run the previous whole-cell staged fill (kept as the "
            "audited baseline V1_LATTICE).",
            "Classify every height-field pixel with the unchanged safety rules (pipeline.coverage.eligibility): edge "
            "margin, surface support, uncertainty gate and at least one layer of room below the lower envelope.",
            "Search hexagon side (5 sizes from the previous side down to two strand widths), rotation (0-50 deg in 10 "
            "deg steps) and 16 staggered offsets for three planners: whole cells, adaptive whole cells and whole cells "
            "plus boundary cells clipped to the eligible region (clipped piece >= 1/4 hexagon and >= 2 strand widths "
            "wide).",
            "Stage the fill one whole layer per accepted cell below the lower envelope; re-measure and repeat until no "
            "cell has room; count every boundary, support, envelope and self-intersection violation.",
            "Select the first planner (whole -> adaptive -> clipped) with zero violations and >= 90% eligible-area "
            "coverage; otherwise report the best coverage as PARTIAL.",
            "Map the scaffold and toolpath back to world coordinates and project them into the display photograph.",
            "Vertex colours for the 3D panels are sampled from the display photograph; no numbers are taken from RGB pixels.",
        ],
        "output_paths": [rel(output / name) for name in (
            "patient_pipeline.png", "patient_pipeline.pdf", "coverage_audit.png", "coverage_audit.pdf",
            "coverage_audit.json", "coverage_search.csv", "coverage_planners.csv", "summary.json",
            "scaffold_toolpath_world.csv",
            "provenance.json", "caption.md")],
        "raw_data_committed": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def write_caption(summary: dict, audit: dict, output: Path) -> None:
    fill, v1 = summary["fill"], summary["fill_v1_superseded"]
    regions = audit["regions"]
    stages = ", ".join(f"stage {s['stage']}: {s['active_cells']} cells" for s in summary["stages"])
    kinds = ", ".join(f"{n} {k}" for k, n in fill["cells_by_kind"].items())
    text = f"""# Patient-level full-pipeline figure: caption draft

**Full pipeline on a rendered patient leg.** (1) One of {summary['views_used']} rendered photographs of a
leg from the public Syn3DWound dataset ({summary['sequence']}). (2, left) The dataset's wound mask for the
same view, outlined in yellow. (2, right) The {summary['views_used']} 2D masks are transferred to the 3D leg
mesh by projecting every vertex into every photograph and taking a majority vote; {summary['wound_vertices']}
vertices are labelled as wound (red). (3-4) The mesh is resampled as a height map in a local skin frame, and
a quadratic surface fitted to the surrounding healthy ring serves as an illustrative operator-accepted
target (ring RMS residual {summary['target_ring_rms_du'] * 1e3:.2f} mDU). The colour shows how far the wound bed lies below that
target (median {summary['initial_median_gap_du'] * 1e3:.2f} mDU, maximum {summary['initial_max_gap_du'] * 1e3:.2f} mDU). (5) Coverage map. Of the
{regions['wound_area_mdu2']:.2f} mDU² wound, {regions['boundary_margin_area_mdu2']:.2f} mDU² is an excluded edge margin (pink) and
{regions['eligible_safe_area_mdu2']:.2f} mDU² is safe to print on (eligible). The selected planner
({summary['selected_planner']}) uses {fill['cells']} cells ({kinds}) and covers {fill['eligible_coverage'] * 100:.1f}% of the safe area
(teal) and {fill['total_wound_coverage'] * 100:.1f}% of the whole wound (at most {regions['maximum_possible_total_wound_coverage'] * 100:.1f}% is
possible because of the margin); red marks safe area left uncovered. The previous planner used one fixed
lattice of whole hexagons and kept only {v1['lattice_cells_inside_supported_region']} cells in a central strip
({audit['planners']['V1_LATTICE']['eligible_coverage'] * 100:.1f}% of the safe area). (6) Each stage prints one whole layer
({summary['parameters']['layer_thickness_du'] * 1e3:.2f} mDU) in every cell that still has at least one layer of room below the lower
uncertainty envelope, then re-measures ({stages}). The run stops with {fill['stop_reason']}:
{STOP_EXPLANATIONS.get(fill['stop_reason'], 'see summary.json')}. {fill['final_eligible_area_still_one_layer_below_envelope_du2'] * 1e6:.2f} mDU² of safe area still lies at least
one layer below the envelope, so the cavity is not filled to the target everywhere. Violations: {sum(fill['violations'].values())}
(outside wound, edge margin, unsupported, above envelope, path self-intersections). (7) The resulting scaffold
on the 3D leg and (OUTPUT) projected back onto the photograph with the wound boundary (yellow), the safe-region
boundary (cyan, dashed), the accepted honeycomb and the uncovered safe area (red). Status: {fill['status']}
(coverage gate {fill['coverage_gate'] * 100:.0f}% of the safe area, zero violations). The target is not anatomical truth, the
geometry is mesh-rendered (aligned RGB-D route), units are dataset units with no certified physical scale,
and printing is disabled. Data: Syn3DWound, CC BY-NC-SA 4.0, doi:10.25919/5rwz-ts17.
"""
    (output / "caption.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run_patient(args.scene)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(result)
    summary["toolpath_rows"] = write_toolpath(args.output / "scaffold_toolpath_world.csv", result["fill"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit = coverage_audit(result)
    (args.output / "coverage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_coverage_tables(audit, result["coverage"]["search"]["records"], args.output)
    make_figure(result, summary, args.output)
    make_coverage_figure(result, audit, args.output)
    write_provenance(result, args.output)
    write_caption(summary, audit, args.output)
    fill = summary["fill"]
    print(f"{summary['wound_vertices']} wound vertices; planner={summary['selected_planner']}; {fill['cells']} cells; "
          f"safe-area coverage={fill['eligible_coverage']:.3f}; wound coverage={fill['total_wound_coverage']:.3f}; "
          f"{fill['stage_count']} stages; stop={fill['stop_reason']}; violations={sum(fill['violations'].values())}; "
          f"status={fill['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
