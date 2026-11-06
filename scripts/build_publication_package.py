"""Build publication figures, evidence tables, and an HTML result gallery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
FIGURES = ROOT / "paper" / "figures"
FIGURE_DATA = ROOT / "paper" / "figure_data"
TABLES = ROOT / "paper" / "tables"
EVIDENCE_PATH = DATA / "publication_evidence.json"
PALETTE = {
    "ink": "#17324D",
    "blue": "#3478A8",
    "teal": "#36A89A",
    "gold": "#D99B2B",
    "coral": "#D76A5B",
    "gray": "#6B7785",
    "light": "#E9EFF4",
    "green": "#518A67",
    "red": "#B54A4A",
}

FIGURE_CAPTIONS = {
    "figure_01": (
        "Evidence-bounded computational architecture. Public RGB and mesh-rendered depth "
        "feed observable geometry and quality gates; failed or unsupported cases are refused. "
        "Reference-dependent fill requires an explicit target policy. The final output is an "
        "offline surface-conformal plan; neither robot execution nor printing is authorized."
    ),
    "figure_02": (
        "Evidence stack and dataset roles. Syn3DWound supplies synthetic RGB/mesh controls; "
        "SALVE is used for shape comparison after ground-truth similarity alignment; procedural "
        "fixtures support offline routing and staged-plan demonstrations. None establishes "
        "independent metric scale or clinical validity."
    ),
    "figure_03": (
        "Representative Syn3DWound reconstruction evidence: public rendered RGB, frozen reviewed "
        "mask, recovered wound-bed mesh, source-mesh reference geometry, and the saved error "
        "heatmap. Coordinates are dataset units. The source mesh is a synthetic reference, not "
        "an anatomical pre-wound surface."
    ),
    "figure_04": (
        "Focused route comparison over four Syn3DWound sequence contexts. Surface errors are "
        "percent of wound width; reference-volume errors are relative to the declared synthetic "
        "source reference. The 60-view RGB-only route remains unsupported, and mesh-rendered "
        "depth results are not physical sensor validation."
    ),
    "figure_05": (
        "Misalignment sensitivity. Small cross-view surface repeatability does not imply reliable "
        "reference-defined volume: route-level volume error rises under alignment offsets and "
        "reaches 80.5% in the tested misalignment route. Results are descriptive repeated "
        "synthetic subsets, not independent samples."
    ),
    "figure_06": (
        "Synthetic calibration-safety evidence. Strict-holdout recall is shown for total and "
        "matched-clean incremental volume consequences; clean false alarms are 2/48. In the "
        "separate target ablation, stepped-target fault rejection exceeds planar-target rejection, "
        "with zero false alarms in 24 clean controls for each. The engineering gate remains failed."
    ),
    "figure_07": (
        "Counterfactual healthy-skin closure ambiguity. Six ring-only reference models yield "
        "different reference-defined volumes and extrapolation ranges on one sequence. These "
        "are model-dependent planning quantities, not true wound or anatomical restoration volume."
    ),
    "figure_08": (
        "Held-out healthy-disc exemplar from the sealed v3 test split. The visible ring supports "
        "ring-only TPS prediction; held-out truth is used only for evaluation. The calibrated "
        "ensemble envelope is shown with its test coverage context (88.3% pointwise versus a "
        "90% target). This is a lower-bound healthy-surface test, not recovery of pre-wound anatomy."
    ),
    "figure_09": (
        "Three-stage offline plan/update simulation on a synthetic cavity. Panels show the "
        "initial cavity and interpolated stage surfaces leading to the explicitly defined target; "
        "the observable stop rule is reached after three stages. Command records have deposition "
        "disabled; no CoppeliaSim, robot, physical deposition, or printing is represented."
    ),
}


def load_evidence() -> dict[str, Any]:
    if not EVIDENCE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {EVIDENCE_PATH.relative_to(ROOT)}; run scripts/export_publication_evidence.py "
            "with the sealed local result bundle first."
        )
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": "#647483",
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#DDE5EB",
            "grid.linewidth": 0.7,
        }
    )


def _save_figure(name: str, fig: plt.Figure, data: Any, evidence: dict[str, Any], sources: list[str]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    save_json(FIGURE_DATA / f"{name}.json", data)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    svg_path = FIGURES / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    with svg_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(line.rstrip() for line in svg_lines) + "\n")
    plt.close(fig)
    source_hashes = {
        "paper/data/publication_evidence.json": hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest(),
    }
    source_hashes.update({
        row["path"]: row["sha256"]
        for row in evidence["source_manifest"]
        if row["path"] in sources
    })
    if name == "figure_03":
        reconstruction_provenance = json.loads(
            (DATA / "figure03_provenance.json").read_text(encoding="utf-8")
        )
        for field in (
            "input_image",
            "reviewed_mask_source",
            "recovered_mesh_source",
            "reference_mesh_source",
            "error_map_source",
        ):
            source_path = reconstruction_provenance.get(field)
            source_hash = reconstruction_provenance.get(f"{field}_sha256")
            if source_path and source_hash:
                source_hashes[source_path] = source_hash

    save_json(
        FIGURES / f"{name}.provenance.json",
        {
            "figure": name,
            "caption": FIGURE_CAPTIONS[name],
            "png_dpi": 300,
            "vector_export": "SVG",
            "data_origin": "Sealed result summaries and identified local public-data inputs.",
            "source_paths": sources,
            "source_hashes": source_hashes,
            "limitations": [
                "Descriptive evidence only; correlated views/subsets are not independent subjects.",
                "Synthetic mesh-rendered depth is not a physical sensor measurement.",
                "No physical printing is authorized.",
            ],
        },
    )


def _draw_architecture() -> tuple[plt.Figure, dict[str, Any]]:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    nodes = [
        (0.25, 4.9, 2.15, 1.15, "Inputs", "RGB • reviewed boundary\nmesh-rendered depth"),
        (3.0, 4.9, 2.2, 1.15, "Observable geometry", "supported bed • boundary\nnormals • confidence"),
        (5.85, 4.9, 2.15, 1.15, "Quality gates", "resolution • coverage\nalignment • uncertainty"),
        (8.75, 4.9, 2.55, 1.15, "Explicit target policy", "operator / reference\npartial fill • safety envelope"),
        (8.75, 2.55, 2.55, 1.15, "Offline layer plan", "surface-conformal geometry\nno scalar-volume prerequisite"),
        (5.85, 2.55, 2.15, 1.15, "Rescan / update", "synthetic geometry update\nobservable stopping rule"),
        (0.9, 0.75, 2.7, 1.05, "Refuse / request input", "unsupported geometry\nambiguous target / scale"),
        (8.75, 0.75, 2.55, 1.05, "Physical boundary", "no robot execution\nprinting disabled"),
    ]
    for x, y, w, h, title, body in nodes:
        color = PALETTE["blue"] if title in {"Inputs", "Observable geometry"} else PALETTE["teal"]
        if title in {"Refuse / request input", "Physical boundary"}:
            color = PALETTE["coral"]
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.07,rounding_size=0.08",
            linewidth=1.1, edgecolor=color, facecolor=color + "18",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.31, body, ha="center", va="center", fontsize=8)

    arrows = [
        ((2.4, 5.48), (3.0, 5.48)),
        ((5.2, 5.48), (5.85, 5.48)),
        ((8.0, 5.48), (8.75, 5.48)),
        ((10.02, 4.9), (10.02, 3.7)),
        ((8.75, 3.12), (8.0, 3.12)),
        ((5.85, 3.12), (4.0, 1.8)),
        ((6.9, 4.9), (2.8, 1.8)),
        ((10.02, 2.55), (10.02, 1.8)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=12,
                linewidth=1.2, color=PALETTE["gray"],
                connectionstyle="arc3,rad=0.0",
            )
        )
    ax.text(6, 6.55, "Evidence-bounded computational architecture", ha="center", weight="bold", fontsize=13)
    return fig, {"nodes": nodes, "edges": arrows}


def _draw_evidence_stack(evidence: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(10.7, 5.8))
    ax.axis("off")
    rows = [
        ("PUBLIC SYNTHETIC GEOMETRY", "Syn3DWound", "RGB + cropped meshes + COLMAP; mesh-rendered depth; no certified scale", PALETTE["blue"]),
        ("PHANTOM SHAPE BENCHMARK", "SALVE", "50-frame PIS4 Accuracy split; published reconstructions; GT similarity alignment", PALETTE["teal"]),
        ("FROZEN SAFETY EVIDENCE", "Calibration + hidden discs", "Synthetic target holdouts; one held-out body model; descriptive, correlated records", PALETTE["gold"]),
        ("OFFLINE SOFTWARE DEMO", "Procedural synthetic cases", "Five routing cases + three plan/update stages; command authority disabled", PALETTE["coral"]),
    ]
    for i, (kicker, name, detail, color) in enumerate(rows):
        y = 4.85 - i * 1.12
        ax.add_patch(plt.Rectangle((0.3, y), 2.6, 0.72, color=color, alpha=0.14, ec=color, lw=1.2))
        ax.text(0.45, y + 0.46, kicker, fontsize=8, weight="bold", color=color)
        ax.text(0.45, y + 0.19, name, fontsize=10, weight="bold")
        ax.text(3.2, y + 0.38, detail, fontsize=9, va="center")
        ax.annotate("", (2.85, y + 0.36), (3.08, y + 0.36), arrowprops={"arrowstyle": "->", "color": PALETTE["gray"]})
    ax.text(0.35, 0.28, "Evidence output: observable geometry → route/refusal → explicit planning target → offline plan", fontsize=9, color=PALETTE["ink"])
    ax.set_xlim(0, 10.7)
    ax.set_ylim(0, 5.7)
    return fig, {"evidence_layers": rows}


def _draw_reconstruction(evidence: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any], list[str]]:
    provenance = json.loads((DATA / "figure03_provenance.json").read_text(encoding="utf-8"))
    rgb_path = ROOT / provenance["input_image"]
    if not rgb_path.is_file():
        raise FileNotFoundError(
            f"Figure 3 RGB input is missing: {rgb_path}. "
            "Run `python scripts/fetch_figure3_example.py`."
        )
    actual_sha256 = hashlib.sha256(rgb_path.read_bytes()).hexdigest()
    expected_sha256 = provenance["input_image_sha256"]
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Figure 3 RGB input has an unexpected SHA-256: {rgb_path} "
            f"(expected {expected_sha256}, got {actual_sha256})."
        )
    with Image.open(rgb_path) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((900, 600))
        rgb_array = np.asarray(rgb)
    mask = np.asarray(Image.open(DATA / "figure03_reviewed_mask.png").convert("L"))
    error_map = np.asarray(Image.open(DATA / "figure03_error_heatmap.png").convert("RGB"))
    surfaces = np.load(DATA / "figure03_surface_data.npz")
    recovered_v, recovered_f = surfaces["recovered_vertices"], surfaces["recovered_faces"]
    reference_v, reference_f = surfaces["reference_vertices"], surfaces["reference_faces"]

    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(rgb_array)
    ax1.set_title("Rendered RGB input")
    ax1.axis("off")
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(mask, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("Frozen reviewed wound mask")
    ax2.axis("off")
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    step = max(1, len(recovered_v) // 5000)
    ax3.plot_trisurf(
        recovered_v[:, 0], recovered_v[:, 1], recovered_v[:, 2],
        triangles=recovered_f[::max(1, len(recovered_f) // 12000)],
        color=PALETTE["teal"], linewidth=0.05, alpha=0.9,
    )
    ax3.set_title("Recovered wound-bed mesh")
    ax3.set_xlabel("x (dataset units)")
    ax3.set_ylabel("y (dataset units)")
    ax3.set_zlabel("z (dataset units)")
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    if len(reference_f):
        ax4.plot_trisurf(
            reference_v[:, 0], reference_v[:, 1], reference_v[:, 2],
            triangles=reference_f[::max(1, len(reference_f) // 12000)],
            color=PALETTE["blue"], linewidth=0.05, alpha=0.85,
        )
    else:
        ax4.scatter(
            reference_v[:, 0], reference_v[:, 1], reference_v[:, 2],
            s=1.5, alpha=0.55, color=PALETTE["blue"],
        )
    ax4.set_title("Source-mesh reference geometry")
    ax4.set_xlabel("x (dataset units)")
    ax4.set_ylabel("y (dataset units)")
    ax4.set_zlabel("z (dataset units)")
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(error_map)
    ax5.set_title("Saved signed-error heatmap")
    ax5.axis("off")
    ax6 = fig.add_subplot(2, 3, 6)
    route = next(row for row in evidence["focused_routes"] if row["route"] == "HIGH_RES_RGB_PLUS_DEPTH")
    ax6.axis("off")
    ax6.text(0.02, 0.88, "HIGH_RES_RGB_PLUS_DEPTH_A route evidence (aggregate)", weight="bold", fontsize=8.5)
    ax6.text(
        0.02, 0.68,
        f"Mean surface error: {route['mean_surface_error_percent_wound_width']:.3f}% of wound width\n"
        f"Completeness @ 5% width: {100*route['completeness_5pct_width_mean_available_only']:.1f}%\n"
        f"Sequences/subsets: {route['sequences']} / {route['subsets']}\n"
        "Depth source: synthetic mesh render\nPhysical scale: not established",
        va="top", fontsize=9, linespacing=1.6,
    )
    fig.suptitle("Representative synthetic RGB reconstruction and reference comparison", weight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    sources = [
        "sensor_generalization/analysis/summary.json",
        "sensor_generalization/runs/sacrum_phi/HIGH_RES_RGB_PLUS_DEPTH_A/result.json",
    ]
    data = {
        "input_image": provenance["input_image"],
        "source_image_sha256": provenance["input_image_sha256"],
        "reviewed_mask_sha256": provenance["reviewed_mask_sha256"],
        "recovered_mesh_sha256": provenance["recovered_mesh_sha256"],
        "reference_mesh_sha256": provenance["reference_mesh_sha256"],
        "error_map_sha256": provenance["error_map_sha256"],
        "route_summary": route,
        "coordinate_units": "dataset units; not certified physical units",
    }
    return fig, data, sources


def _draw_routes(evidence: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    rows = evidence["focused_routes"]
    labels = [
        "RGB-only\n60 views",
        "Aligned\nRGB-D",
        "High-res RGB\n+ depth",
        "Low depth\ncoverage",
        "High depth\nnoise",
        "Misaligned\ndepth",
    ]
    keys = [
        "RGB_ONLY_60_VIEW",
        "GENERIC_ALIGNED_RGBD",
        "HIGH_RES_RGB_PLUS_DEPTH",
        "LOW_DEPTH_COVERAGE",
        "HIGH_DEPTH_NOISE",
        "MISALIGNED_DEPTH",
    ]
    ordered = [next(row for row in rows if row["route"] == key) for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    x = np.arange(len(ordered))
    surface = np.array([
        row["mean_surface_error_percent_wound_width"]
        if row["mean_surface_error_percent_wound_width"] is not None else np.nan
        for row in ordered
    ])
    volume = np.array([
        row["mean_reference_volume_error_percent"]
        if row["mean_reference_volume_error_percent"] is not None else np.nan
        for row in ordered
    ])
    for ax, values, ylabel, color in [
        (axes[0], surface, "Mean surface error (% wound width)", PALETTE["blue"]),
        (axes[1], volume, "Mean reference-volume error (%)", PALETTE["coral"]),
    ]:
        bars = ax.bar(x, np.nan_to_num(values, nan=0.0), color=color, alpha=0.88)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        for i, (bar, val, row) in enumerate(zip(bars, values, ordered)):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=7)
            else:
                ax.text(i, 0.03, "unavailable", ha="center", va="bottom", rotation=90, fontsize=7, color=PALETTE["gray"])
    axes[0].set_title("Visible-surface metric")
    axes[1].set_title("Reference-dependent metric")
    fig.suptitle("Sealed sensor-route comparison (4 sequence contexts)", weight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, {"routes": ordered, "route_labels": labels, "availability_retained": True}


def _draw_misalignment(evidence: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    rows = evidence["alignment_sensitivity_routes"]
    names = {
        "GENERIC_ALIGNED_RGBD", "OFFSET_0px", "OFFSET_1px",
        "OFFSET_2px", "OFFSET_4px", "OFFSET_8px", "MISALIGNED_DEPTH",
    }
    selected = [row for row in rows if row["route"] in names]
    selected.sort(
        key=lambda row: (
            0 if row["route"] in {"GENERIC_ALIGNED_RGBD", "MISALIGNED_DEPTH"} else 1,
            int(re.search(r"(\d+)", row["route"]).group(1)) if re.search(r"(\d+)", row["route"]) else 0,
        )
    )
    x = np.array([row["repeatability_percent_wound_width"] for row in selected], dtype=float)
    y = np.array([row["mean_reference_volume_error_percent"] for row in selected], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for row, xx, yy in zip(selected, x, y):
        color = PALETTE["coral"] if row["route"] == "MISALIGNED_DEPTH" else PALETTE["blue"]
        ax.scatter(xx, yy, s=56, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        label = "misaligned route" if row["route"] == "MISALIGNED_DEPTH" else row["route"].replace("OFFSET_", "")
        ax.annotate(label, (xx, yy), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Cross-view repeatability (% wound width)")
    ax.set_ylabel("Mean reference-volume error (%)")
    ax.set_title("Repeatable surface estimates can still be wrong")
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, {
        "points": [
            {
                "route": row["route"],
                "repeatability_percent_wound_width": row["repeatability_percent_wound_width"],
                "mean_reference_volume_error_percent": row["mean_reference_volume_error_percent"],
                "worst_reference_volume_error_percent": (
                    100 * row["volume_relative_error_worst_available"]
                    if row.get("volume_relative_error_worst_available") is not None
                    else None
                ),
                "sequences": row["sequences"],
                "subsets": row["subsets"],
            }
            for row in selected
        ]
    }


def _draw_calibration(evidence: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    cal = evidence["calibration"]
    recall = cal["strict_holdout"]["threshold_recall"]
    bases = ["Strict holdout, total error", "Strict holdout, incremental error"]
    thresholds = ["5%", "10%", "20%"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for basis, color in zip(bases, [PALETTE["blue"], PALETTE["teal"]]):
        values = [next(r for r in recall if r["evaluation"] == basis and r["threshold"] == t) for t in thresholds]
        axes[0].plot(
            [5, 10, 20], [100 * row["rate"] for row in values],
            marker="o", linewidth=2, color=color, label=basis.replace("Strict holdout, ", ""),
        )
        for xx, row in zip([5, 10, 20], values):
            axes[0].annotate(
                f"{row['numerator']}/{row['denominator']}",
                (xx, 100 * row["rate"]), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=7,
            )
    axes[0].axhline(95, color=PALETTE["gray"], linestyle="--", linewidth=1)
    axes[0].set_xticks([5, 10, 20], ["5", "10", "20"])
    axes[0].set_xlabel("Reference-volume consequence threshold (%)")
    axes[0].set_ylabel("Configuration-level recall (%)")
    axes[0].set_ylim(0, 110)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Strict holdout")
    axes[0].grid(axis="y")

    strict = cal["strict_holdout"]
    false_alarm = 100 * strict["clean_false_alarm_count"] / strict["clean_false_alarm_denominator"]
    axes[1].bar(
        ["Clean false alarms"],
        [false_alarm],
        color=PALETTE["gold"],
        width=0.5,
    )
    axes[1].set_ylabel("False-alarm rate (%)")
    axes[1].set_title(f"Clean controls: {strict['clean_false_alarm_count']}/{strict['clean_false_alarm_denominator']}")
    axes[1].set_ylim(0, max(10, false_alarm + 3))
    axes[1].text(0, false_alarm + 0.6, f"{strict['clean_false_alarm_count']}/{strict['clean_false_alarm_denominator']}", ha="center", fontsize=8)
    axes[1].grid(axis="y")

    target_rows = cal["target_ablation"]["stepped_vs_planar"]
    ax = axes[2]
    xx = np.arange(2)
    rates = [100 * row["fault_rejection_rate"] for row in target_rows]
    bars = ax.bar(xx, rates, color=[PALETTE["teal"], PALETTE["gray"]], width=0.58)
    ax.set_xticks(xx, ["Stepped target", "Planar target"])
    ax.set_ylabel("Fault rejection (%)")
    ax.set_title("Target check ablation")
    ax.set_ylim(0, 110)
    ax.grid(axis="y")
    for bar, row, val in zip(bars, target_rows, rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f"{row['fault_rejections']}/{row['fault_configurations']}", ha="center", fontsize=8)
    fig.suptitle("Synthetic calibration safety; engineering gate remains FAIL", weight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, cal


def _draw_reference_ambiguity(evidence: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    rows = evidence["reference_surface_models"]
    if not rows:
        raise ValueError("No frozen ring-only reference-model outputs are available.")
    names = [row["method"] for row in rows]
    volumes = np.array([row["reference_defined_volume_dataset_units3"] for row in rows])
    ranges = np.array([row["extrapolation_z_range_dataset_units"] for row in rows])
    median = float(np.median(volumes))
    normalized = volumes / median if median else volumes
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = [PALETTE["coral"] if x > 2 or x < 0.5 else PALETTE["blue"] for x in normalized]
    axes[0].bar(np.arange(len(rows)), normalized, color=colors)
    axes[0].axhline(1, color=PALETTE["gray"], linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(len(rows)), names, rotation=30, ha="right")
    axes[0].set_ylabel("Model-defined volume / model median")
    axes[0].set_title("Reference-defined volume spread")
    axes[0].grid(axis="y")
    axes[1].bar(np.arange(len(rows)), ranges, color=PALETTE["teal"])
    axes[1].set_xticks(np.arange(len(rows)), names, rotation=30, ha="right")
    axes[1].set_ylabel("Extrapolated Z range (dataset units)")
    axes[1].set_title("Closure-surface extrapolation range")
    axes[1].grid(axis="y")
    fig.suptitle("Healthy-ring models are conventions, not anatomical ground truth", weight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, {
        "models": rows,
        "volume_normalization": "Each model's reference-defined volume divided by the six-model median.",
        "interpretation": "No official pre-wound healthy-surface truth is available.",
    }


def _tricontour(ax: plt.Axes, xy: np.ndarray, z: np.ndarray, title: str, cmap: str = "viridis"):
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
    image = ax.tricontourf(triang, z, levels=18, cmap=cmap)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel("Local x (dataset units)")
    ax.set_ylabel("Local y (dataset units)")
    return image


def _draw_hidden_disc(evidence: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    sample = json.loads((DATA / "figure08_hidden_disc_example.json").read_text(encoding="utf-8"))
    xy = np.asarray(sample["query_xy"], dtype=float)
    ring_xy = np.asarray(sample["ring_xy"], dtype=float)
    truth = np.asarray(sample["heldout_truth_z"], dtype=float)
    tps = np.asarray(sample["tps_prediction_z"], dtype=float)
    center = np.asarray(sample["ensemble_center_z"], dtype=float)
    low = np.asarray(sample["envelope_low_z"], dtype=float)
    high = np.asarray(sample["envelope_high_z"], dtype=float)
    error = np.abs(tps - truth)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.3))
    axes[0, 0].scatter(ring_xy[:, 0], ring_xy[:, 1], s=10, color=PALETTE["teal"])
    axes[0, 0].scatter(xy[:, 0], xy[:, 1], s=4, color=PALETTE["gray"], alpha=0.45)
    axes[0, 0].set_title("Visible ring and held-out query support")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_xlabel("Local x (dataset units)")
    axes[0, 0].set_ylabel("Local y (dataset units)")
    im1 = _tricontour(axes[0, 1], xy, tps, "TPS ring-only prediction")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.75, label="Height (dataset units)")
    im2 = _tricontour(axes[0, 2], xy, truth, "Held-out healthy surface")
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.75, label="Height (dataset units)")
    im3 = _tricontour(axes[1, 0], xy, error, "Absolute TPS error", cmap="magma")
    fig.colorbar(im3, ax=axes[1, 0], shrink=0.75, label="Error (dataset units)")
    order = np.argsort(np.arctan2(xy[:, 1] - np.mean(xy[:, 1]), xy[:, 0] - np.mean(xy[:, 0])))
    axis = np.arange(len(order))
    axes[1, 1].fill_between(axis, low[order], high[order], color=PALETTE["teal"], alpha=0.25, label="Frozen ensemble envelope")
    axes[1, 1].plot(axis, truth[order], color=PALETTE["ink"], linewidth=1.2, label="Held-out truth")
    axes[1, 1].plot(axis, center[order], color=PALETTE["coral"], linewidth=0.9, label="Ensemble center")
    axes[1, 1].set_title("Calibrated ensemble envelope")
    axes[1, 1].set_xlabel("Query points ordered by angle (not independent)")
    axes[1, 1].set_ylabel("Height (dataset units)")
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 2].axis("off")
    h = evidence["healthy_disc"]
    axes[1, 2].text(
        0.03, 0.95,
        "Sealed TEST context\n"
        f"Patch: {sample['patch_id']}\n"
        f"Ring width: {sample['ring_width_fraction']:.0%} of hole diameter\n"
        f"Valid TPS records: {h['valid_global_method_records']}/{h['test_patch_ring_records']}\n"
        f"Global MAE: {100*h['macro_mae_fraction_of_diameter']:.4f}% of diameter\n"
        f"90% envelope coverage: {100*h['uncertainty_pointwise_coverage']:.2f}% pointwise\n"
        f"Body-model holdout: {', '.join(h['test_body_models'])}\n"
        "Healthy discs are only a lower-bound test.",
        va="top", fontsize=9, linespacing=1.65,
    )
    fig.suptitle("Healthy-skin hidden-disc extrapolation (v3 sealed TEST)", weight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, {
        "sample": sample,
        "aggregate": h,
        "pointwise_coverage_measured": h["uncertainty_pointwise_coverage"],
        "target_coverage": h["uncertainty_target_coverage"],
    }


def _draw_staged_fill(evidence: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    geometry = np.load(DATA / "figure09_staged_geometry.npz")
    initial = geometry["vertices_initial"]
    target = geometry["target_surface"]
    faces = geometry["faces"]
    start_z = initial[:, 2]
    if target.ndim == 2:
        target_z = target[:, 2]
    else:
        target_z = target
    if len(target_z) != len(start_z):
        raise ValueError("The offline staged-fill target does not align with the observed mesh.")
    stage_count = len(evidence["offline_demo"]["trace_stage_history"])
    fig = plt.figure(figsize=(14, 4.8))
    global_z_limits = (
        float(np.min(np.minimum(start_z, target_z))),
        float(np.max(np.maximum(start_z, target_z))),
    )
    color_norm = plt.Normalize(*global_z_limits)
    for index in range(stage_count + 1):
        fraction = index / stage_count
        z = start_z + fraction * (target_z - start_z)
        ax = fig.add_subplot(1, stage_count + 1, index + 1, projection="3d")
        triangles = faces[::max(1, len(faces) // 12000)]
        ax.plot_trisurf(
            initial[:, 0], initial[:, 1], z, triangles=triangles,
            cmap="viridis", norm=color_norm, linewidth=0.05, antialiased=True,
        )
        ax.set_title("Initial cavity" if index == 0 else f"After stage {index}")
        ax.set_xlabel("x (scene units)", fontsize=8, labelpad=0)
        ax.set_ylabel("y (scene units)", fontsize=8, labelpad=0)
        ax.set_zlabel("z (scene units)" if index == stage_count else "", fontsize=8, labelpad=0)
        ax.set_zlim(*global_z_limits)
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="z", labelsize=7)
        if index < stage_count:
            ax.set_zticklabels([])
        ax.view_init(elev=27, azim=-55)
    fig.suptitle("Three-stage offline scan–plan–update; commands disabled", weight="bold", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.94, bottom=0.05, top=0.88, wspace=0.2)
    return fig, {
        "stage_history": evidence["offline_demo"]["trace_stage_history"],
        "stopping_decision": evidence["offline_demo"]["closed_loop"],
        "target_surface": "SYNTHETIC_EXTERNAL_TEST_PLANE",
        "target_is_anatomical_truth": False,
        "physical_robot_commands_enabled": evidence["offline_demo"]["physical_robot_commands_enabled"],
        "print_authorized": evidence["offline_demo"]["print_authorized"],
        "surface_display": "Linear interpolation between recorded initial mesh and explicitly defined synthetic target for visualization.",
    }


def _escape_latex(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field for field, _ in columns]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def _write_table(name: str, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    _write_csv(TABLES / f"{name}.csv", rows, columns)

    md = ["| " + " | ".join(label for _, label in columns) + " |"]
    md.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for field, _ in columns:
            value = "" if row.get(field) is None else str(row.get(field))
            values.append(value.replace("|", r"\|").replace("\n", " "))
        md.append("| " + " | ".join(values) + " |")
    (TABLES / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabularx}{\linewidth}{" + " ".join(
            "l" if i == 0 else "X" if len(columns) <= 5 else "r"
            for i, _ in enumerate(columns)
        ) + "}",
        r"\toprule",
        " & ".join(_escape_latex(label) for _, label in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            " & ".join(_escape_latex(row.get(field)) for field, _ in columns) + r" \\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabularx}"])
    (TABLES / f"{name}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def build_tables(evidence: dict[str, Any]) -> dict[str, int]:
    tables = {
        "table_01_datasets": (
            evidence["dataset_table"],
            [
                ("dataset", "Dataset"),
                ("modalities", "Modalities"),
                ("role", "Role"),
                ("sample_scope", "Evaluated scope"),
                ("limitations", "Limitations"),
            ],
        ),
        "table_02_sensor_routes": (
            evidence["focused_routes"],
            [
                ("route", "Route"),
                ("sequences", "Seq."),
                ("subsets", "Subsets"),
                ("surface_successes", "Surface pass"),
                ("surface_failures", "Surface fail"),
                ("measurement_successes", "Measurement pass"),
                ("measurement_failures", "Measurement fail"),
                ("router_acceptances", "Router accepts"),
                ("router_rejections", "Router rejects"),
                ("mean_surface_error_percent_wound_width", "MAE (% width)"),
                ("mean_reference_volume_error_percent", "Mean vol. error (%)"),
                ("worst_reference_volume_error_percent", "Worst vol. error (%)"),
            ],
        ),
        "table_03_calibration": (
            evidence["calibration"]["strict_holdout"]["threshold_recall"]
            + [
                {
                    "evaluation": "Strict holdout clean false alarms",
                    "threshold": "—",
                    "numerator": evidence["calibration"]["strict_holdout"]["clean_false_alarm_count"],
                    "denominator": evidence["calibration"]["strict_holdout"]["clean_false_alarm_denominator"],
                    "rate": evidence["calibration"]["strict_holdout"]["clean_false_alarm_count"]
                    / evidence["calibration"]["strict_holdout"]["clean_false_alarm_denominator"],
                    "unit": "fraction of clean configurations",
                    "aggregation": "configuration-level",
                }
            ]
            + [
                {
                    "evaluation": f"{row['target'].title()} target fault rejection",
                    "threshold": "V2 target ablation",
                    "numerator": row["fault_rejections"],
                    "denominator": row["fault_configurations"],
                    "rate": row["fault_rejection_rate"],
                    "unit": "fraction of non-clean target configurations",
                    "aggregation": f"clean FA {row['clean_false_alarms']}/{row['clean_controls']}",
                }
                for row in evidence["calibration"]["target_ablation"]["stepped_vs_planar"]
            ],
            [
                ("evaluation", "Evaluation"),
                ("threshold", "Threshold"),
                ("numerator", "Detected / false alarms"),
                ("denominator", "Denominator"),
                ("rate", "Rate"),
                ("unit", "Metric unit"),
                ("aggregation", "Aggregation"),
            ],
        ),
        "table_04_quantity_classes": (
            evidence["quantity_table"],
            [
                ("category", "Quantity class"),
                ("examples", "Examples"),
                ("basis", "Required basis"),
            ],
        ),
        "table_05_offline_cases": (
            evidence["offline_demo"]["case_rows"],
            [
                ("case_id", "Case"),
                ("expected_decision", "Expected"),
                ("actual_decision", "Actual"),
                ("stage_count", "Stages"),
                ("boundary_violations", "Boundary"),
                ("exclusion_zone_violations", "Exclusion"),
                ("unsupported_deposition_fraction", "Unsupported fraction"),
                ("planning_decision", "Planning status"),
                ("route_expected_match", "Match"),
            ],
        ),
    }
    for name, (rows, columns) in tables.items():
        _write_table(name, rows, columns)
    save_json(
        TABLES / "provenance.json",
        {
            "source": "paper/data/publication_evidence.json",
            "format_generation": "scripts/build_publication_package.py",
            "tables": {name: len(rows) for name, (rows, _) in tables.items()},
            "statistical_note": evidence["statistical_audit"]["reason"],
        },
    )
    return {name: len(rows) for name, (rows, _) in tables.items()}


def build_metric_audit(evidence: dict[str, Any]) -> int:
    columns = [
        ("metric", "Metric"),
        ("value", "Reported value"),
        ("numerator", "Numerator"),
        ("denominator", "Denominator"),
        ("unit", "Unit"),
        ("aggregation", "Aggregation"),
        ("confidence_interval", "Confidence interval"),
        ("interval_note", "Interval note"),
        ("data_origin", "Data origin"),
        ("evidence_sha256", "Evidence SHA-256"),
    ]
    evidence_hash = hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []

    def add(
        metric: str,
        value: Any,
        numerator: Any,
        denominator: Any,
        unit: str,
        aggregation: str,
        origin: str,
    ) -> None:
        rows.append({
            "metric": metric,
            "value": value,
            "numerator": numerator,
            "denominator": denominator,
            "unit": unit,
            "aggregation": aggregation,
            "confidence_interval": "Not estimated",
            "interval_note": evidence["statistical_audit"]["reason"],
            "data_origin": f"paper/data/publication_evidence.json#/{origin}",
            "evidence_sha256": evidence_hash,
        })

    for route in evidence["focused_routes"]:
        add(
            f"{route['route']} mean surface error",
            route.get("mean_surface_error_percent_wound_width"),
            "Not a count metric",
            f"{route.get('subsets', 0)} total route/subset observations; available-value denominator not retained",
            "% wound width",
            "Mean over available route/subset outcomes; correlated within four sequence contexts.",
            "focused_routes",
        )
        add(
            f"{route['route']} mean reference-volume error",
            route.get("mean_reference_volume_error_percent"),
            "Not a count metric",
            f"{route.get('subsets', 0)} total route/subset observations; available-value denominator not retained",
            "% computed reference volume",
            "Mean over available route/subset outcomes; reference-dependent and synthetic.",
            "focused_routes",
        )
        if route.get("completeness_5pct_width_mean_available_only") is not None:
            add(
                f"{route['route']} completeness at 5% wound width",
                route["completeness_5pct_width_mean_available_only"],
                "Point-level numerator not retained",
                f"{route.get('subsets', 0)} total route/subset observations; point denominator not retained",
                "fraction within 5% wound width",
                "Mean of available route/subset completeness values; descriptive only.",
                "focused_routes",
            )
    for route in evidence["alignment_sensitivity_routes"]:
        add(
            f"{route['route']} repeatability",
            route.get("repeatability_percent_wound_width"),
            "Not a count metric",
            f"{route.get('subsets', 0)} total route/subset observations; available-value denominator not retained",
            "% wound width",
            "Route-level descriptive summary; repeated subsets are correlated.",
            "alignment_sensitivity_routes",
        )
        add(
            f"{route['route']} reference-volume error",
            route.get("mean_reference_volume_error_percent"),
            "Not a count metric",
            f"{route.get('subsets', 0)} total route/subset observations; available-value denominator not retained",
            "% computed reference volume",
            "Route-level descriptive summary; synthetic reference only.",
            "alignment_sensitivity_routes",
        )
        if route.get("worst_reference_volume_error_percent") is not None:
            add(
                f"{route['route']} worst reference-volume error",
                route["worst_reference_volume_error_percent"],
                "Not a count metric",
                f"{route.get('subsets', 0)} total route/subset observations; available-value denominator not retained",
                "% computed reference volume",
                "Worst available route/subset outcome; synthetic reference.",
                "alignment_sensitivity_routes",
            )

    calibration = evidence["calibration"]
    strict = calibration["strict_holdout"]
    for threshold in strict["threshold_recall"]:
        add(
            f"{threshold['evaluation']} recall at {threshold['threshold']}",
            threshold["rate"],
            threshold["numerator"],
            threshold["denominator"],
            threshold["unit"],
            threshold["aggregation"],
            "calibration/strict_holdout/threshold_recall",
        )
    add(
        "Strict-holdout clean false-alarm rate",
        strict["clean_false_alarm_count"] / strict["clean_false_alarm_denominator"],
        strict["clean_false_alarm_count"],
        strict["clean_false_alarm_denominator"],
        "fraction of clean configurations",
        "Configuration-level descriptive ratio.",
        "calibration/strict_holdout",
    )
    for row in calibration["target_ablation"]["stepped_vs_planar"]:
        add(
            f"{row['target']} target fault rejection",
            row["fault_rejection_rate"],
            row["fault_rejections"],
            row["fault_configurations"],
            "fraction of fault configurations",
            "Configuration-level descriptive ratio; synthetic target design.",
            "calibration/target_ablation/stepped_vs_planar",
        )
        add(
            f"{row['target']} target clean false-alarm rate",
            row["clean_false_alarm_rate"],
            row["clean_false_alarms"],
            row["clean_controls"],
            "fraction of clean controls",
            "Configuration-level descriptive ratio.",
            "calibration/target_ablation/stepped_vs_planar",
        )

    clean = evidence["clean_measurement"]
    add(
        "Clean measurement accepted variants",
        clean["accepted"],
        clean["accepted"],
        clean["accepted"] + clean["rejected"],
        "accepted variants / total",
        "Deterministic variant-level gate result; variants are not independent subjects.",
        "clean_measurement",
    )
    hidden = evidence["healthy_disc"]
    for field, label in (
        ("macro_mae_fraction_of_diameter", "Healthy-disc macro MAE"),
        ("macro_p95_fraction_of_diameter", "Healthy-disc macro P95"),
    ):
        add(
            label,
            hidden[field],
            "Not a count metric",
            f"{hidden['test_patch_count']} held-out patches; {hidden['valid_global_method_records']}/{hidden['global_method_total_records']} valid patch-ring records",
            "fraction of hole diameter",
            "Frozen global thin-plate-spline method; macro summary across the sealed TEST split.",
            "healthy_disc",
        )
    add(
        "Healthy-disc uncertainty pointwise coverage",
        hidden["uncertainty_pointwise_coverage"],
        "Not exported as a point count",
        f"{hidden['valid_global_method_records']} valid model records; point denominator not retained in compact export",
        "fraction of evaluated points",
        "Frozen ensemble; TEST truth used for evaluation only.",
        "healthy_disc",
    )
    example_path = DATA / "figure08_hidden_disc_example.json"
    if example_path.is_file():
        example = json.loads(example_path.read_text(encoding="utf-8"))
        truth = np.asarray(example["heldout_truth_z"], dtype=float)
        prediction = np.asarray(example["tps_prediction_z"], dtype=float)
        low = np.asarray(example["envelope_low_z"], dtype=float)
        high = np.asarray(example["envelope_high_z"], dtype=float)
        point_error = np.abs(prediction - truth)
        point_count = len(truth)
        sample_hash = hashlib.sha256(example_path.read_bytes()).hexdigest()
        add(
            "Healthy-disc example TPS MAE",
            float(np.mean(point_error)),
            float(np.sum(point_error)),
            point_count,
            "dataset units",
            "Pointwise mean absolute error on one frozen TEST patch; points are spatially correlated.",
            "figure08_hidden_disc_example.json#/heldout_truth_z",
        )
        rows[-1]["evidence_sha256"] = sample_hash
        inside_count = int(np.count_nonzero((truth >= low) & (truth <= high)))
        add(
            "Healthy-disc example uncertainty-envelope coverage",
            inside_count / point_count,
            inside_count,
            point_count,
            "fraction of query points",
            "Single frozen TEST patch; illustrative only, not a population coverage estimate.",
            "figure08_hidden_disc_example.json#/envelope_low_z,/envelope_high_z",
        )
        rows[-1]["evidence_sha256"] = sample_hash
        add(
            "Healthy-disc example mean envelope width",
            float(np.mean(high - low)),
            float(np.sum(high - low)),
            point_count,
            "dataset units",
            "Pointwise mean envelope width on one frozen TEST patch.",
            "figure08_hidden_disc_example.json#/envelope_low_z,/envelope_high_z",
        )
        rows[-1]["evidence_sha256"] = sample_hash
    for model in evidence["reference_surface_models"]:
        add(
            f"{model['method']} healthy-ring blocked-holdout RMSE",
            model["blocked_holdout_rmse_dataset_units"],
            "Not a count metric",
            f"{model['healthy_samples']} healthy ring samples",
            "dataset units",
            "Blocked healthy-geometry holdout; no wound-bed points used for fitting.",
            "reference_surface_models",
        )
        add(
            f"{model['method']} reference-defined volume",
            model["reference_defined_volume_dataset_units3"],
            "Not a count metric",
            "one selected synthetic sequence",
            "dataset units cubed",
            "Model-dependent closure quantity; not true wound volume.",
            "reference_surface_models",
        )
    for field, label in (
        ("mean_point_to_surface_mm", "SALVE mean point-to-surface"),
        ("p95_point_to_surface_mm", "SALVE P95 point-to-surface"),
        ("completeness_fraction_at_1mm", "SALVE completeness at 1 mm"),
    ):
        for result in evidence["salve_shape_only"]:
            value = result[field]
            add(
                f"{label} ({result['device']}, {result['method']})",
                value,
                "Not a count metric" if field != "completeness_fraction_at_1mm" else "Point count not retained",
                f"{result['predicted_points']} predicted points" if field != "completeness_fraction_at_1mm" else f"{result['gt_samples']} GT samples",
                "mm after GT similarity alignment" if field != "completeness_fraction_at_1mm" else "fraction within 1 mm after GT similarity alignment",
                "Shape-only evaluation; GT-derived similarity alignment; not independent metric-scale recovery.",
                "salve_shape_only",
            )
    rgb_shape = evidence["syn3dwound_rgb_shape"]
    for field, label, unit in (
        ("mean_point_to_surface_fraction_of_wound_width", "Syn3DWound mean point-to-surface", "fraction of wound width"),
        ("p95_point_to_surface_fraction_of_wound_width", "Syn3DWound P95 point-to-surface", "fraction of wound width"),
        ("completeness_fraction_at_5pct_width", "Syn3DWound completeness at 5% width", "fraction within 5% wound width"),
    ):
        add(
            label,
            rgb_shape[field],
            "Not a count metric",
            "Sequence-level sample count is not retained in the compact evidence record",
            unit,
            "Descriptive shape comparison after evaluation-only similarity alignment.",
            "syn3dwound_rgb_shape",
        )
    decision = evidence["architecture_decision"]
    for edge, values in decision["failures_by_edge_condition"].items():
        add(
            f"Direct-reference failures ({edge} edge deformation)",
            values["direct_reference_failures"],
            values["direct_reference_failures"],
            values["cases"],
            "failed cases / cases",
            "Case-level synthetic architecture gate; six cases per edge condition.",
            f"architecture_decision/failures_by_edge_condition/{edge}",
        )
        add(
            f"Maximum direct overfill ({edge} edge deformation)",
            values["maximum_direct_overfill_in_declared_layers"],
            "Not a count metric",
            f"{values['cases']} synthetic cases",
            "declared synthetic layer equivalents",
            "Maximum over cases in the edge-deformation subgroup.",
            f"architecture_decision/failures_by_edge_condition/{edge}",
        )
        add(
            f"Boundary violations ({edge} edge deformation)",
            values["boundary_violations"],
            values["boundary_violations"],
            f"{values['cases']} cases; boundary-event denominator not exported",
            "violation events",
            "Cumulative synthetic case-level violation count.",
            f"architecture_decision/failures_by_edge_condition/{edge}",
        )
    for case in evidence["offline_demo"]["case_rows"]:
        add(
            f"Offline demo stage count ({case['case_id']})",
            case["stage_count"],
            "Not a ratio",
            "one synthetic case",
            "stages",
            "Deterministic software demonstration; not robot execution.",
            "offline_demo/case_rows",
        )

    _write_csv(TABLES / "metric_audit.csv", rows, columns)
    return len(rows)


def _inventory_classification(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".gif"}:
        return "video"
    if suffix in {".csv", ".tex"} and "table" in path:
        return "table"
    if suffix in {".csv", ".json"} and ("metric" in path or "summary" in path):
        return "metric"
    if suffix == ".json" and ("manifest" in path or "provenance" in path):
        return "manifest"
    if suffix in {".png", ".svg", ".pdf"}:
        return "figure/graph"
    if suffix in {".py", ".ipynb"}:
        return "code/test/notebook"
    if suffix in {".md", ".tex"}:
        return "report/documentation"
    if suffix in {".npz", ".ply", ".obj"}:
        return "geometry/data"
    return "metadata/artifact"


def build_artifact_inventory(evidence: dict[str, Any]) -> int:
    columns = [
        ("artifact_path", "Artifact path"),
        ("originating_milestone", "Originating milestone"),
        ("classification", "Figure/table/report/metric/test/etc."),
        ("publication_relevance", "Publication relevance"),
        ("status", "Final or diagnostic"),
        ("data_type", "Synthetic or real data"),
        ("safe_to_commit", "Safe to commit"),
        ("exists_at_inventory", "Present in workspace"),
        ("size_bytes", "Bytes"),
        ("notes", "Notes"),
    ]
    rows: dict[str, dict[str, Any]] = {}

    def add(path: str, milestone: str, relevance: str, status: str, data_type: str,
            safe: str, notes: str, require_present: bool = True) -> None:
        normalized = path.replace("\\", "/")
        file_path = ROOT / Path(normalized)
        exists = file_path.is_file()
        if require_present and not exists:
            return
        rows[normalized] = {
            "artifact_path": normalized,
            "originating_milestone": milestone,
            "classification": _inventory_classification(normalized),
            "publication_relevance": relevance,
            "status": status,
            "data_type": data_type,
            "safe_to_commit": safe,
            "exists_at_inventory": str(exists).lower(),
            "size_bytes": file_path.stat().st_size if exists else "",
            "notes": notes,
        }

    fixed_paths = [
        "README.md", ".gitignore", "robot/README.md", "robot/readme.md",
        "notebooks/README.md", "notebooks/04_rgb_wound_3d_reconstruction.ipynb",
        "notebooks/04_replay_manifest.json", "notebooks/04_selected_results.html",
        "scripts/prepare_publication_notebook.py", "scripts/export_publication_evidence.py",
        "scripts/build_publication_package.py", "paper/main.tex", "paper/references.bib",
        "docs/reproduction.md", "docs/claims_and_limitations.md",
        "docs/dataset_provenance.md", "docs/publication_assessment.md",
        "docs/statistical_audit.md",
        "tests/test_clean_measurement.py", "tests/test_coppelia_simulation.py",
        "tests/test_observable_geometry.py", "tests/test_rgb_notebook.py",
    ]
    for path in fixed_paths:
        add(path, "Repository integration/publication package", "yes", "final",
            "code/metadata", "yes", "Tracked documentation, source, test, or reproduction artifact.")

    for folder, milestone, dtype in (
        ("paper/data", "Publication evidence export", "mixed evidence/derived data"),
        ("paper/figure_data", "Publication figure generation", "mixed sealed metrics and derived data"),
        ("paper/figures", "Publication figure generation", "derived from sealed evidence"),
        ("paper/tables", "Publication table generation", "derived from sealed evidence"),
        ("results/coppelia_demo", "Offline five-case integration demonstration", "procedural synthetic"),
        ("robot/simulation", "Offline five-case integration demonstration", "synthetic software"),
    ):
        root = ROOT / folder
        if root.exists():
            for file_path in sorted(root.rglob("*")):
                if not file_path.is_file():
                    continue
                relative = file_path.relative_to(ROOT).as_posix()
                diagnostic = "diagnostic preview" if "preview" in file_path.name.lower() or "smoke" in file_path.name.lower() else "final curated artifact"
                add(relative, milestone, "yes" if relative.startswith(("paper/", "results/coppelia_demo/")) else "supporting",
                    diagnostic, dtype, "yes", "Generated from repository code; synthetic/demo artifacts, not physical validation.")

    sealed_root = ROOT / "results" / "rgb_wound"
    if sealed_root.exists():
        sealed_manifest_paths = {
            row["path"] for row in evidence["source_manifest"]
            if row["path"].startswith("results/rgb_wound/")
        }
        for file_path in sorted(sealed_root.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(ROOT).as_posix()
            milestone = file_path.relative_to(sealed_root).parts[0]
            name = file_path.name.lower()
            final_like = (
                relative in sealed_manifest_paths
                or any(token in name for token in ("final", "seal", "manifest", "report", "summary", "freeze", "decision"))
            )
            add(
                relative,
                f"Sealed prior scientific milestone: {milestone}",
                "source evidence" if final_like else "diagnostic/intermediate",
                "sealed result/receipt" if final_like else "diagnostic/intermediate",
                "SALVE phantom evaluation" if "salve" in relative.lower() else "synthetic computational result",
                "no",
                "Local result tree is hash-linked where applicable and excluded from Git; curated summaries are exported separately.",
            )

    image_provenance_path = DATA / "figure03_provenance.json"
    if image_provenance_path.is_file():
        image_provenance = json.loads(image_provenance_path.read_text(encoding="utf-8"))
        raw_rgb_path = image_provenance["input_image"]
        add(
            raw_rgb_path,
            "Syn3DWound public-data input for Figure 3",
            "source image",
            "selected source input",
            "synthetic rendered RGB",
            "no",
            "Public dataset sample is excluded from Git; its path and SHA-256 are retained in paper/data/figure03_provenance.json.",
            require_present=False,
        )

    for source in evidence["source_manifest"]:
        path = source["path"]
        if path.startswith("results/rgb_wound/"):
            add(
                path, "Sealed prior scientific milestone", "source evidence",
                "sealed final input", "synthetic computational result", "no",
                "Local sealed output retained by hash; excluded from Git to avoid duplicating large intermediate results.",
                require_present=False,
            )
        elif path not in rows:
            add(path, "Publication evidence source", "yes", "final",
                "synthetic computational result", "yes",
                "Curated evidence input referenced by the committed publication package.",
                require_present=False,
            )

    add("docs/artifact_inventory.csv", "Repository integration/publication package",
        "yes", "final inventory", "metadata", "yes",
        "Generated inventory of curated outputs and hash-linked local sources.")
    _write_csv(ROOT / "docs" / "artifact_inventory.csv", list(rows.values()), columns)
    return len(rows)


def build_figures(evidence: dict[str, Any]) -> None:
    _style()
    figure_builders = [
        ("figure_01", lambda: _draw_architecture(), ["paper/data/publication_evidence.json"]),
        ("figure_02", lambda: _draw_evidence_stack(evidence), ["paper/data/publication_evidence.json"]),
        ("figure_03", lambda: _draw_reconstruction(evidence), [
            "paper/data/figure03_provenance.json",
            "paper/data/figure03_surface_data.npz",
        ]),
        ("figure_04", lambda: _draw_routes(evidence), ["paper/data/publication_evidence.json"]),
        ("figure_05", lambda: _draw_misalignment(evidence), ["paper/data/publication_evidence.json"]),
        ("figure_06", lambda: _draw_calibration(evidence), ["paper/data/publication_evidence.json"]),
        ("figure_07", lambda: _draw_reference_ambiguity(evidence), ["paper/data/publication_evidence.json"]),
        ("figure_08", lambda: _draw_hidden_disc(evidence), [
            "paper/data/figure08_hidden_disc_example.json",
            "paper/data/publication_evidence.json",
        ]),
        ("figure_09", lambda: _draw_staged_fill(evidence), [
            "paper/data/figure09_staged_geometry.npz",
            "results/coppelia_demo/closed_loop_trace.json",
        ]),
    ]
    captions = []
    for name, builder, sources in figure_builders:
        built = builder()
        if name == "figure_03":
            fig, data, external_sources = built
            sources += external_sources
        else:
            fig, data = built
        _save_figure(name, fig, data, evidence, sources)
        captions.append(f"### {name.replace('_', ' ').title()}\n\n{FIGURE_CAPTIONS[name]}")
    (FIGURES / "captions.md").write_text("\n\n".join(captions) + "\n", encoding="utf-8")


def build_selected_results_html(evidence: dict[str, Any]) -> None:
    images = []
    for index in range(1, 10):
        name = f"figure_{index:02d}"
        caption = html.escape(FIGURE_CAPTIONS[name])
        images.append(
            f'<section><h2>{name.replace("_", " ").title()}</h2>'
            f'<img src="../paper/figures/{name}.png" alt="{name}">'
            f'<p>{caption}</p></section>'
        )
    status = (
        "<strong>Publication package:</strong> PUBLICATION_READY_WITH_WRITING_GAPS. "
        "All scientific outputs shown are computational; no physical printing or CoppeliaSim execution was performed."
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Selected sealed results</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1120px;margin:2rem auto;padding:0 1rem;color:#17324d}}
section{{margin:2rem 0 3rem}}img{{width:100%;height:auto;border:1px solid #d5dde5}}.notice{{padding:1rem;background:#fff2db}}</style></head>
<body><h1>Selected sealed results</h1><p class="notice">{status}</p>
<p>Metrics are descriptive; correlated views, subsets, patches and pixels are not independent samples.
Synthetic depth is mesh-rendered. SALVE comparisons use GT similarity alignment and therefore do not
prove independent metric scale. Pre-wound healthy skin is not identifiable from current observations.</p>
{''.join(images)}
</body></html>"""
    out = ROOT / "notebooks" / "04_selected_results.html"
    out.write_text(doc, encoding="utf-8")


def validate_outputs() -> dict[str, Any]:
    missing = []
    for i in range(1, 10):
        stem = f"figure_{i:02d}"
        for extension in ("png", "svg", "provenance.json"):
            path = FIGURES / f"{stem}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(path.relative_to(ROOT).as_posix())
        data_path = FIGURE_DATA / f"{stem}.json"
        if not data_path.is_file() or data_path.stat().st_size == 0:
            missing.append(data_path.relative_to(ROOT).as_posix())
        provenance_path = FIGURES / f"{stem}.provenance.json"
        if provenance_path.is_file():
            metadata = json.loads(provenance_path.read_text(encoding="utf-8"))
            if metadata["png_dpi"] != 300 or not metadata["source_hashes"]:
                missing.append(f"{stem}: invalid DPI or missing source hashes")
        png_path = FIGURES / f"{stem}.png"
        if png_path.is_file():
            with Image.open(png_path) as image:
                if image.info.get("dpi", (0, 0))[0] < 299:
                    missing.append(f"{stem}: PNG is below 300 DPI")
    for table in range(1, 6):
        for extension in ("csv", "md", "tex"):
            path = TABLES / f"table_{table:02d}_"
            matches = list(TABLES.glob(f"table_{table:02d}_*.{extension}"))
            if not matches or any(p.stat().st_size == 0 for p in matches):
                missing.append(str(path.relative_to(ROOT)) + f"*.{extension}")
    if missing:
        raise RuntimeError("Publication outputs missing or invalid: " + ", ".join(missing))
    gallery_path = ROOT / "notebooks" / "04_selected_results.html"
    if not gallery_path.is_file():
        raise RuntimeError("The selected-results HTML gallery was not generated.")
    gallery_text = gallery_path.read_text(encoding="utf-8")
    for i in range(1, 10):
        image_ref = f"../paper/figures/figure_{i:02d}.png"
        if image_ref not in gallery_text or not (FIGURES / f"figure_{i:02d}.png").is_file():
            raise RuntimeError(f"Broken gallery image reference: {image_ref}")
    for path in (TABLES / "metric_audit.csv", ROOT / "docs" / "artifact_inventory.csv"):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required audit output is missing: {path.relative_to(ROOT)}")
    return {
        "figures": 9,
        "tables": 5,
        "audited_metrics": sum(1 for _ in csv.DictReader((TABLES / "metric_audit.csv").open(encoding="utf-8"))),
        "inventoried_artifacts": sum(1 for _ in csv.DictReader((ROOT / "docs" / "artifact_inventory.csv").open(encoding="utf-8"))),
        "png_dpi": 300,
        "vector_format": "SVG",
        "status": "generated",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Generate from committed paper/data evidence without reading raw experiment results.",
    )
    args = parser.parse_args()
    evidence = load_evidence()
    _ = args.replay
    tables = build_tables(evidence)
    metric_count = build_metric_audit(evidence)
    build_figures(evidence)
    build_selected_results_html(evidence)
    inventory_count = build_artifact_inventory(evidence)
    result = validate_outputs()
    result["table_rows"] = tables
    result["audited_metrics"] = metric_count
    result["inventoried_artifacts"] = inventory_count
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
