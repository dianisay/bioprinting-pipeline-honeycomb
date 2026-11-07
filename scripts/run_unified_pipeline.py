"""Run the unified offline pipeline on the five frozen cases and save its evidence.

Usage:
    python scripts/run_unified_pipeline.py [--output results/unified_pipeline]

Outputs:
    summary.json                  per-case stage trace and final decision
    supported_case_toolpath.csv   staged honeycomb toolpath preview (case A)
    unified_pipeline.png / .svg   six-panel figure from the values above
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.unified import STAGE_NAMES, public_summary, run_all_cases  # noqa: E402

DEFAULT_OUTPUT = ROOT / "results" / "unified_pipeline"


def write_toolpath(path: Path, result: dict) -> int:
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "cell", "deposition_planned", "x_m", "y_m", "z_m", "nx", "ny", "nz"])
        for stage in result["plan"]["stages"]:
            tp = stage["_toolpath"]
            for p, n, c, d in zip(tp["points_m"], tp["normals"], tp["cell"], tp["deposit_flag"]):
                writer.writerow([stage["stage"], int(c), int(d), *(f"{v:.6f}" for v in p), *(f"{v:.6f}" for v in n)])
                rows += 1
    return rows


def make_figure(results: list[dict], png: Path, svg: Path) -> None:
    supported = next(r for r in results if r.get("plan"))
    obs, ident, surf, plan = supported["_observation"], supported["_identified"], supported["_surface"], supported["plan"]
    pixel_mm = supported["pixel_size_m_declared"] * 1000
    cx, cy = ident["centroid_px"]
    ys, xs = np.nonzero(ident["mask"])
    pad = 12
    crop = (slice(ys.min() - pad, ys.max() + pad), slice(xs.min() - pad, xs.max() + pad))
    extent_mm = [(xs.min() - pad - cx) * pixel_mm, (xs.max() + pad - cx) * pixel_mm,
                 (ys.max() + pad - cy) * pixel_mm, (ys.min() - pad - cy) * pixel_mm]

    fig = plt.figure(figsize=(15, 9.2))
    grid = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(obs["rgb"][:, :, ::-1])
    ax.set_title("(a) OBSERVE: synthetic RGB image")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 1])
    base = np.zeros(ident["mask"].shape + (3,))
    base[ident["mask"]] = (0.85, 0.85, 0.85)
    base[plan["supported"]] = (0.55, 0.8, 0.55)
    base[plan["in_lattice"] & ident["mask"]] = (0.25, 0.6, 0.35)
    ax.imshow(base[crop], extent=extent_mm)
    side_px = plan["honeycomb"]["hex_side_m"] / supported["pixel_size_m_declared"]
    stage1 = plan["stages"][0]["_toolpath"]
    for cell in np.unique(stage1["cell"]):
        centre = plan["honeycomb"]["centres_px"][cell]
        ang = np.radians(np.arange(0, 361, 60))
        ax.plot((centre[0] + side_px * np.cos(ang) - cx) * pixel_mm,
                (centre[1] + side_px * np.sin(ang) - cy) * pixel_mm, color="k", lw=0.7)
    ax.set_title("(b) IDENTIFY + HONEYCOMB\nwound (grey), safe region (light green),\ncells used (dark green)")
    ax.set_xlabel("x (mm, synthetic scale)")
    ax.set_ylabel("y (mm, synthetic scale)")

    ax = fig.add_subplot(grid[0, 2])
    image = ax.imshow((surf["height_m"] * 1000)[crop], extent=extent_mm, cmap="viridis")
    fig.colorbar(image, ax=ax, label="height vs. skin plane (mm)")
    ax.set_title("(c) SURFACE: wound-bed height from depth\n(negative = below the skin plane)")
    ax.set_xlabel("x (mm, synthetic scale)")

    ax = fig.add_subplot(grid[1, 0], projection="3d")
    colours = plt.cm.plasma(np.linspace(0.1, 0.8, len(plan["stages"])))
    for stage, colour in zip(plan["stages"], colours):
        tp = stage["_toolpath"]
        dep = tp["deposit_flag"]
        p = tp["points_m"] * 1000
        ax.plot(p[dep, 0], p[dep, 1], p[dep, 2], ".", ms=1.4, color=colour, label=f"stage {stage['stage']}")
    ax.set_title("(d) Conformal honeycomb toolpath\n(tool positions, offset along surface normal)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(grid[1, 1])
    row = int(round(cy))
    cols = np.arange(xs.min(), xs.max() + 1)
    x_mm = (cols - cx) * pixel_mm
    ax.plot(x_mm, plan["initial_height_m"][row, cols] * 1000, color="k", lw=2, label="measured before filling")
    for stage, colour in zip(plan["stages"], colours):
        ax.plot(x_mm, stage["_surface_after"][row, cols] * 1000, color=colour, label=f"after stage {stage['stage']}")
    ax.axhline(0.0, color="tab:red", ls="--", label="operator final target")
    ax.set_title(f"(e) STAGED FILL: centre cross-section\nstop rule: {plan['summary']['stop_reason']}")
    ax.set_xlabel("x (mm, synthetic scale)")
    ax.set_ylabel("height (mm)")
    ax.legend(fontsize=8, loc="lower center")

    ax = fig.add_subplot(grid[1, 2])
    reached = [STAGE_NAMES.index(r["final_stage"]) for r in results]
    labels = [r["case_id"] for r in results]
    colours_bar = ["tab:green" if r.get("plan") else "tab:orange" for r in results]
    ax.barh(range(len(results)), [x + 1 for x in reached], color=colours_bar)
    for i, r in enumerate(results):
        ax.text(0.1, i, r["final_decision"], va="center", fontsize=7.5, color="black")
    ax.set_yticks(range(len(results)), labels, fontsize=8)
    ax.set_xticks(range(1, len(STAGE_NAMES) + 1), STAGE_NAMES, rotation=35, ha="right", fontsize=8)
    ax.invert_yaxis()
    ax.set_title("(f) Last stage reached by each case\n(orange = stopped safely, green = plan preview)")

    fig.suptitle("Unified offline pipeline: one synthetic case from image to staged honeycomb toolpath "
                 "(no printing, no robot execution)", fontsize=12)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    results = run_all_cases()
    summary = {
        "description": "Unified offline pipeline over the five frozen synthetic cases.",
        "stages": list(STAGE_NAMES),
        "print_authorized": False,
        "physical_robot_commands_enabled": False,
        "cases": [public_summary(r) for r in results],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    supported = next(r for r in results if r.get("plan"))
    rows = write_toolpath(args.output / "supported_case_toolpath.csv", supported)
    make_figure(results, args.output / "unified_pipeline.png", args.output / "unified_pipeline.svg")
    for r in results:
        print(f"{r['case_id']:<26} stopped at {r['final_stage']:<12} -> {r['final_decision']}")
    print(f"toolpath rows: {rows}; outputs in {args.output}")
    return summary


if __name__ == "__main__":
    main()
