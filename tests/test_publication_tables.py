from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import build_publication_package as publication


TABLES = Path(__file__).resolve().parents[1] / "paper" / "tables"


@pytest.fixture
def rebuilt_tables(tmp_path, monkeypatch):
    evidence = publication.load_evidence()
    monkeypatch.setattr(publication, "TABLES", tmp_path)
    counts = publication.build_tables(evidence)
    return tmp_path, evidence, counts


def _latex_rows(panel: str) -> list[list[str]]:
    body = panel.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    return [
        line.strip().removesuffix(r" \\").split(" & ")
        for line in body.strip().splitlines()
    ]


def test_table_rebuild_matches_committed_formats(rebuilt_tables):
    generated, _, counts = rebuilt_tables
    assert counts == {
        "table_01_datasets": 4,
        "table_02_sensor_routes": 6,
        "table_03_calibration": 9,
        "table_04_quantity_classes": 4,
        "table_05_offline_cases": 5,
    }
    for name in counts:
        for extension in ("csv", "md", "tex"):
            filename = f"{name}.{extension}"
            assert (generated / filename).read_text(encoding="utf-8") == (
                TABLES / filename
            ).read_text(encoding="utf-8")


def test_sensor_table_renders_all_counts_and_source_errors(rebuilt_tables):
    generated, evidence, _ = rebuilt_tables
    tex = (generated / "table_02_sensor_routes.tex").read_text(encoding="utf-8")
    decision_panel, error_panel = tex.strip().split("\n\n")
    decisions = _latex_rows(decision_panel)
    errors = _latex_rows(error_panel)
    assert len(decisions) == len(errors) == len(evidence["focused_routes"])

    for decision, error, source in zip(decisions, errors, evidence["focused_routes"]):
        assert decision[0] == error[0]
        assert decision[1:] == [
            str(source["sequences"]),
            str(source["subsets"]),
            f'{source["surface_successes"]}/{source["surface_failures"]}',
            f'{source["measurement_successes"]}/{source["measurement_failures"]}',
            f'{source["router_acceptances"]}/{source["router_rejections"]}',
        ]
        assert error[1:] == [
            f'{source["mean_surface_error_percent_wound_width"]:.2f}',
            f'{source["mean_reference_volume_error_percent"]:.2f}',
            f'{source["worst_reference_volume_error_percent"]:.2f}',
        ]


def test_calibration_table_keeps_counts_and_converts_rates_to_percent(rebuilt_tables):
    generated, evidence, _ = rebuilt_tables
    tex = (generated / "table_03_calibration.tex").read_text(encoding="utf-8")
    assert r"Rate (\%)" in tex
    rows = _latex_rows(tex)
    with (generated / "table_03_calibration.csv").open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    assert len(rows) == len(source_rows) == 9
    for row, source in zip(rows, source_rows):
        threshold = "--" if source["threshold"] == "—" else publication._escape_latex(
            source["threshold"]
        )
        assert row[1:] == [
            threshold,
            f'{source["numerator"]}/{source["denominator"]}',
            f'{100 * float(source["rate"]):.2f}',
        ]
    for target in evidence["calibration"]["target_ablation"]["stepped_vs_planar"]:
        assert target["clean_false_alarms"] == 0
        assert target["clean_controls"] == 24


def test_offline_table_preserves_all_case_outcomes(rebuilt_tables):
    generated, evidence, _ = rebuilt_tables
    tex = (generated / "table_05_offline_cases.tex").read_text(encoding="utf-8")
    decision_panel, outcome_panel = tex.strip().split("\n\n")
    decisions = _latex_rows(decision_panel)
    outcomes = _latex_rows(outcome_panel)
    source_rows = evidence["offline_demo"]["case_rows"]
    assert len(decisions) == len(outcomes) == len(source_rows)

    labels = {
        "RECONSTRUCTION_ACCEPTED_FOR_SIMULATION": "Accepted for simulation",
        "BOUNDARY_RESOLUTION_INSUFFICIENT": "Boundary resolution insufficient",
        "SURFACE_INCOMPLETE": "Surface incomplete",
        "RGB_DEPTH_CALIBRATION_FAILED": "RGB-depth calibration failed",
        "EXTERNAL_TARGET_REQUIRED": "External target required",
        "SIMULATION_PLAN_COMPLETE": "Simulation plan complete",
        "REFUSE_TO_PLAN": "Refuse to plan",
        "NO_SAFE_DEPOSITION_REGION": "No safe deposition region",
    }
    for decision, outcome, source in zip(decisions, outcomes, source_rows):
        assert decision[0] == outcome[0]
        assert decision[1:] == [
            labels[source["expected_decision"]],
            labels[source["actual_decision"]],
            str(source["stage_count"]),
        ]
        assert outcome[1:] == [
            labels[source["planning_decision"]],
            str(source["boundary_violations"]),
            str(source["exclusion_zone_violations"]),
            f'{source["unsupported_deposition_fraction"]:.2f}',
            "Yes" if source["route_expected_match"] else "No",
        ]


def test_missing_values_and_invalid_case_matches_are_explicit():
    assert publication._display_number(None) == "--"
    with pytest.raises(TypeError, match="match must be a boolean"):
        publication._compact_latex_table("table_05_offline_cases", [{
            "case_id": "test",
            "route_expected_match": "False",
        }])


def test_offline_table_does_not_hide_a_decision_mismatch():
    case = dict(publication.load_evidence()["offline_demo"]["case_rows"][0])
    case["actual_decision"] = "SURFACE_INCOMPLETE"
    case["route_expected_match"] = False
    table = publication._compact_latex_table("table_05_offline_cases", [case])
    assert table is not None
    decision_panel, outcome_panel = table.split("\n\n")
    assert _latex_rows(decision_panel)[0][1:3] == [
        "Accepted for simulation", "Surface incomplete"
    ]
    assert _latex_rows(outcome_panel)[0][-1] == "No"
