from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from scripts import build_publication_package as publication


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "paper" / "data"


def _configure_replay(tmp_path: Path, monkeypatch, source: Path | None) -> None:
    root = tmp_path
    data = root / "paper" / "data"
    data.mkdir(parents=True)
    for name in (
        "figure03_reviewed_mask.png",
        "figure03_error_heatmap.png",
        "figure03_surface_data.npz",
    ):
        shutil.copy2(DATA_ROOT / name, data / name)

    input_path = Path(
        "data/rgb_wound/syn_audit/Syn3DWound/subject_1/"
        "Syn3DWound_slough-pressure-ulcer-008_sacrum_weak_lighting_phi/"
        "images/image_000.png"
    )
    provenance = json.loads((DATA_ROOT / "figure03_provenance.json").read_text(encoding="utf-8"))
    provenance["input_image"] = input_path.as_posix()
    if source is not None:
        target = root / input_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        provenance["input_image_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (data / "figure03_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    monkeypatch.setattr(publication, "ROOT", root)
    monkeypatch.setattr(publication, "DATA", data)


def _create_rgb_fixture(path: Path) -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 90
    image[:, :, 1] = 140
    Image.fromarray(image, mode="RGB").save(path)


def test_figure3_replay_fails_clearly_when_source_image_is_missing(tmp_path, monkeypatch):
    _configure_replay(tmp_path, monkeypatch, source=None)
    with pytest.raises(FileNotFoundError, match="scripts/fetch_figure3_example.py"):
        publication._draw_reconstruction(publication.load_evidence())


def test_figure3_replay_rejects_an_image_with_the_wrong_hash(tmp_path, monkeypatch):
    fixture = tmp_path / "incorrect.png"
    _create_rgb_fixture(fixture)
    _configure_replay(tmp_path / "replay", monkeypatch, source=fixture)
    provenance_path = publication.DATA / "figure03_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_image_sha256"] = (
        "9355d1cc45c097b29244d98a59e2ae33088086f5c9a40042a412a462cd5765ad"
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected SHA-256"):
        publication._draw_reconstruction(publication.load_evidence())


def test_figure3_replay_succeeds_with_a_hash_verified_source(tmp_path, monkeypatch):
    fixture = tmp_path / "valid.png"
    _create_rgb_fixture(fixture)
    _configure_replay(tmp_path / "replay", monkeypatch, source=fixture)
    fig, data, _ = publication._draw_reconstruction(publication.load_evidence())
    try:
        assert len(fig.axes) == 6
        assert data["source_image_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    finally:
        plt.close(fig)
