"""Fetch and verify the single public Syn3DWound image used by Figure 3."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / (
    "data/rgb_wound/syn_audit/Syn3DWound/subject_1/"
    "Syn3DWound_slough-pressure-ulcer-008_sacrum_weak_lighting_phi/"
    "images/image_000.png"
)
SOURCE_URL = "https://data.csiro.au/dap/ws/v2/collections/73258/data/91243876"
EXPECTED_SHA256 = "9355d1cc45c097b29244d98a59e2ae33088086f5c9a40042a412a462cd5765ad"
EXPECTED_SIZE = 7_027_644


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Figure 3 source image is missing: {path}. "
            "Run `python scripts/fetch_figure3_example.py`."
        )
    actual = sha256_file(path)
    if actual != EXPECTED_SHA256:
        raise ValueError(
            f"Figure 3 source image has an unexpected SHA-256: {path} "
            f"(expected {EXPECTED_SHA256}, got {actual}). "
            "Do not use this file; remove or replace it with the identified dataset file."
        )
    if path.stat().st_size != EXPECTED_SIZE:
        raise ValueError(
            f"Figure 3 source image has an unexpected size: {path} "
            f"(expected {EXPECTED_SIZE}, got {path.stat().st_size})."
        )
    return path


def fetch_image(destination: Path = DESTINATION) -> Path:
    if destination.exists():
        return verify_image(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(SOURCE_URL, headers={"User-Agent": "bioprinting-pipeline/figure3"})
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=".figure3-", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
        verify_image(temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return verify_image(destination)


def main() -> None:
    path = fetch_image()
    print(f"Verified Figure 3 input: {path.relative_to(ROOT).as_posix()}")
    print(f"SHA-256: {EXPECTED_SHA256}")
    print("Source: Syn3DWound, DOI 10.25919/5rwz-ts17, CSIRO file 91243876")


if __name__ == "__main__":
    main()
