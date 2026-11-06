"""Remove bulky saved cell outputs while preserving a verifiable replay receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04_rgb_wound_3d_reconstruction.ipynb"
RECEIPT = ROOT / "notebooks" / "04_replay_manifest.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare() -> dict:
    raw = NOTEBOOK.read_bytes()
    document = json.loads(raw)
    if RECEIPT.exists():
        existing = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if existing["clean_notebook_sha256"] == _sha256(raw):
            return existing
        raise RuntimeError(f"Existing replay receipt does not match {NOTEBOOK.name}.")

    cells = document.get("cells", [])
    output_receipts = []
    removed_bytes = 0
    for index, cell in enumerate(cells):
        outputs = cell.get("outputs", [])
        if outputs:
            serialized = json.dumps(
                outputs, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            output_receipts.append(
                {
                    "cell_index": index,
                    "output_count": len(outputs),
                    "serialized_bytes": len(serialized),
                    "sha256": _sha256(serialized),
                }
            )
            removed_bytes += len(serialized)
        cell["outputs"] = []
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None

    document["metadata"].pop("widgets", None)
    clean_bytes = (json.dumps(document, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    NOTEBOOK.write_bytes(clean_bytes)
    receipt = {
        "schema": "publication-notebook-output-removal-v1",
        "notebook": NOTEBOOK.relative_to(ROOT).as_posix(),
        "source_notebook_sha256_before_cleanup": _sha256(raw),
        "clean_notebook_sha256": _sha256(clean_bytes),
        "source_bytes": len(raw),
        "clean_bytes": len(clean_bytes),
        "removed_serialized_output_bytes": removed_bytes,
        "output_cell_count": len(output_receipts),
        "output_receipts": output_receipts,
        "replay_data_and_embedded_manifests_retained_in_code_cells": True,
    }
    RECEIPT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
