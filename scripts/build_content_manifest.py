#!/usr/bin/env python3
"""Build a deterministic PANDA content manifest for a local model directory."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from verify_content_manifest import _aggregate, _sha256


def build_manifest(root: Path, model_id: str) -> dict:
    """Return the manifest consumed by ``verify_content_manifest.py``."""
    if not root.is_dir():
        raise ValueError("content root must be an existing directory")
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or ".git" in path.relative_to(root).parts:
            continue
        rows.append(
            {
                "name": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": "panda.content_manifest.v2",
        "model_id": model_id,
        "aggregate_algorithm": (
            "sha256 over sorted UTF-8 lines: relative_path NUL size NUL "
            "file_sha256 LF; .git excluded"
        ),
        "aggregate_sha256": _aggregate(rows),
        "files": rows,
    }


def atomic_write_manifest(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("manifest output must be outside the content root")
    payload = build_manifest(root, args.model_id)
    atomic_write_manifest(output, payload)
    print(payload["aggregate_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
