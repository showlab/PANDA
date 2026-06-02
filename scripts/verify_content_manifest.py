#!/usr/bin/env python3
"""Verify a deterministic file-content manifest against a local directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["name"]):
        digest.update(
            "{}\0{}\0{}\n".format(row["name"], row["size"], row["sha256"]).encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def verify(root: Path, manifest_path: Path, expected: str) -> str:
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("content root and manifest must exist")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ValueError("manifest files must be a list")
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("manifest file names must be unique strings")
        names.append(row["name"])
    if len(names) != len(set(names)):
        raise ValueError("manifest file names must be unique strings")
    if any(
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or ".git" in Path(name).parts
        for name in names
    ):
        raise ValueError("manifest file names must be safe deployment-content paths")
    actual_names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )
    if sorted(names) != actual_names:
        raise ValueError("manifest file set does not match content root")
    for row in rows:
        name = row["name"]
        path = root / name
        size = row.get("size")
        digest = row.get("sha256")
        if not isinstance(size, int) or size < 0 or not isinstance(digest, str):
            raise ValueError(f"invalid manifest entry: {name}")
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ValueError(f"content mismatch: {name}")
    aggregate = _aggregate(rows)
    if payload.get("aggregate_sha256") != aggregate or expected.lower() != aggregate:
        raise ValueError("manifest aggregate does not match expected content digest")
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    print(verify(Path(args.root), Path(args.manifest), args.expected_sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
