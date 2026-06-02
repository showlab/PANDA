"""Local-only content identities for paper-mode vision tools.

Checkpoint files use file SHA-256. CLIP directories use the same sorted
(name, size, file SHA-256) aggregate as build_content_manifest.py. These checks
identify supplied assets; they do not establish model accuracy or licensing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

TOOL_MODEL_ROLES = {
    "object_detection": "YOLO",
    "super_resolve_images": "SR",
    "retrieve_past_keyframes": "CLIP",
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def tool_asset_identity(tool: str) -> dict:
    role = TOOL_MODEL_ROLES[tool]
    variable = f"PANDA_{role}_MODEL"
    raw = os.getenv(variable + "_PATH", "").strip()
    path = Path(raw)
    if not raw or not (path.is_dir() if role == "CLIP" else path.is_file()):
        raise ValueError(
            f"{variable}_PATH must reference an existing {'directory' if role == 'CLIP' else 'checkpoint file'}"
        )
    expected = os.getenv(variable + "_SHA256", "")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{variable}_SHA256 must be an explicit lowercase SHA-256")
    if role == "CLIP":
        files = sorted(
            (
                p
                for p in path.rglob("*")
                if p.is_file() and ".git" not in p.relative_to(path).parts
            ),
            key=lambda p: p.relative_to(path).as_posix(),
        )
        if not files:
            raise ValueError("CLIP content directory is empty")
        digest = hashlib.sha256()
        for file in files:
            digest.update(
                f"{file.relative_to(path).as_posix()}\0{file.stat().st_size}\0{_hash_file(file)}\n".encode()
            )
        actual = digest.hexdigest()
    else:
        actual = _hash_file(path)
    if actual != expected:
        raise ValueError(f"{variable} content does not match the declared SHA-256")
    return {"model_id": path.name, "sha256": actual}


def verify_tool_assets() -> dict:
    """Fail before inference if any advertised paper-model tool lacks its asset."""
    return {tool: tool_asset_identity(tool) for tool in TOOL_MODEL_ROLES}


def loaded_tool_identity(tool: str) -> dict:
    """Bind a freshly loaded model; engineering-only unbound loads remain explicit."""
    variable = f"PANDA_{TOOL_MODEL_ROLES[tool]}_MODEL_SHA256"
    return tool_asset_identity(tool) if os.getenv(variable) else {}


def verify_tool_result(tool: str, result: dict, expected: dict) -> dict:
    if tool not in TOOL_MODEL_ROLES:
        return {}
    observed = result.get("asset_identity")
    if not expected.get(tool) or observed != expected[tool]:
        raise ValueError(f"Tool {tool} loaded asset identity does not match preflight")
    return dict(observed)


if __name__ == "__main__":
    print(json.dumps(verify_tool_assets(), sort_keys=True))
