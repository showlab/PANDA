"""Reflection-memory backends.

The opt-in JSONL backend uses deterministic token overlap. Paper mode
keeps per-video Long-CoM in graph state and loads a local semantic embedding
model only when historical retrieval is requested.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def finalize_paper_memory(
    state: dict,
    *,
    refined_reasoning: dict,
    window_index: int | None = None,
) -> list[dict]:
    """Return a new Long-CoM with one completed three-stage window record.

    The input state and all nested values remain untouched so graph snapshots from
    earlier nodes cannot be polluted.  Exhaustion handling may call this same
    helper after it has selected its final decision.
    """
    if state.get("memory_backend", "none") != "paper":
        raise ValueError("finalize_paper_memory requires memory_backend='paper'")
    current_index = state.get("window_index", 0)
    completed_index = current_index if window_index is None else window_index
    if (
        not isinstance(current_index, int)
        or isinstance(current_index, bool)
        or not isinstance(completed_index, int)
        or isinstance(completed_index, bool)
        or completed_index > current_index
    ):
        raise ValueError("cannot finalize a future window")

    memory = deepcopy(list(state.get("long_memory", [])))
    for previous in memory:
        previous_index = (
            previous.get("window_index") if isinstance(previous, dict) else None
        )
        if (
            not isinstance(previous_index, int)
            or isinstance(previous_index, bool)
            or previous_index >= completed_index
        ):
            raise ValueError("long_memory contains a current or future window record")

    results = [
        result
        for result in state.get("analysis_result", [])
        if isinstance(result, dict) and result.get("index") == completed_index
    ]
    initial_reasoning = results[0] if results else refined_reasoning
    reflections = [
        attempt
        for attempt in state.get("reflection_attempts", [])
        if isinstance(attempt, dict)
        and attempt.get("window_index", completed_index) == completed_index
    ]
    insufficient_reason = (
        reflections[0].get("insufficient_reason", "") if reflections else ""
    )
    record = deepcopy(
        {
            "window_index": completed_index,
            "initial_reasoning": initial_reasoning,
            "reflections": reflections,
            "refined_reasoning": refined_reasoning,
            "insufficient_reason": insufficient_reason,
        }
    )
    # Long-CoM lives in graph state and must remain checkpoint/JSON safe.
    json.dumps(record, ensure_ascii=False, allow_nan=False)
    memory.append(record)
    return memory


def _embedding_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_local_embedding_model():
    model_path = os.environ.get("EMBEDDING_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError("EMBEDDING_MODEL_PATH is required for paper Long-CoM")
    if not Path(model_path).exists():
        raise RuntimeError("EMBEDDING_MODEL_PATH must reference a local model")
    try:
        module = importlib.import_module("sentence_transformers")
        sentence_transformer = module.SentenceTransformer
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "sentence-transformers is required for paper Long-CoM"
        ) from exc
    try:
        return sentence_transformer(model_path, local_files_only=True)
    except Exception as exc:
        raise RuntimeError("failed to load the local embedding model") from exc


def _vector(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise RuntimeError("embedding model returned an invalid vector")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("embedding model returned an invalid vector") from exc
    if any(not math.isfinite(item) for item in vector):
        raise RuntimeError("embedding model returned a non-finite vector")
    return vector


def retrieve_paper_memory(
    long_memory: list[dict],
    query: Any,
    *,
    current_window_index: int,
    embedding_model: Any | None = None,
) -> list[dict]:
    """Return semantic Top-1 from completed prior windows using exact L2."""
    if not isinstance(current_window_index, int) or isinstance(
        current_window_index, bool
    ):
        raise ValueError("current_window_index must be an integer")

    historical = []
    for record in long_memory:
        if not isinstance(record, dict):
            raise ValueError("paper Long-CoM record must be an object")
        record_index = record.get("window_index")
        if not isinstance(record_index, int) or isinstance(record_index, bool):
            raise ValueError("paper Long-CoM record has an invalid window_index")
        if record_index > current_window_index:
            raise ValueError("paper Long-CoM contains a future window record")
        if record_index < current_window_index:
            required = {
                "window_index",
                "initial_reasoning",
                "reflections",
                "refined_reasoning",
                "insufficient_reason",
            }
            if set(record) != required:
                raise ValueError("paper Long-CoM record is incomplete")
            historical.append(record)
    if not historical:
        return []

    model = (
        embedding_model
        if embedding_model is not None
        else _load_local_embedding_model()
    )
    texts = [_embedding_text(query)] + [
        _embedding_text(record["insufficient_reason"]) for record in historical
    ]
    try:
        embeddings = model.encode(texts)
    except Exception as exc:
        raise RuntimeError("paper Long-CoM embedding failed") from exc
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    if not isinstance(embeddings, (list, tuple)) or len(embeddings) != len(texts):
        raise RuntimeError("embedding model returned an invalid batch")
    query_vector = _vector(embeddings[0])
    candidates = [_vector(value) for value in embeddings[1:]]
    if any(len(value) != len(query_vector) for value in candidates):
        raise RuntimeError("embedding model returned inconsistent dimensions")

    distances = [
        sum((left - right) ** 2 for left, right in zip(query_vector, candidate))
        for candidate in candidates
    ]
    top_index = min(range(len(distances)), key=distances.__getitem__)
    return [deepcopy(historical[top_index])]


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def append_memory(
    path: str | os.PathLike[str],
    *,
    insufficient_reason: Any,
    final_decision: dict,
    reflections: list,
    timestamp: str | None = None,
) -> dict:
    """Append one schema-checked record under an exclusive file lock."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "insufficient_reason": insufficient_reason,
        "final_decision": final_decision,
        "reflections": reflections,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (
            ImportError
        ):  # pragma: no cover - Windows fallback keeps append atomic per write
            fcntl = None
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def _valid_record(value: Any) -> bool:
    required = {
        "insufficient_reason",
        "final_decision",
        "reflections",
        "timestamp",
        "schema_version",
    }
    return (
        isinstance(value, dict)
        and set(value) == required
        and value.get("schema_version") == SCHEMA_VERSION
        and isinstance(value.get("final_decision"), dict)
        and isinstance(value.get("reflections"), list)
    )


def retrieve_memories(
    path: str | os.PathLike[str], query: str, *, top_k: int = 3
) -> tuple[list[dict], list[dict]]:
    """Return deterministic token-overlap matches and structured read failures."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    target = Path(path)
    if not target.is_file():
        return [], []

    query_tokens = _tokens(str(query))
    scored: list[tuple[int, int, dict]] = []
    failures: list[dict] = []
    with target.open("r", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        except ImportError:  # pragma: no cover
            fcntl = None
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                failures.append({"type": "memory_corrupt_line", "line": line_number})
                continue
            if not _valid_record(record):
                failures.append({"type": "memory_invalid_record", "line": line_number})
                continue
            overlap = len(query_tokens & _tokens(str(record["insufficient_reason"])))
            if overlap:
                scored.append((overlap, line_number, record))
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k]], failures
