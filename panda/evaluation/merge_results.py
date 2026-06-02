"""Strict chunk merge and complete-sequence offline scoring CLI."""

import argparse
import glob
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from panda.evaluation.io import atomic_write_json as _atomic_write_json
from panda.evaluation.metrics import compute_primary_metric, get_primary_metric
from panda.evaluation.postprocess import apply_score_smoothing
from panda.protocol import validate_smoothing
from panda.tool_assets import TOOL_MODEL_ROLES, verify_tool_result

_COVERAGE_KEYS = ("expected", "processed", "missing", "failed", "frames")
_RAW_PROTOCOL_FIELDS = (
    "sampling_fps",
    "window",
    "perception_sampling",
    "score_level",
    "smoothing",
)


def _non_negative_int(value: Any, field: str, chunk_idx: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Chunk {chunk_idx} coverage.{field} must be a non-negative integer"
        )
    return value


def _validate_protocols(ordered: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    protocols = [chunk.get("protocol") for chunk in ordered]
    if all(protocol is None for protocol in protocols):
        return None
    if any(not isinstance(protocol, dict) for protocol in protocols):
        raise ValueError("All chunks must carry a protocol object when any chunk does")
    first = protocols[0]
    missing = [field for field in _RAW_PROTOCOL_FIELDS if field not in first]
    if missing:
        raise ValueError(f"Chunk protocol is missing fields: {missing}")
    if first["score_level"] != "frame" or first["smoothing"] != "none":
        raise ValueError(
            "Chunk protocol must contain raw frame scores with smoothing=none"
        )
    for index, protocol in enumerate(protocols[1:], start=1):
        if protocol != first:
            raise ValueError(
                f"Chunk {ordered[index]['chunk_idx']} protocol is inconsistent"
            )
    return dict(first)


def _validate_model_identities(chunks: Sequence[Dict[str, Any]]) -> dict | None:
    identities = [chunk.get("model_identity") for chunk in chunks]
    declared = identities[0]
    if any(identity is not None for identity in identities):
        if (
            not isinstance(declared, dict)
            or set(declared) != {"planning", "reflection"}
            or any(
                not isinstance(value, str) or not value for value in declared.values()
            )
            or any(identity != declared for identity in identities)
        ):
            raise ValueError("Chunk model identities are missing or inconsistent")
    observed = dict(declared or {})
    for chunk in chunks:
        for record in chunk.get("provider_diagnostics", []):
            diagnostics = record.get("diagnostics", {})
            for role in ("planning", "reflection"):
                if role not in diagnostics:
                    continue
                model = diagnostics[role].get("model")
                if model is None:
                    continue  # Model identity is optional in provider diagnostics.
                if not isinstance(model, str) or not model:
                    raise ValueError("Invalid provider model diagnostic")
                if role in observed and observed[role] != model:
                    raise ValueError(f"Inconsistent actual {role} model")
                observed[role] = model
    return dict(declared) if declared is not None else None


def _paper_trace(video: dict, protocol: dict) -> dict:
    """Preserve the complete trajectory and bind its decisions to raw scores."""
    from panda.evaluation.runner import _frame_predictions

    fields = ("analysis_result", "long_memory", "exhausted_windows")
    if any(not isinstance(video.get(key), list) for key in fields):
        raise ValueError("Paper trace is missing or malformed")
    window = protocol.get("window")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("Paper trace requires a positive window size")
    n_frames = len(video["y_pred"])
    windows = [
        [None] * min(window, n_frames - offset) for offset in range(0, n_frames, window)
    ]
    try:
        predicted = _frame_predictions(
            windows,
            video["analysis_result"],
            exhausted_score_policy=protocol.get("exhausted_score_policy", "reject"),
        )
    except ValueError as exc:
        raise ValueError(f"Paper trace has invalid decisions: {exc}") from exc
    if predicted.tolist() != video["y_pred"]:
        raise ValueError("Paper trace final scores differ from raw predictions")
    latest = {row["index"]: row for row in video["analysis_result"]}
    initial = {}
    for row in video["analysis_result"]:
        initial.setdefault(row["index"], row)
    exhausted = sorted(
        index
        for index, row in latest.items()
        if row.get("resolution") == "reflection_exhausted"
    )
    if video["exhausted_windows"] != exhausted:
        raise ValueError("Paper trace exhaustion markers are inconsistent")
    memory = video["long_memory"]
    if len(memory) != len(windows):
        raise ValueError("Paper trace memory coverage is incomplete")
    for index, record in enumerate(memory):
        if (
            not isinstance(record, dict)
            or record.get("window_index") != index
            or record.get("initial_reasoning") != initial[index]
            or record.get("refined_reasoning") != latest[index]
        ):
            raise ValueError("Paper trace memory does not match analysis decisions")
        attempts = record.get("reflections")
        if not isinstance(attempts, list) or any(
            not isinstance(a, dict) or a.get("window_index") != index for a in attempts
        ):
            raise ValueError("Paper trace reflection window is inconsistent")
        if index in exhausted and len(attempts) != 3:
            raise ValueError(
                "Paper trace exhausted window lacks three reflection rounds"
            )
    return deepcopy(
        {key: video[key] for key in (*fields, "tool_diagnostics") if key in video}
    )


def merge_chunk_results(chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and concatenate an exact, complete set of raw chunk outputs."""
    if not chunks:
        raise ValueError("Cannot merge an empty chunk list")
    if any(not isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("Every chunk must be an object")

    first = chunks[0]
    dataset = first.get("dataset")
    metric = first.get("metric")
    run_id = first.get("run_id")
    source = first.get("source")
    chunk_total = first.get("chunk_total")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("Chunk dataset must be a non-empty string")
    if not isinstance(metric, str) or not metric:
        raise ValueError("Chunk metric must be a non-empty string")
    if metric != get_primary_metric(dataset):
        raise ValueError(
            f"Chunk metric {metric!r} is not the primary metric for {dataset!r}"
        )
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Chunk run_id must be a non-empty string")
    required_source = {
        "annotation_sha256",
        "vlm_model_sha256",
        "planning_provider",
        "reflection_provider",
        "rag_backend",
        "num_rules",
    }
    faiss_source = {
        "rag_embedding_model_id",
        "rag_embedding_model_sha256",
        "rag_rules_sha256",
        "rag_index_sha256",
    }
    expected_source = (
        required_source | faiss_source
        if isinstance(source, dict) and source.get("rag_backend") == "faiss"
        else required_source
    )
    if not isinstance(source, dict) or set(source) != expected_source:
        raise ValueError("Chunk source metadata is missing or invalid")
    digest_fields = ["annotation_sha256", "vlm_model_sha256"]
    if source.get("rag_backend") == "faiss":
        digest_fields.extend(
            ["rag_embedding_model_sha256", "rag_rules_sha256", "rag_index_sha256"]
        )
    for digest_field in digest_fields:
        digest = source.get(digest_field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(
                f"Chunk source {digest_field} must be a SHA-256 hex digest"
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(
                f"Chunk source {digest_field} must be a SHA-256 hex digest"
            ) from exc
    if source.get("rag_backend") == "faiss":
        model_id = source.get("rag_embedding_model_id")
        if (
            not isinstance(model_id, str)
            or not model_id
            or "/" in model_id
            or "\\" in model_id
        ):
            raise ValueError(
                "Chunk source rag_embedding_model_id must be a safe local model id"
            )
    if (
        isinstance(chunk_total, bool)
        or not isinstance(chunk_total, int)
        or chunk_total <= 0
    ):
        raise ValueError("chunk_total must be a positive integer")

    indices = [chunk.get("chunk_idx") for chunk in chunks]
    if (
        len(indices) != chunk_total
        or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        )
        or set(indices) != set(range(chunk_total))
    ):
        raise ValueError(
            f"chunk_idx values must be exactly {list(range(chunk_total))}; got {indices}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"chunk_idx values contain duplicates: {indices}")

    ordered = sorted(chunks, key=lambda chunk: chunk["chunk_idx"])
    protocol = _validate_protocols(ordered)
    merged_y_true: List[Any] = []
    merged_y_pred: List[Any] = []
    merged_videos: List[Dict[str, Any]] = []
    merged_provider_diagnostics: List[Any] = []
    merged_rag_diagnostics: List[Any] = []
    seen_video_names: set[str] = set()
    total_coverage = {key: 0 for key in _COVERAGE_KEYS}

    for chunk in ordered:
        chunk_idx = chunk["chunk_idx"]
        if chunk.get("dataset") != dataset:
            raise ValueError(f"Chunk {chunk_idx} dataset is inconsistent")
        if chunk.get("metric") != metric:
            raise ValueError(f"Chunk {chunk_idx} metric is inconsistent")
        if chunk.get("run_id") != run_id:
            raise ValueError(f"Chunk {chunk_idx} run_id is inconsistent")
        if chunk.get("source") != source:
            raise ValueError(f"Chunk {chunk_idx} source is inconsistent")
        if chunk.get("chunk_total") != chunk_total:
            raise ValueError(f"Chunk {chunk_idx} chunk_total is inconsistent")
        if chunk.get("partial") is not False:
            raise ValueError(f"Chunk {chunk_idx} is partial and cannot be merged")
        if chunk.get("status", "success") != "success" or chunk.get("failures", []):
            raise ValueError(f"Chunk {chunk_idx} failed and cannot be merged")
        for failure_field in ("stage_failures", "reflection_failures", "fail_logs"):
            failures = chunk.get(failure_field, [])
            if not isinstance(failures, list):
                raise ValueError(f"Chunk {chunk_idx} {failure_field} must be a list")
            if failures:
                raise ValueError(
                    f"Chunk {chunk_idx} has non-empty {failure_field} and cannot be merged"
                )
        for diagnostics_field, destination in (
            ("provider_diagnostics", merged_provider_diagnostics),
            ("rag_diagnostics", merged_rag_diagnostics),
        ):
            diagnostics = chunk.get(diagnostics_field, [])
            if not isinstance(diagnostics, list):
                raise ValueError(
                    f"Chunk {chunk_idx} {diagnostics_field} must be a list"
                )
            destination.extend(diagnostics)

        coverage = chunk.get("coverage")
        if not isinstance(coverage, dict):
            raise ValueError(f"Chunk {chunk_idx} coverage must be an object")
        values = {
            key: _non_negative_int(coverage.get(key), key, chunk_idx)
            for key in _COVERAGE_KEYS
        }
        if (
            values["expected"] <= 0
            or values["processed"] != values["expected"]
            or values["missing"] != 0
            or values["failed"] != 0
            or values["frames"] <= 0
        ):
            raise ValueError(f"Chunk {chunk_idx} coverage is incomplete: {values}")

        y_true = chunk.get("y_true")
        y_pred = chunk.get("y_pred")
        if not isinstance(y_true, list) or not isinstance(y_pred, list):
            raise ValueError(f"Chunk {chunk_idx} frame arrays must be lists")
        if not y_true or len(y_true) != len(y_pred):
            raise ValueError(
                f"Chunk {chunk_idx} frame arrays must be non-empty and equal length"
            )
        if values["frames"] != len(y_true):
            raise ValueError(
                f"Chunk {chunk_idx} coverage.frames={values['frames']} does not match arrays={len(y_true)}"
            )
        try:
            finite = (
                np.isfinite(np.asarray(y_true)).all()
                and np.isfinite(np.asarray(y_pred)).all()
            )
        except TypeError as exc:
            raise ValueError(f"Chunk {chunk_idx} frame arrays must be numeric") from exc
        if not finite:
            raise ValueError(f"Chunk {chunk_idx} frame arrays contain NaN or Inf")

        videos = chunk.get("videos")
        if not isinstance(videos, list) or len(videos) != values["processed"]:
            raise ValueError(
                f"Chunk {chunk_idx} videos must contain one entry per processed video"
            )
        chunk_video_y_true: List[Any] = []
        chunk_video_y_pred: List[Any] = []
        for video_idx, video in enumerate(videos):
            if not isinstance(video, dict):
                raise ValueError(
                    f"Chunk {chunk_idx} video {video_idx} must be an object"
                )
            video_name = video.get("video_name")
            if not isinstance(video_name, str) or not video_name:
                raise ValueError(
                    f"Chunk {chunk_idx} video {video_idx} video_name must be non-empty"
                )
            if video_name in seen_video_names:
                raise ValueError(f"Duplicate video_name across chunks: {video_name}")
            video_y_true = video.get("y_true")
            video_y_pred = video.get("y_pred")
            if not isinstance(video_y_true, list) or not isinstance(video_y_pred, list):
                raise ValueError(
                    f"Chunk {chunk_idx} video {video_name} arrays must be lists"
                )
            if not video_y_true or len(video_y_true) != len(video_y_pred):
                raise ValueError(
                    f"Chunk {chunk_idx} video {video_name} arrays must be non-empty and equal length"
                )
            try:
                video_finite = (
                    np.isfinite(np.asarray(video_y_true)).all()
                    and np.isfinite(np.asarray(video_y_pred)).all()
                )
            except TypeError as exc:
                raise ValueError(
                    f"Chunk {chunk_idx} video {video_name} arrays must be numeric"
                ) from exc
            if not video_finite:
                raise ValueError(
                    f"Chunk {chunk_idx} video {video_name} arrays contain NaN or Inf"
                )
            seen_video_names.add(video_name)
            chunk_video_y_true.extend(video_y_true)
            chunk_video_y_pred.extend(video_y_pred)
            merged_videos.append(
                {
                    "video_name": video_name,
                    "y_true": list(video_y_true),
                    "y_pred": list(video_y_pred),
                }
            )
            if protocol and protocol.get("profile", "engineering").startswith("paper-"):
                merged_videos[-1].update(_paper_trace(video, protocol))
        if chunk_video_y_true != y_true or chunk_video_y_pred != y_pred:
            raise ValueError(
                f"Chunk {chunk_idx} flattened arrays do not match its videos"
            )

        merged_y_true.extend(y_true)
        merged_y_pred.extend(y_pred)
        for key in _COVERAGE_KEYS:
            total_coverage[key] += values[key]

    if total_coverage["frames"] != len(merged_y_true):
        raise ValueError("Merged coverage does not match merged frame arrays")

    merged = {
        "dataset": dataset,
        "metric": metric,
        "run_id": run_id,
        "source": dict(source),
        "chunk_total": chunk_total,
        "partial": False,
        "coverage": total_coverage,
        "y_true": merged_y_true,
        "y_pred": merged_y_pred,
        "videos": merged_videos,
        "provider_diagnostics": merged_provider_diagnostics,
        "rag_diagnostics": merged_rag_diagnostics,
    }
    if protocol is not None:
        merged["protocol"] = protocol
    model_identity = _validate_model_identities(ordered)
    if model_identity is not None:
        merged["model_identity"] = model_identity
    if protocol and protocol.get("profile", "engineering").startswith("paper-"):
        assets = first.get("tool_assets")
        if (
            not isinstance(assets, dict)
            or set(assets) != set(TOOL_MODEL_ROLES)
            or any(chunk.get("tool_assets") != assets for chunk in chunks)
        ):
            raise ValueError("Paper tool asset identities are missing or inconsistent")
        for identity in assets.values():
            if (
                not isinstance(identity, dict)
                or not isinstance(identity.get("model_id"), str)
                or not identity["model_id"]
                or not isinstance(identity.get("sha256"), str)
                or len(identity["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in identity["sha256"])
            ):
                raise ValueError("Paper tool asset identity is invalid")
        for video in merged_videos:
            diagnostics = video.get("tool_diagnostics", [])
            if not isinstance(diagnostics, list):
                raise ValueError("Paper tool diagnostics must be a list")
            for record in diagnostics:
                if not isinstance(record, dict):
                    raise ValueError("Paper tool diagnostic is invalid")
                verify_tool_result(record.get("tool", ""), record, assets)
        merged["tool_assets"] = deepcopy(assets)
    return merged


def _load_chunks(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    if not paths:
        raise ValueError("No chunk result files were selected")
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Chunk result file list contains duplicates")
    chunks: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Chunk result file does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid chunk JSON at {path}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Chunk JSON must contain an object: {path}")
        chunks.append(payload)
    return chunks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge complete PANDA chunks and score offline"
    )
    parser.add_argument("files", nargs="*", help="Explicit chunk result JSON files")
    parser.add_argument(
        "--glob", dest="glob_pattern", help="Glob selecting chunk JSON files"
    )
    parser.add_argument(
        "--output", required=True, help="Atomic merged JSON output path"
    )
    parser.add_argument(
        "--smoothing", choices=["none", "mean", "median"], default="none"
    )
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument(
        "--smoothing-boundary", choices=["reflect", "nearest"], default="reflect"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if bool(args.files) == bool(args.glob_pattern):
        raise ValueError("Provide exactly one of explicit files or --glob")
    paths = [Path(path) for path in args.files]
    if args.glob_pattern:
        paths = [Path(path) for path in sorted(glob.glob(args.glob_pattern))]

    merged = merge_chunk_results(_load_chunks(paths))
    validate_smoothing(
        merged.get("protocol", {}).get("profile", "engineering"), args.smoothing
    )
    if "protocol" not in merged:
        raise ValueError(
            "Merge CLI requires runner chunks with explicit protocol metadata"
        )
    smoothed: List[float] = []
    for video in merged["videos"]:
        video_smoothed = apply_score_smoothing(
            video["y_pred"],
            method=args.smoothing,
            window=args.smoothing_window,
            boundary=args.smoothing_boundary,
        )
        video["y_pred_smoothed"] = video_smoothed
        smoothed.extend(video_smoothed)
    merged["raw_score"] = compute_primary_metric(
        merged["dataset"], merged["y_true"], merged["y_pred"]
    )
    merged["smoothed_score"] = compute_primary_metric(
        merged["dataset"], merged["y_true"], smoothed
    )
    merged["score"] = merged["smoothed_score"]
    merged["y_pred_smoothed"] = smoothed
    merged["protocol"] = {
        **merged["protocol"],
        "smoothing": args.smoothing,
        "smoothing_window": args.smoothing_window,
        "smoothing_boundary": args.smoothing_boundary,
        "smoothing_stage": "per_video_after_chunk_merge",
    }
    _atomic_write_json(Path(args.output), merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
