"""Fail-closed PANDA frame-level evaluation runner.

CLI: ``python -m panda.evaluation.runner``
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from panda.config import DATASET_CONFIGS
from panda.evaluation.annotation_parser import build_frame_label, parse_annotation_line
from panda.evaluation.io import atomic_write_json as _atomic_write_json
from panda.evaluation.metrics import compute_primary_metric, get_primary_metric
from panda.protocol import PROFILES, configure_profile
from panda.tool_assets import verify_tool_assets
from panda.tools.video_utils import extract_keyframes_base64, slice_frames_into_windows

_EVIDENCE_MAX_DEPTH = 6
_EVIDENCE_MAX_ITEMS = 50
_EVIDENCE_MAX_STRING = 512
_MEDIA_EVIDENCE_FULL_LIST_PATHS = frozenset(
    {
        ("analysis_media",),
        ("perception_media", "source_frame_indices"),
        ("perception_media", "source_timestamps_seconds"),
    }
)
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PANDA Evaluation Runner")
    parser.add_argument("--profile", choices=PROFILES, default="engineering")
    parser.add_argument(
        "--allow-model-substitution",
        action="store_true",
        help="Allow alternative model and provider selections",
    )
    parser.add_argument(
        "--video-root", required=True, help="Root directory containing dataset videos"
    )
    parser.add_argument(
        "--annotation-file", required=True, help="Path to annotation file"
    )
    parser.add_argument("--dataset", required=True, choices=["ucf", "xd", "ub", "csad"])
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--vlm-api", default="qwen2_5_7b_v0")
    parser.add_argument(
        "--run-id",
        default=os.getenv("PANDA_RUN_ID", ""),
        help="Shared non-secret identifier required when chunk-total > 1",
    )
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--chunk-total", type=int, default=1)
    parser.add_argument("--num-rules", type=int, default=5)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--perception-frames-num", type=int, default=300)
    parser.add_argument(
        "--perception-sampling",
        choices=["legacy_initial", "paper_uniform"],
        default="legacy_initial",
    )
    parser.add_argument(
        "--allow-perception-fallback",
        action="store_true",
        help="Explicitly permit Unknown perception metadata while recording stage failure",
    )
    parser.add_argument(
        "--rag-backend", choices=["lexical", "faiss"], default="lexical"
    )
    parser.add_argument("--reflection-memory-path", default="")
    parser.add_argument(
        "--planning-provider",
        choices=["gemini", "openai"],
        default=os.getenv("PLANNER_PROVIDER", "gemini"),
    )
    parser.add_argument(
        "--reflection-provider",
        choices=["gemini", "openai"],
        default=os.getenv("REFLECTION_PROVIDER", "gemini"),
    )
    parser.add_argument("--vlm-temperature", type=float, default=0.2)
    parser.add_argument("--vlm-top-p", type=float, default=0.5)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a score from processed videos, while marking coverage partial",
    )
    return parser


def _slice_frames_into_windows(frames: Sequence[Any], window: int) -> List[List[Any]]:
    """Compatibility wrapper around the CLI/evaluation shared window helper."""
    return slice_frames_into_windows(frames, window)


def _load_video_with_exact_tail(
    video_path: str, *, fps: float, window: int
) -> Tuple[List[List[Any]], float]:
    frames, video_fps = extract_keyframes_base64(video_path, fps=fps, duration=99999999)
    return _slice_frames_into_windows(frames, window), video_fps


def _chunk_rows(
    lines: Sequence[str], chunk_idx: int, chunk_total: int
) -> List[Tuple[int, str]]:
    if chunk_total <= 0:
        raise ValueError("chunk-total must be positive")
    if chunk_idx < 0 or chunk_idx >= chunk_total:
        raise ValueError("chunk-idx must satisfy 0 <= chunk-idx < chunk-total")
    quotient, remainder = divmod(len(lines), chunk_total)
    start = chunk_idx * quotient + min(chunk_idx, remainder)
    size = quotient + (1 if chunk_idx < remainder else 0)
    return [(index + 1, lines[index]) for index in range(start, start + size)]


def _resolve_video_path(video_root: str, video_name: str) -> Path:
    exact = Path(video_root) / video_name
    if exact.is_file():
        return exact
    if not Path(video_name).suffix:
        mp4 = Path(video_root) / f"{video_name}.mp4"
        if mp4.is_file():
            return mp4
    return exact


def _validated_sha256_env(name: str, *, required: bool = False) -> str:
    value = os.getenv(name, "").strip().lower()
    if not value:
        if required:
            raise ValueError(f"{name} must contain a SHA-256 content digest")
        return ""
    if len(value) != 64:
        raise ValueError(f"{name} must contain a SHA-256 content digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must contain a SHA-256 content digest") from exc
    return value


def _vlm_model_sha256(vlm_api: str) -> str:
    """Return the required verified model-content digest without persisting a path."""
    del vlm_api  # Endpoint names are not model-content identities.
    return _validated_sha256_env("VLM_MODEL_SHA256", required=True)


def _text_model_identity(args: Any) -> Dict[str, str]:
    """Snapshot model selectors without constructing clients or reading secrets."""
    models = {}
    for role in ("planning", "reflection"):
        provider = getattr(args, f"{role}_provider", "gemini")
        if provider == "gemini":
            models[role] = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        else:
            prefix = "PLANNER" if role == "planning" else "REFLECTION"
            models[role] = os.getenv(
                f"{prefix}_MODEL", os.getenv("PLANNER_MODEL", "unconfigured")
            )
    return models


def _rag_source_identity(args: Any) -> Dict[str, str]:
    """Return safe FAISS asset identity without persisting private absolute paths."""
    if getattr(args, "rag_backend", "lexical") != "faiss":
        return {}
    from panda.rag.retriever import DATASET_RULES

    dataset = str(args.dataset)
    specification = DATASET_RULES.get(dataset)
    if not specification or not specification.get("index"):
        raise FileNotFoundError(
            f"No dataset-specific FAISS RAG asset mapping for {dataset}"
        )
    rules_dir = Path(os.getenv("PANDA_RULES_DIR", ""))
    model_dir = Path(os.getenv("EMBEDDING_MODEL_PATH", ""))
    model_sha256 = _validated_sha256_env("EMBEDDING_MODEL_SHA256", required=True)
    if not rules_dir.is_dir() or not model_dir.is_dir():
        raise FileNotFoundError(
            "FAISS evaluation requires existing PANDA_RULES_DIR and EMBEDDING_MODEL_PATH directories"
        )
    rules_path = rules_dir / str(specification["json"])
    index_path = rules_dir / str(specification["index"])
    for path in (rules_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required dataset-specific RAG asset does not exist: {path.name}"
            )
    return {
        "rag_embedding_model_id": model_dir.name,
        "rag_embedding_model_sha256": model_sha256,
        "rag_rules_sha256": hashlib.sha256(rules_path.read_bytes()).hexdigest(),
        "rag_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }


def _frame_predictions(
    all_windows: Sequence[Sequence[Any]],
    analysis_result: Any,
    *,
    exhausted_score_policy: str = "reject",
) -> np.ndarray:
    if not isinstance(analysis_result, list):
        raise ValueError("analysis_result must be a list")
    if not all_windows or any(
        not isinstance(window, (list, tuple)) or len(window) == 0
        for window in all_windows
    ):
        raise ValueError("Video produced zero frames or an empty window")

    latest_results: Dict[int, Dict[str, Any]] = {}
    for result in analysis_result:
        if not isinstance(result, dict):
            raise ValueError("Each analysis result must be an object")
        index = result.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"Invalid analysis window index: {index!r}")
        if index < 0 or index >= len(all_windows):
            raise ValueError(f"Analysis window index out of range: {index}")
        latest_results[index] = result

    missing_windows = sorted(set(range(len(all_windows))) - set(latest_results))
    if missing_windows:
        raise ValueError(f"Missing analysis results for windows: {missing_windows}")

    latest_scores: Dict[int, float] = {}
    for index, result in latest_results.items():
        status = result.get("status")
        paper_exhausted = (
            exhausted_score_policy == "last_score"
            and status == "insufficient"
            and result.get("resolution") == "reflection_exhausted"
            and result.get("reflection_rounds") == 3
        )
        if status not in {"normal", "abnormal"} and not paper_exhausted:
            raise ValueError(
                f"Final analysis status for window {index} must be normal or abnormal"
            )
        raw_score = result.get("score")
        if isinstance(raw_score, (bool, np.bool_)) or not isinstance(
            raw_score, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"Invalid score for window {index}")
        score = float(raw_score)
        if not np.isfinite(score):
            raise ValueError(f"Non-finite score for window {index}")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score out of range for window {index}")
        latest_scores[index] = score

    return np.asarray(
        [
            latest_scores[index]
            for index, window in enumerate(all_windows)
            for _ in window
        ],
        dtype=np.float64,
    )


def _bounded_evidence(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
    _path: Tuple[str, ...] = (),
    _preserve_media_lists: bool = False,
) -> Any:
    """Return bounded, JSON-safe diagnostics without credential-shaped values."""
    if _depth >= _EVIDENCE_MAX_DEPTH:
        return "[TRUNCATED]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else "[NON_FINITE]"
    if isinstance(value, str):
        if len(value) <= _EVIDENCE_MAX_STRING:
            return value
        return value[:_EVIDENCE_MAX_STRING] + "...[TRUNCATED]"

    seen = _seen if _seen is not None else set()
    if isinstance(value, dict):
        identity = id(value)
        if identity in seen:
            return "[CYCLE]"
        seen.add(identity)
        output: Dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:_EVIDENCE_MAX_ITEMS]:
            safe_key = str(key)[:_EVIDENCE_MAX_STRING]
            normalized_key = safe_key.lower().replace("-", "_")
            if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
                output[safe_key] = "[REDACTED]"
            else:
                output[safe_key] = _bounded_evidence(
                    item,
                    _depth=_depth + 1,
                    _seen=seen,
                    _path=(*_path, safe_key),
                    _preserve_media_lists=_preserve_media_lists,
                )
        if len(items) > _EVIDENCE_MAX_ITEMS:
            output["__truncated_items__"] = len(items) - _EVIDENCE_MAX_ITEMS
        seen.remove(identity)
        return output

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return "[CYCLE]"
        seen.add(identity)
        max_items = (
            len(value)
            if _preserve_media_lists and _path in _MEDIA_EVIDENCE_FULL_LIST_PATHS
            else _EVIDENCE_MAX_ITEMS
        )
        output = [
            _bounded_evidence(
                item,
                _depth=_depth + 1,
                _seen=seen,
                _path=(*_path, "[]"),
                _preserve_media_lists=_preserve_media_lists,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            output.append(f"[TRUNCATED {len(value) - max_items} ITEMS]")
        seen.remove(identity)
        return output

    return f"[UNSERIALIZABLE:{type(value).__name__}]"


def _failure(
    line_number: int, line: str, kind: str, reason: str, video_name: str = ""
) -> Dict[str, Any]:
    return {
        "line_number": line_number,
        "line": line.strip(),
        "video_name": video_name,
        "kind": kind,
        "reason": reason,
    }


def run_evaluation(
    args: argparse.Namespace,
    *,
    graph: Any,
    video_loader: Callable[..., Tuple[Sequence[Sequence[Any]], float]] | None = None,
) -> Tuple[Dict[str, Any], int]:
    """Run one evaluation chunk and atomically persist its complete evidence."""
    args = configure_profile(args)
    tool_assets = verify_tool_assets() if args.profile != "engineering" else {}
    loader = video_loader or _load_video_with_exact_tail
    dataset_config = DATASET_CONFIGS[args.dataset]
    user_query = dataset_config["query_template"]
    metric_name = get_primary_metric(args.dataset)
    model_identity = _text_model_identity(args)
    rag_source_identity = _rag_source_identity(args)
    annotation_path = Path(args.annotation_file)
    annotation_bytes = annotation_path.read_bytes()
    annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    lines = annotation_bytes.decode("utf-8").splitlines()
    run_id = str(getattr(args, "run_id", "") or "").strip()
    if args.chunk_total > 1 and not run_id:
        raise ValueError("--run-id is required when chunk-total > 1")
    if not run_id:
        run_id = f"single-{annotation_sha256[:12]}"
    rows = _chunk_rows(lines, args.chunk_idx, args.chunk_total)

    coverage = {
        "expected": len(rows),
        "processed": 0,
        "missing": 0,
        "failed": 0,
        "frames": 0,
    }
    failures: List[Dict[str, Any]] = []
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    videos: List[Dict[str, Any]] = []
    stage_failures: List[Dict[str, Any]] = []
    fail_logs: List[Dict[str, Any]] = []
    reflection_failures: List[Dict[str, Any]] = []
    provider_diagnostics: List[Dict[str, Any]] = []
    rag_diagnostics: List[Dict[str, Any]] = []

    if not rows:
        failures.append(
            _failure(0, "", "empty_chunk", "Chunk contains no annotation rows")
        )

    for line_number, line in rows:
        video_name = ""
        try:
            annotation = parse_annotation_line(line, dataset=args.dataset)
            video_name = annotation["video_name"]
            video_path = _resolve_video_path(args.video_root, video_name)
            if not video_path.is_file():
                coverage["missing"] += 1
                failures.append(
                    _failure(
                        line_number,
                        line,
                        "missing",
                        f"Video file not found: {video_path}",
                        video_name,
                    )
                )
                continue

            all_windows, video_fps = loader(
                str(video_path), fps=args.fps, window=args.window
            )
            total_frames = sum(len(window) for window in all_windows)
            if total_frames == 0:
                coverage["failed"] += 1
                failures.append(
                    _failure(
                        line_number,
                        line,
                        "zero_frames",
                        "Video produced zero sampled frames",
                        video_name,
                    )
                )
                continue

            from panda.state import make_default_state

            state = make_default_state(
                video_path=str(video_path),
                tool_assets=tool_assets,
                user_query=user_query,
                dataset_name=args.dataset,
                vlm_api=args.vlm_api,
                vlm_models=dict(getattr(args, "vlm_models", {})),
                num_rules=args.num_rules,
                all_windows=all_windows,
                perception_frames_num=args.perception_frames_num,
                perception_sampling=getattr(
                    args, "perception_sampling", "legacy_initial"
                ),
                allow_perception_fallback=getattr(
                    args, "allow_perception_fallback", False
                ),
                rag_backend=getattr(args, "rag_backend", "lexical"),
                reflection_memory_path=getattr(args, "reflection_memory_path", ""),
                planning_provider=getattr(args, "planning_provider", "gemini"),
                reflection_provider=getattr(args, "reflection_provider", "gemini"),
                profile=args.profile,
                memory_backend=args.memory_backend,
                exhausted_score_policy=args.exhausted_score_policy,
                output_dir=args.output_dir,
                fps=args.fps,
                source_video_fps=video_fps,
                vlm_temperature=args.vlm_temperature,
                vlm_top_p=args.vlm_top_p,
            )
            result = graph.invoke(state, {"recursion_limit": 10000})
            if not isinstance(result, dict):
                raise ValueError("Graph result must be an object")
            result_stage_failures = result.get("stage_failures", [])
            if not isinstance(result_stage_failures, list) or any(
                not isinstance(failure, dict) for failure in result_stage_failures
            ):
                raise ValueError("stage_failures must be a list of objects")
            stage_failures.extend(
                {"video_name": video_name, **_bounded_evidence(failure)}
                for failure in result_stage_failures
            )
            result_fail_logs = result.get("fail_logs", [])
            if not isinstance(result_fail_logs, list):
                raise ValueError("fail_logs must be a list")
            fail_logs.extend(
                {"video_name": video_name, "entry": _bounded_evidence(entry)}
                for entry in result_fail_logs
            )
            result_reflection_failures = result.get("reflection_failures", [])
            if not isinstance(result_reflection_failures, list) or any(
                not isinstance(failure, dict) for failure in result_reflection_failures
            ):
                raise ValueError("reflection_failures must be a list of objects")
            reflection_failures.extend(
                {"video_name": video_name, **_bounded_evidence(failure)}
                for failure in result_reflection_failures
            )
            result_provider_diagnostics = result.get("provider_diagnostics", {})
            if not isinstance(result_provider_diagnostics, dict):
                raise ValueError("provider_diagnostics must be an object")
            provider_diagnostics.append(
                {
                    "video_name": video_name,
                    "diagnostics": _bounded_evidence(
                        result_provider_diagnostics, _preserve_media_lists=True
                    ),
                }
            )
            result_rag_diagnostics = result.get("rag_diagnostics", {})
            if not isinstance(result_rag_diagnostics, dict):
                raise ValueError("rag_diagnostics must be an object")
            rag_diagnostics.append(
                {
                    "video_name": video_name,
                    "diagnostics": _bounded_evidence(result_rag_diagnostics),
                }
            )
            if args.profile != "engineering" and (
                result_stage_failures or result_fail_logs or result_reflection_failures
            ):
                raise ValueError(
                    "Paper execution contains provider/tool/stage failures; this is not valid reflection exhaustion"
                )
            prediction = _frame_predictions(
                all_windows,
                result.get("analysis_result"),
                exhausted_score_policy=args.exhausted_score_policy,
            )
            truth = build_frame_label(
                total_frames,
                annotation["segments"],
                fps_original=video_fps,
                fps_sampled=args.fps,
            )
            if len(truth) != len(prediction):
                raise ValueError(
                    f"Frame arrays differ: y_true={len(truth)}, y_pred={len(prediction)}"
                )

            video_y_true = truth.tolist()
            video_y_pred = prediction.tolist()
            y_true_all.extend(video_y_true)
            y_pred_all.extend(video_y_pred)
            videos.append(
                {
                    "video_name": video_name,
                    "y_true": video_y_true,
                    "y_pred": video_y_pred,
                }
            )
            if args.profile != "engineering":
                videos[-1].update(
                    analysis_result=result.get("analysis_result", []),
                    long_memory=result.get("long_memory", []),
                    exhausted_windows=result.get("exhausted_windows", []),
                    tool_diagnostics=result.get("tool_diagnostics", []),
                )
            coverage["processed"] += 1
            coverage["frames"] += len(truth)
        except Exception as exc:
            coverage["failed"] += 1
            failures.append(
                _failure(
                    line_number,
                    line,
                    "exception",
                    f"{type(exc).__name__}: {exc}",
                    video_name,
                )
            )

    complete_coverage = (
        coverage["expected"] > 0
        and coverage["processed"] == coverage["expected"]
        and coverage["missing"] == 0
        and coverage["failed"] == 0
        and coverage["frames"] > 0
    )
    payload: Dict[str, Any] = {
        "dataset": args.dataset,
        "model_identity": model_identity,
        "metric": metric_name,
        "run_id": run_id,
        "source": {
            "annotation_sha256": annotation_sha256,
            "vlm_model_sha256": _vlm_model_sha256(args.vlm_api),
            "planning_provider": getattr(args, "planning_provider", "gemini"),
            "reflection_provider": getattr(args, "reflection_provider", "gemini"),
            "rag_backend": getattr(args, "rag_backend", "lexical"),
            "num_rules": args.num_rules,
            **rag_source_identity,
        },
        "chunk_idx": args.chunk_idx,
        "chunk_total": args.chunk_total,
        "partial": not complete_coverage,
        "coverage": coverage,
        "failures": failures,
        "stage_failures": stage_failures,
        "fail_logs": fail_logs,
        "reflection_failures": reflection_failures,
        "provider_diagnostics": provider_diagnostics,
        "rag_diagnostics": rag_diagnostics,
        "protocol": {
            "user_query": user_query,
            "sampling_fps": float(args.fps),
            "window": args.window,
            "perception_sampling": getattr(
                args, "perception_sampling", "legacy_initial"
            ),
            "score_level": "frame",
            "smoothing": "none",
        },
        "n_frames": coverage["frames"],
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "videos": videos,
    }

    metric_succeeded = False
    if args.profile != "engineering":
        payload["tool_assets"] = tool_assets
        payload["protocol"].update(
            profile=args.profile,
            memory_backend=args.memory_backend,
            exhausted_score_policy=args.exhausted_score_policy,
            perception_frames_num=args.perception_frames_num,
            vlm_temperature=args.vlm_temperature,
            vlm_top_p=args.vlm_top_p,
            vlm_seed=2025,
            model_substitution=args.model_substitution,
            vlm_models={
                role: Path(model).name if Path(model).is_absolute() else model
                for role, model in args.vlm_models.items()
            },
            vlm_media_contract="native_video_base_v1_still_tool_evidence",
        )
    # A worker may legitimately contain only normal (or only abnormal) frames.
    # Raw-complete shards are mergeable; only the full run needs both classes.
    metric_deferred = (
        complete_coverage and args.chunk_total > 1 and len(set(y_true_all)) == 1
    )
    if metric_deferred:
        payload["metric_status"] = "deferred_single_class"
    elif complete_coverage or (args.allow_partial and coverage["frames"] > 0):
        try:
            payload["score"] = compute_primary_metric(
                args.dataset, y_true_all, y_pred_all
            )
            metric_succeeded = True
        except ValueError as exc:
            failures.append(_failure(0, "", "metric", str(exc)))

    accepted = (metric_succeeded or metric_deferred) and (
        complete_coverage or args.allow_partial
    )
    payload["status"] = (
        "partial"
        if accepted and payload["partial"]
        else ("success" if accepted else "failed")
    )
    result_path = Path(args.output_dir) / f"results_gpu{args.chunk_idx}.json"
    _atomic_write_json(result_path, payload)
    return payload, 0 if accepted else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from panda.graph.graph import build_graph

    payload, exit_code = run_evaluation(args, graph=build_graph())
    print(
        json.dumps(
            {
                "dataset": payload["dataset"],
                "chunk": f"{payload['chunk_idx']}/{payload['chunk_total']}",
                "status": payload["status"],
                "coverage": payload["coverage"],
                "score": payload.get("score"),
            },
            indent=2,
        )
    )
    print(
        f"Results saved to: {Path(args.output_dir) / f'results_gpu{args.chunk_idx}.json'}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
