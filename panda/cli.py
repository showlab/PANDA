"""PANDA CLI entry point: python -m panda.cli"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from panda.protocol import PROFILES, configure_profile
from panda.state import make_default_state
from panda.tool_assets import verify_tool_assets
from panda.tools.video_utils import load_video_and_slice


def main():
    """Run anomaly detection on one video and save its results."""
    parser = argparse.ArgumentParser(description="PANDA Video Anomaly Detection Agent")
    parser.add_argument("--profile", choices=PROFILES, default="engineering")
    parser.add_argument("--allow-model-substitution", action="store_true")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--query",
        type=str,
        default="Detect abnormal events in this video.",
        help="User query describing what to detect",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ucf",
        choices=["ucf", "xd", "ub", "csad"],
        help="Dataset name (for RAG rules)",
    )
    parser.add_argument(
        "--vlm-api", type=str, default="qwen2_5_7b_v0", help="VLM endpoint key"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-rules", type=int, default=5, help="Number of RAG rules to retrieve"
    )
    parser.add_argument("--window", type=int, default=5, help="Window size in frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate")
    parser.add_argument(
        "--perception-sampling",
        choices=["legacy_initial", "paper_uniform"],
        default="legacy_initial",
    )
    parser.add_argument(
        "--allow-perception-fallback",
        action="store_true",
        help="Permit Unknown perception metadata while recording a stage failure",
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

    args = configure_profile(parser.parse_args())
    tool_assets = verify_tool_assets() if args.profile != "engineering" else {}

    from panda.graph.graph import build_graph

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and slice video
    all_windows, video_fps = load_video_and_slice(
        args.video,
        fps=args.fps,
        window=args.window,
    )

    # Build initial state
    state = make_default_state(
        video_path=args.video,
        tool_assets=tool_assets,
        user_query=args.query,
        dataset_name=args.dataset,
        vlm_api=args.vlm_api,
        vlm_models=dict(getattr(args, "vlm_models", {})),
        num_rules=args.num_rules,
        all_windows=all_windows,
        fps=args.fps,
        perception_sampling=args.perception_sampling,
        source_video_fps=video_fps,
        perception_frames_num=getattr(args, "perception_frames_num", 300),
        profile=args.profile,
        memory_backend=args.memory_backend,
        exhausted_score_policy=args.exhausted_score_policy,
        allow_perception_fallback=args.allow_perception_fallback,
        rag_backend=args.rag_backend,
        reflection_memory_path=args.reflection_memory_path,
        planning_provider=args.planning_provider,
        reflection_provider=args.reflection_provider,
        output_dir=args.output_dir,
    )

    # Build and run graph
    graph = build_graph()
    print(f"Starting PANDA analysis on {args.video} ({len(all_windows)} windows)...")
    result = graph.invoke(state, {"recursion_limit": 10000})

    # Persist a machine-readable single-video result.
    result_path = Path(args.output_dir) / f"{Path(args.video).stem}.json"
    payload = {
        "video_path": args.video,
        "dataset": args.dataset,
        "query": args.query,
        "vlm_api": args.vlm_api,
        "video_fps": float(video_fps),
        "window_size": args.window,
        "num_windows": len(all_windows),
        "sampling_fps": args.fps,
        "analysis_result": result.get("analysis_result", []),
        "fail_logs": result.get("fail_logs", []),
        "stage_failures": result.get("stage_failures", []),
        "reflection_failures": result.get("reflection_failures", []),
        "provider_diagnostics": result.get("provider_diagnostics", {}),
        "protocol": {
            "sampling_fps": args.fps,
            "window": args.window,
            "perception_sampling": args.perception_sampling,
            "rag_backend": args.rag_backend,
            "score_level": "window",
        },
    }
    if args.profile != "engineering":
        payload["tool_assets"] = tool_assets
        payload["tool_diagnostics"] = result.get("tool_diagnostics", [])
        payload["long_memory"] = result.get("long_memory", [])
        payload["exhausted_windows"] = result.get("exhausted_windows", [])
        payload["protocol"].update(
            profile=args.profile,
            perception_frames_num=args.perception_frames_num,
            memory_backend=args.memory_backend,
            exhausted_score_policy=args.exhausted_score_policy,
            model_substitution=args.model_substitution,
            vlm_models={
                role: Path(model).name if Path(model).is_absolute() else model
                for role, model in args.vlm_models.items()
            },
            vlm_media_contract="native_video_base_v1_still_tool_evidence",
        )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=result_path.parent,
            prefix=f".{result_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, result_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print("\n=== Detection Complete ===")
    for r in payload["analysis_result"]:
        print(
            f"  Window {r['index']}: status={r['status']} score={r.get('score', 'N/A')}"
        )
    if payload["fail_logs"]:
        print(f"\nFailed windows: {payload['fail_logs']}")
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
