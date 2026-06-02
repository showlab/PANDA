"""Analysis agent — per-window anomaly scoring via VLM."""

import math
import time
from pathlib import Path

from panda.agents.utils import load_provider_object
from panda.memory import append_memory, finalize_paper_memory
from panda.paper_prompts import analysis_prompt as paper_analysis_prompt
from panda.tool_assets import verify_tool_result
from panda.tools.tool_manager import PLANNING_PREPROCESSING_ALIASES, TOOL_MANAGER
from panda.tools.vlm_client import call_vlm


def build_analysis_prompt(
    query,
    planning_info,
    anomaly_rules,
    history_result_prompt,
    enhancement_prompt_blocks,
    index,
):
    """Build analysis system/user prompts."""
    json_template = """{
      "score": float,
      "status": "normal/abnormal/insufficient",
      "reason": "A detailed explanation of your reasoning..."
    }"""
    out_template = """{
      "score": 0.95,
      "status": "abnormal",
      "reason": "Multiple individuals are chasing and grappling with each other while holding objects, which strongly suggests a fighting behavior."
    }"""

    if enhancement_prompt_blocks:
        formatted = []
        for idx, block in enumerate(enhancement_prompt_blocks, 1):
            formatted.append(f"[Reflection #{idx}]\n{block.strip()}")
        formatted_enh = "\n\n".join(formatted)
    else:
        formatted_enh = "No additional enhancement or reflection information available."

    system_prompt = f"""You are a highly skilled expert in video anomaly detection.

[User Requirement]
{query}

[Historical Detection Info]
{history_result_prompt if history_result_prompt.strip() else "No reliable historical detection information available."}

[Current Video Clip Index]
Clip {index}

[Potential Anomalies]
{planning_info.get("potential_anomalies", "")}

[Anomaly Detection Rules]
{anomaly_rules}

[Heuristic Prompt]
{planning_info.get("heuristic_prompts", "")}

[Enhancement and Reflection Information]
{formatted_enh}

Please strictly output your response in the following JSON format:
{json_template}

Your analysis should follow three steps:
1. Describe the main visible actions and interactions between people or objects in the scene.
2. Assess how strongly these actions match any known abnormal event patterns.
3. Based on your assessment, assign a score and label, and explain your reasoning clearly.

Scoring Guidelines:
- A score close to 1.0 indicates clear and confident abnormal behavior.
- A score close to 0.0 indicates clearly normal behavior.
- A score near 0.5 means uncertain, ambiguous behavior or mixed signals.

Example Output:
{out_template}

***Please return only the raw JSON object, without any markdown code block or explanation."""

    user_prompt = "Please analyze the input video frames and output your result in the specified JSON format."
    return system_prompt.strip(), user_prompt.strip()


class PlanningPreprocessingError(ValueError):
    def __init__(self, *, kind: str, step_index: int, requested_step: str):
        super().__init__(kind)
        self.kind = kind
        self.step_index = step_index
        self.requested_step = requested_step


def _planning_preprocessing_steps(declaration) -> list[str]:
    if not isinstance(declaration, str) or not declaration.strip():
        return []
    steps = []
    for step_index, raw_step in enumerate(declaration.split("->")):
        name = PLANNING_PREPROCESSING_ALIASES.get(raw_step.strip().casefold())
        if name is None:
            raise PlanningPreprocessingError(
                kind="unsupported_step",
                step_index=step_index,
                requested_step=raw_step.strip(),
            )
        steps.append(name)
    return steps


def _preprocess_window(state: dict, frames: list[str], window_index: int) -> list[str]:
    current = list(frames)
    output_dir = str(state.get("output_dir", "")).strip()
    for step_index, tool_name in enumerate(
        _planning_preprocessing_steps(
            state.get("planning_info", {}).get("preprocessing", "")
        )
    ):
        try:
            result = TOOL_MANAGER.execute(
                tool_name,
                frames=current,
                save_dir=str(
                    Path(output_dir)
                    / f"window_{window_index}"
                    / "planning_preprocess"
                    / tool_name
                ),
            )
            if state.get("profile", "engineering").startswith("paper-"):
                if result.get("frame_failures"):
                    raise ValueError("Paper preprocessing returned frame failures")
                identity = verify_tool_result(
                    tool_name, result, state.get("tool_assets", {})
                )
                if identity:
                    state["tool_diagnostics"].append(
                        {
                            "tool": tool_name,
                            "window_index": window_index,
                            "stage": "planning_preprocessing",
                            "asset_identity": identity,
                        }
                    )
        except Exception:
            raise PlanningPreprocessingError(
                kind="tool_execution_error",
                step_index=step_index,
                requested_step=tool_name,
            ) from None
        enhanced = result.get("enhanced_frames") if isinstance(result, dict) else None
        if (
            not isinstance(enhanced, list)
            or len(enhanced) != len(current)
            or not enhanced
            or any(not isinstance(frame, str) or not frame for frame in enhanced)
        ):
            raise PlanningPreprocessingError(
                kind="invalid_tool_output",
                step_index=step_index,
                requested_step=tool_name,
            )
        current = list(enhanced)
    return current


def analysis_agent(state: dict) -> dict:
    """Analysis node: score current video window."""
    state = {**state, "tool_diagnostics": list(state.get("tool_diagnostics", []))}
    window_index = state.get("window_index", 0)
    all_windows = state.get("all_windows", [])
    enhance_frames = state.get("enhance_frames", [])
    enhancement_blocks = state.get("enhancement_prompt_blocks", [])
    planning_info = state.get("planning_info", {})
    anomaly_rules = state.get("anomaly_rules", {})

    if window_index >= len(all_windows):
        return state

    # Planning preprocessing runs once per window. Reflection retries continue from
    # that cached base (or a later reflection enhancement), never from raw frames.
    preprocessing_updates = {}
    if state.get("planning_preprocessing_window_index") == window_index:
        preprocessed_base = list(state.get("planning_preprocessed_frames", []))
    else:
        try:
            preprocessed_base = _preprocess_window(
                state, list(all_windows[window_index]), window_index
            )
        except PlanningPreprocessingError as exc:
            planning_failures = list(state.get("planning_preprocessing_failures", []))
            failure = {
                "type": exc.kind,
                "window_index": window_index,
                "step_index": exc.step_index,
                "requested_step": exc.requested_step,
            }
            planning_failures.append(failure)
            stage_failures = list(state.get("stage_failures", []))
            stage_failures.append({"stage": "planning_preprocessing", **failure})
            return {
                **state,
                "window_index": window_index + 1,
                "planning_preprocessing_failures": planning_failures,
                "stage_failures": stage_failures,
            }
        preprocessing_updates = {
            "planning_preprocessing_window_index": window_index,
            "planning_preprocessed_frames": list(preprocessed_base),
        }
    # Work on a new list: reflection/retrieval must never mutate state-owned windows.
    current_window = list(enhance_frames if enhance_frames else preprocessed_base)

    # History frames (past 5 windows)
    import itertools

    history_frames = list(
        itertools.chain.from_iterable(
            all_windows[max(0, window_index - 5) : window_index]
        )
    )

    # History results text: each prior window contributes only its latest retry.
    history_result_prompt = ""
    history_start = max(0, window_index - 5)
    latest_by_index = {}
    for result in state.get("analysis_result", []):
        if not isinstance(result, dict):
            continue
        result_index = result.get("index")
        if (
            isinstance(result_index, int)
            and not isinstance(result_index, bool)
            and history_start <= result_index < window_index
        ):
            latest_by_index[result_index] = result
    for i in range(history_start, window_index):
        result = latest_by_index.get(i)
        if result is not None:
            status = result.get("status", "insufficient")
            if status not in ("normal", "abnormal", "insufficient", "violative"):
                status = "insufficient"
            try:
                score = float(result.get("score", 0.5))
            except (TypeError, ValueError):
                score = 0.5
            if not math.isfinite(score):
                score = 0.5
            score = min(1.0, max(0.0, score))
            reason = result.get("reason", "unavailable")
            if isinstance(reason, dict):
                reason = reason.get("reason", "unavailable")
            if not isinstance(reason, (str, int, float)):
                reason = "unavailable"
            reason = " ".join(str(reason).split())[:500] or "unavailable"
            history_result_prompt += (
                f"Clip {i} result: status={status}; score={score:g}; reason={reason}\n"
            )

    prompt_builder = (
        paper_analysis_prompt
        if state.get("profile", "engineering").startswith("paper-")
        else build_analysis_prompt
    )
    system_prompt, user_prompt = prompt_builder(
        state["user_query"],
        planning_info,
        anomaly_rules,
        history_result_prompt,
        enhancement_blocks,
        window_index,
    )

    media_kwargs = {}
    media_evidence = None
    if state.get("profile", "engineering").startswith("paper-"):
        from panda.tools.video_utils import _positive_finite_number

        sampling_fps = _positive_finite_number(state.get("fps", 1.0), "fps")
        video_fps = sampling_fps
        source_fps = state.get("source_video_fps")
        if source_fps is not None:
            source_fps = _positive_finite_number(source_fps, "source_video_fps")
            video_fps = min(video_fps, source_fps)
        media_kwargs = {"media_format": "native_video", "video_fps": video_fps}
        media_evidence = {
            "format": "native_video",
            "history_frames": len(history_frames),
            "current_frames": len(current_window),
            "video_fps": video_fps,
            "source_fps": source_fps,
            "time_policy": (
                "source_limited_sampling"
                if video_fps < sampling_fps
                else "sampling_target_grid"
            ),
        }
        if enhance_frames:
            # Tool outputs can be crops or unordered retrieved frames. Without
            # source timestamps, preserve their existing still-image semantics
            # rather than claiming that adjacent items are one second apart.
            media_kwargs = {"media_format": "image_sequence"}
            media_evidence.update(
                format="image_sequence",
                video_fps=None,
                time_policy="unknown_tool_evidence_no_temporal_claim",
            )

    start = time.time()
    result = call_vlm(
        vlm_api=state["vlm_api"],
        model_path=state.get("vlm_models", {}).get("analysis", ""),
        history_frames=history_frames,
        current_window=current_window,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=state.get("vlm_temperature", 0.2),
        top_p=state.get("vlm_top_p", 0.5),
        **media_kwargs,
    )

    text = load_provider_object(result, "Analysis")
    status = text.get("status")
    allowed_statuses = ("normal", "abnormal", "insufficient")
    if not state.get("profile", "engineering").startswith("paper-"):
        allowed_statuses += ("violative",)  # Engineering-mode compatibility alias.
    if not isinstance(status, str) or status not in allowed_statuses:
        raise ValueError("Analysis requires an explicit valid status")
    score = text.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("Analysis requires a finite numeric score in [0, 1]")
    if not isinstance(text.get("reason"), str) or not text["reason"].strip():
        raise ValueError("Analysis requires a non-empty reason string")

    new_result = {
        "index": window_index,
        "status": status,
        "score": score,
        "reason": text,
    }
    media_updates = {}
    if media_evidence is not None:
        # Transport observations are not reasoning. Keep them out of Short-CoM
        # prompts and Long-CoM records/embeddings so logging cannot change retrieval.
        diagnostics = dict(state.get("provider_diagnostics", {}))
        records = list(diagnostics.get("analysis_media", []))
        records.append({"window_index": window_index, **media_evidence})
        diagnostics["analysis_media"] = records
        media_updates["provider_diagnostics"] = diagnostics

    print(
        f"analysis_time: {time.time() - start:.1f}s  window={window_index}  status={status}"
    )

    is_insufficient = status == "insufficient"
    attempts = list(state.get("reflection_attempts", []))
    failures = list(state.get("reflection_failures", []))
    remaining_attempts = attempts
    memory_updates = {}

    # Paper Long-CoM is per-video graph state and must never touch the shared
    # cross-video JSONL path. Every completed window gets all three stages,
    # including an explicit empty reflection stage for direct decisions.
    if not is_insufficient and state.get("memory_backend", "none") == "paper":
        try:
            memory_updates["long_memory"] = finalize_paper_memory(
                state, refined_reasoning=new_result, window_index=window_index
            )
            remaining_attempts = []
        except Exception as exc:
            if state.get("profile", "engineering").startswith("paper-"):
                raise
            failures.append(
                {
                    "type": "memory_write_error",
                    "window_index": window_index,
                    "error_type": type(exc).__name__,
                }
            )
    # JSONL persistence remains opt-in per state/video. A final decision remains
    # authoritative even if memory persistence fails.
    elif not is_insufficient and attempts:
        memory_path = str(state.get("reflection_memory_path", "")).strip()
        if memory_path:
            first_reason = attempts[0].get(
                "insufficient_reason", state.get("insufficient_reason", "")
            )
            try:
                append_memory(
                    memory_path,
                    insufficient_reason=first_reason,
                    final_decision=new_result,
                    reflections=attempts,
                )
                remaining_attempts = []
            except Exception as exc:
                failures.append(
                    {
                        "type": "memory_write_error",
                        "window_index": window_index,
                        "error_type": type(exc).__name__,
                    }
                )
        else:
            # Memory is disabled, but attempts are scoped to this completed window.
            remaining_attempts = []

    return {
        **state,
        **preprocessing_updates,
        **memory_updates,
        **media_updates,
        "current_window": current_window,
        "analysis_prompt": system_prompt,
        "analysis_result": state.get("analysis_result", []) + [new_result],
        "insufficient_reason": new_result["reason"] if is_insufficient else "",
        "retry_count": state.get("retry_count", 0) if is_insufficient else 0,
        "enhancement_prompt_blocks": []
        if not is_insufficient
        else state.get("enhancement_prompt_blocks", []),
        "enhance_frames": []
        if not is_insufficient
        else state.get("enhance_frames", []),
        "tools_already_used": []
        if not is_insufficient
        else state.get("tools_already_used", []),
        "reflection_attempts": remaining_attempts,
        "reflection_failures": failures,
        "window_index": window_index if is_insufficient else window_index + 1,
    }
