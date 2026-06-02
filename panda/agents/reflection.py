"""Reflection agent: select and execute one fail-closed information tool."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from panda.agents.utils import load_provider_object
from panda.memory import finalize_paper_memory, retrieve_memories, retrieve_paper_memory
from panda.paper_prompts import reflection_prompt as paper_reflection_prompt
from panda.providers import request_json_text
from panda.tool_assets import verify_tool_result
from panda.tools.tool_manager import TOOL_MANAGER

MAX_REFLECTION_RETRIES = 3


def _reason_text(reason: Any) -> str:
    if isinstance(reason, str):
        return reason
    return json.dumps(reason, ensure_ascii=False, sort_keys=True)


def _memory_for_prompt(state: dict) -> tuple[list[dict], list[dict]]:
    if state.get("memory_backend", "none") == "paper":
        return (
            retrieve_paper_memory(
                state.get("long_memory", []),
                _reason_text(state.get("insufficient_reason", "")),
                current_window_index=int(state.get("window_index", 0)),
            ),
            [],
        )
    path = str(state.get("reflection_memory_path", "")).strip()
    if not path:
        return [], []
    return retrieve_memories(
        path, _reason_text(state.get("insufficient_reason", "")), top_k=3
    )


def build_reflection_prompt(
    state: dict,
    *,
    memory_records: list[dict] | None = None,
) -> tuple[str, str]:
    """Build a prompt from the live tool registry and safe memory records."""
    env_info = state.get("env_info", {})
    anomaly_rules = state.get("anomaly_rules", {})
    planning_info = state.get("planning_info", {})
    analysis_result = state.get("analysis_result", [])
    reason = _reason_text(state.get("insufficient_reason", ""))
    tools_used = list(state.get("tools_already_used", []))

    if memory_records is None:
        memory_records, _ = _memory_for_prompt(state)
    memory_context = (
        json.dumps(memory_records, ensure_ascii=False, sort_keys=True)
        if memory_records
        else "No matching reflection memory."
    )
    historical = (
        "\n".join(
            f"  * Clip {item.get('index', '?')}: {item.get('reason', '')}"
            for item in analysis_result[-5:]
        )
        or "No historical results."
    )
    tool_description_text = "\n".join(
        f"- {name}: {TOOL_MANAGER.get_tool_description(name)}"
        for name in TOOL_MANAGER.list_all_tools()
    )
    if state.get("profile", "engineering").startswith("paper-"):
        return paper_reflection_prompt(state, memory_records, tool_description_text)

    system_prompt = f"""You are a reflection module in a video anomaly detector.
The current analysis is insufficient. Select at most ONE most critical tool.

[Context]
- User requirement: {state.get("user_query", "")}
- Scene: {env_info.get("Scene overview", "Unknown")}
- Weather/Lighting: {env_info.get("Weather/Lighting", "Unknown")}
- Video quality: {env_info.get("Video quality", "Unknown")}
- Potential anomalies: {planning_info.get("potential_anomalies", "")}
- Anomaly rules: {anomaly_rules}
- Historical results:
{historical}
- Insufficient reason: {reason}
- Tools already used: {tools_used}

[Safe reflection memory]
{memory_context}

[Available tools]
{tool_description_text}

Return only one raw JSON object with this shape:
{{
  "reason": "why information is insufficient",
  "tools_to_use": [{{"tool_name": "one available unused tool", "query": "required query or empty"}}],
  "new_prompt_hint": "guidance for the next analysis",
  "new_anomaly_rule": "optional new rule"
}}
Never request more than one tool and never repeat a used tool."""
    human = "Reflect on the insufficient evidence and provide one safe next step."
    return system_prompt.strip(), human


def _request_reflection(
    system_prompt: str,
    human_prompt: str,
    *,
    provider: str | None = None,
    diagnostics: dict | None = None,
) -> dict:
    """Call the configured reflection provider and parse its JSON response."""
    result_text = request_json_text(
        role="reflection",
        provider=provider,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
        diagnostics=diagnostics,
    )
    return load_provider_object(result_text, "Reflection")


_DEFAULT_REQUEST_REFLECTION = _request_reflection


def _failure(kind: str, window_index: int, **fields: Any) -> dict:
    failure = {"type": kind, "window_index": window_index}
    failure.update({key: value for key, value in fields.items() if value is not None})
    return failure


def _base_frames(state: dict, window_index: int) -> list[str]:
    enhanced = state.get("enhance_frames", [])
    if enhanced:
        return list(enhanced)
    current = state.get("current_window", [])
    if current:
        return list(current)
    windows = state.get("all_windows", [])
    if 0 <= window_index < len(windows):
        return list(windows[window_index])
    return []


def _tool_arguments(
    state: dict,
    tool_name: str,
    query: str,
    base_frames: list[str],
    parameters: dict | None = None,
) -> tuple[dict | None, dict | None]:
    """Build only schema-advertised arguments or return a structured failure."""
    window_index = int(state.get("window_index", 0))
    hint = TOOL_MANAGER.get_tool_parameter_hint(tool_name)
    inputs = set(hint.get("inputs", {}))
    required = set(hint.get("required", inputs))
    args: dict[str, Any] = {}
    if "zoom_factor" in inputs:
        factor = (parameters or {}).get("zoom_factor")
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(factor)
            or factor <= 0
        ):
            return None, _failure(
                "tool_missing_zoom_factor", window_index, tool_name=tool_name
            )
        args["zoom_factor"] = factor

    if "query" in inputs:
        if "query" in required and not query:
            return None, _failure(
                "tool_missing_query", window_index, tool_name=tool_name
            )
        if query:
            args["query"] = query

    historical = "current_window_index" in inputs
    all_windows = state.get("all_windows", [])
    if historical:
        if window_index < 1 or not all_windows[:window_index]:
            return None, _failure(
                "tool_missing_history", window_index, tool_name=tool_name
            )
        current_window = (
            all_windows[window_index]
            if window_index < len(all_windows)
            else base_frames
        )
        if not current_window:
            return None, _failure(
                "tool_missing_history", window_index, tool_name=tool_name
            )
        flat_frames = [
            frame for window in all_windows[:window_index] for frame in list(window)
        ]
        args["current_window_index"] = window_index
        if "window_size" in inputs:
            args["window_size"] = len(all_windows[0])
        if "lookback" in inputs:
            args["lookback"] = int(state.get("reflection_lookback", 3))
        if "top_k" in inputs:
            args["top_k"] = int(state.get("reflection_top_k", 3))
        if "frames" in inputs:
            args["frames"] = flat_frames
    elif "frames" in inputs:
        if not base_frames:
            return None, _failure(
                "tool_missing_frames", window_index, tool_name=tool_name
            )
        args["frames"] = list(base_frames)

    if "save_dir" in inputs:
        output_dir = str(state.get("output_dir", "")).strip()
        if not output_dir:
            return None, _failure(
                "tool_missing_output_dir", window_index, tool_name=tool_name
            )
        args["save_dir"] = str(Path(output_dir) / f"window_{window_index}" / tool_name)
    return args, None


def _enhancement_block(summary: str, reflection: dict) -> str:
    parts = []
    if summary.strip():
        parts.append(summary.strip())
    hint = str(
        reflection.get("new_heuristic_prompt", reflection.get("new_prompt_hint", ""))
    ).strip()
    if hint:
        parts.append(f"***new prompt hint*** {hint}")
    rule = str(reflection.get("new_anomaly_rule", "")).strip()
    if rule:
        parts.append(f"***new anomaly rule*** {rule}")
    return "\n".join(parts)


def _validate_paper_reflection(reflection: Any) -> None:
    """Reject incomplete provider objects before tools, prompts, or memory change."""
    if not isinstance(reflection, dict):
        raise ValueError("Reflection requires a JSON object")
    for field in ("reason", "new_anomaly_rule", "new_heuristic_prompt"):
        if field not in reflection or not isinstance(reflection[field], str):
            raise ValueError(f"Reflection requires a {field} string")
    if not reflection["reason"].strip():
        raise ValueError("Reflection requires a non-empty reason")
    selections = reflection.get("tools_to_use")
    if not isinstance(selections, list) or len(selections) > 1:
        raise ValueError(
            "Reflection requires a tools_to_use list with at most one tool"
        )
    for selection in selections:
        if (
            not isinstance(selection, dict)
            or not isinstance(selection.get("tool_name"), str)
            or not selection["tool_name"].strip()
            or not isinstance(selection.get("query"), str)
            or ("params" in selection and not isinstance(selection["params"], dict))
        ):
            raise ValueError("Reflection tool selection has invalid fields")


def reflection_agent(state: dict) -> dict:
    """Execute at most the provider's first selected, not-yet-used tool."""
    window_index = int(state.get("window_index", 0))
    retry_count = int(state.get("retry_count", 0)) + 1
    failures = list(state.get("reflection_failures", []))

    if retry_count > MAX_REFLECTION_RETRIES:
        if state.get("exhausted_score_policy", "reject") == "last_score":
            results = list(state.get("analysis_result", []))
            positions = [
                i
                for i, value in enumerate(results)
                if value.get("index") == window_index
            ]
            if not positions:
                raise ValueError("Exhausted window has no final analysis score")
            position = positions[-1]
            final = dict(results[position])
            score = final.get("score")
            if (
                final.get("status") != "insufficient"
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError(
                    "Exhausted window requires a finite final insufficient score"
                )
            final.update(
                resolution="reflection_exhausted",
                reflection_rounds=MAX_REFLECTION_RETRIES,
            )
            results[position] = final
            memory_updates = {}
            if state.get("memory_backend") == "paper":
                memory_updates["long_memory"] = finalize_paper_memory(
                    state, refined_reasoning=final, window_index=window_index
                )
            return {
                **state,
                **memory_updates,
                "analysis_result": results,
                "window_index": window_index + 1,
                "retry_count": 0,
                "enhance_frames": [],
                "enhancement_prompt_blocks": [],
                "tools_already_used": [],
                "reflection_attempts": [],
                "exhausted_windows": list(state.get("exhausted_windows", []))
                + [window_index],
            }
        failures.append(
            _failure("retry_exhausted", window_index, attempts=retry_count - 1)
        )
        fail_logs = list(state.get("fail_logs", []))
        fail_logs.append((window_index, state.get("insufficient_reason", "unknown")))
        return {
            **state,
            "window_index": window_index + 1,
            "retry_count": 0,
            "enhance_frames": [],
            "fail_logs": fail_logs,
            "enhancement_prompt_blocks": [],
            "tools_already_used": [],
            "reflection_attempts": [],
            "reflection_failures": failures,
        }

    try:
        memory_records, memory_failures = _memory_for_prompt(state)
    except Exception as exc:
        if state.get("memory_backend", "none") == "paper":
            raise
        memory_records = []
        memory_failures = [
            _failure("memory_read_error", window_index, error_type=type(exc).__name__)
        ]
    failures.extend(memory_failures)
    system_prompt, human = build_reflection_prompt(state, memory_records=memory_records)

    provider_diagnostic: dict[str, Any] = {}
    failure_kind = "reflection_provider_error"
    try:
        # Custom request callbacks accept two prompt arguments. Pass provider
        # configuration and diagnostics only to the built-in implementation.
        if _request_reflection is _DEFAULT_REQUEST_REFLECTION:
            reflection = _request_reflection(
                system_prompt,
                human,
                provider=state.get("reflection_provider"),
                diagnostics=provider_diagnostic,
            )
        else:
            reflection = _request_reflection(system_prompt, human)
        if state.get("profile", "engineering").startswith("paper-"):
            failure_kind = "reflection_schema_error"
            _validate_paper_reflection(reflection)
    except Exception as exc:
        failures.append(
            _failure(failure_kind, window_index, error_type=type(exc).__name__)
        )
        provider_diagnostics = dict(state.get("provider_diagnostics", {}))
        if provider_diagnostic:
            provider_diagnostics["reflection"] = provider_diagnostic
        return {
            **state,
            "reflection_prompt": system_prompt,
            "retry_count": retry_count,
            "reflection_failures": failures,
            "provider_diagnostics": provider_diagnostics,
        }

    attempts = list(state.get("reflection_attempts", []))
    attempts.append(
        {
            "window_index": window_index,
            "insufficient_reason": state.get("insufficient_reason", ""),
            "reflection_result": reflection,
        }
    )
    used = list(state.get("tools_already_used", []))
    base_frames = _base_frames(state, window_index)
    tool_diagnostics = list(state.get("tool_diagnostics", []))
    # An unchanged base clip is still native video, not newly produced tool stills.
    final_frames = list(state.get("enhance_frames", []))
    summary = ""

    selections = reflection.get("tools_to_use", [])
    if not isinstance(selections, list):
        failures.append(_failure("invalid_tool_selection", window_index))
        selections = []
    if len(selections) > 1:
        failures.append(
            _failure("multiple_tools_requested", window_index, count=len(selections))
        )

    if selections:
        selection = selections[0] if isinstance(selections[0], dict) else {}
        tool_name = str(selection.get("tool_name", "")).strip()
        query = str(selection.get("query", "")).strip()
        if not tool_name or TOOL_MANAGER.get_tool_function(tool_name) is None:
            failures.append(
                _failure("tool_unknown", window_index, tool_name=tool_name or None)
            )
        elif tool_name in used:
            failures.append(
                _failure("tool_repeated", window_index, tool_name=tool_name)
            )
        else:
            args, argument_failure = _tool_arguments(
                state,
                tool_name,
                query,
                base_frames,
                parameters=selection.get("params")
                if isinstance(selection.get("params"), dict)
                else {},
            )
            if argument_failure:
                failures.append(argument_failure)
            else:
                try:
                    tool_result = TOOL_MANAGER.execute(tool_name, **(args or {}))
                    identity = {}
                    if state.get("profile", "engineering").startswith("paper-"):
                        identity = verify_tool_result(
                            tool_name, tool_result, state.get("tool_assets", {})
                        )
                        if tool_result.get("frame_failures"):
                            raise ValueError("Paper tool returned frame failures")
                        enhanced = tool_result.get("enhanced_frames")
                        if enhanced is not None and (
                            not isinstance(enhanced, list)
                            or len(enhanced) != len(base_frames)
                            or any(
                                not isinstance(frame, str) or not frame
                                for frame in enhanced
                            )
                        ):
                            raise ValueError(
                                "Paper visual tool must preserve every input frame in order"
                            )
                    tool_frames = list(base_frames)
                    if "enhanced_frames" in tool_result:
                        enhanced = tool_result["enhanced_frames"]
                        if not isinstance(enhanced, list) or not enhanced:
                            raise ValueError("tool returned no enhanced frames")
                        tool_frames = list(enhanced)
                    if "past_keyframes" in tool_result:
                        retrieved = tool_result["past_keyframes"]
                        if not isinstance(retrieved, list) or not retrieved:
                            raise ValueError("tool returned no past keyframes")
                        tool_frames = tool_frames + list(retrieved)
                    if (
                        "enhanced_frames" in tool_result
                        or "past_keyframes" in tool_result
                    ):
                        final_frames = tool_frames
                    summary = str(tool_result.get("summary", ""))
                    used.append(tool_name)
                    if (
                        state.get("profile", "engineering").startswith("paper-")
                        and identity
                    ):
                        tool_diagnostics.append(
                            {
                                "tool": tool_name,
                                "window_index": window_index,
                                "stage": "reflection",
                                "asset_identity": identity,
                            }
                        )
                except Exception as exc:
                    failures.append(
                        _failure(
                            "tool_execution_error",
                            window_index,
                            tool_name=tool_name,
                            error_type=type(exc).__name__,
                        )
                    )

    block = _enhancement_block(summary, reflection)
    blocks = list(state.get("enhancement_prompt_blocks", []))
    if block:
        blocks.append(block)
    provider_diagnostics = dict(state.get("provider_diagnostics", {}))
    if provider_diagnostic:
        provider_diagnostics["reflection"] = provider_diagnostic
    return {
        **state,
        "reflection_prompt": system_prompt,
        "enhancement_prompt_blocks": blocks,
        "retry_count": retry_count,
        "enhance_frames": final_frames,
        "tools_already_used": used,
        "reflection_attempts": attempts,
        "tool_diagnostics": tool_diagnostics,
        "reflection_failures": failures,
        "provider_diagnostics": provider_diagnostics,
    }
