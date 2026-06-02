"""Perception agent with explicit, frame-count-based sampling modes."""

import time

from panda.agents.utils import safe_load_json
from panda.paper_prompts import perception_prompt
from panda.tools.video_utils import extract_perception_keyframes_base64
from panda.tools.vlm_client import call_vlm

_UNKNOWN_ENV_INFO = {
    "Scene overview": "Unknown",
    "Weather/Lighting": "Unknown",
    "Video quality": "Unknown",
    "Potential anomalies": "Unknown",
}
_REQUIRED_ENV_INFO = tuple(_UNKNOWN_ENV_INFO)


def _validate_perception_contract(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != set(_REQUIRED_ENV_INFO):
        raise ValueError(
            "Perception provider must return the complete environment object"
        )
    if any(
        not isinstance(value[key], str) or not value[key].strip()
        for key in _REQUIRED_ENV_INFO
    ):
        raise ValueError("Perception provider fields must be non-empty strings")
    return dict(value)


def build_perception_prompt(user_query: str) -> tuple:
    """Build the perception system and user prompts."""
    json_template = """{
  "Scene overview": "...",
  "Weather/Lighting": "...",
  "Video quality": "...",
  "Potential anomalies": "..."
}"""

    system_prompt = f"""You are an expert in video anomaly perception. Your task is to perform an initial understanding and analysis of the provided video frames based on the user's specified needs.

[User Requirement]:
{user_query}

Please respond by completing the following four aspects:
1. Scene overview: Describe the environment shown in the video (e.g., shopping mall, office, street, surveillance corridor) and briefly summarize the main activities or events observed.
2. Weather/Lighting: Describe the visual lighting conditions (e.g., daytime/nighttime, sunny/overcast, bright/dim).
3. Video Quality:  Comment on the overall quality of the video (e.g., clear, blurry, noisy, low frame rate, low light).
4. Potential Anomalies: Based on the visual content, what types of abnormal events are likely to occur in this video (e.g., Fighting, Stealing).

Please return your output strictly in the following JSON format:
{json_template}
***Please return only the raw JSON object, without any markdown code block or explanation.
Do not wrap it in triple backticks (```), do not explain anything else."""

    user_prompt = "Above is the video frame sequence, please analyze it carefully and follow the format to output the video environment information."
    return system_prompt, user_prompt


def perception_agent(state: dict) -> dict:
    """Extract an explicit perception frame sample and call the VLM.

    Failures propagate by default so the evaluation runner fails the video. The
    optional Unknown fallback is available only through the explicit
    ``allow_perception_fallback`` state flag and leaves structured evidence.
    """
    video_path = state["video_path"]
    max_frames = state.get("perception_frames_num", 300)
    sampling = state.get("perception_sampling", "legacy_initial")
    user_query = state["user_query"]
    paper_mode = state.get("profile", "engineering").startswith("paper-")
    prompt_builder = perception_prompt if paper_mode else build_perception_prompt
    system_prompt, user_prompt = prompt_builder(user_query)
    start = time.time()

    media_kwargs = {}
    media_updates = {}
    try:
        sample = extract_perception_keyframes_base64(
            video_path,
            fps=1.0,
            max_frames=max_frames,
            sampling=sampling,
            **({"return_metadata": True} if paper_mode else {}),
        )
        if paper_mode:
            keyframes, timing = sample
            media_kwargs = {
                "media_format": "native_video",
                "video_fps": timing["video_fps"],
            }
            media_updates = {
                "provider_diagnostics": {
                    **state.get("provider_diagnostics", {}),
                    "perception_media": {"format": "native_video", **timing},
                }
            }
        else:
            keyframes = sample
        result = call_vlm(
            vlm_api=f"per-{state['vlm_api']}",
            model_path=state.get("vlm_models", {}).get("perception", ""),
            current_window=keyframes,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=state.get("vlm_temperature", 0.2),
            top_p=state.get("vlm_top_p", 0.5),
            **media_kwargs,
        )
        value = safe_load_json(result)
        if paper_mode and isinstance(value, dict):
            aliases = {
                "Scene Overview": "Scene overview",
                "Weather Condition": "Weather/Lighting",
                "Video Quality": "Video quality",
                "Potential Anomalies": "Potential anomalies",
            }
            if set(value) == set(aliases):
                value = {aliases[key]: item for key, item in value.items()}
        env_info = _validate_perception_contract(value)
    except Exception as exc:
        if not state.get("allow_perception_fallback", False):
            raise
        keyframes = []
        env_info = dict(_UNKNOWN_ENV_INFO)
        failure = {
            "stage": "perception",
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "fallback": "unknown_env_info",
        }
        stage_failures = [*state.get("stage_failures", []), failure]
        print(f"perception_time: {time.time() - start:.1f}s")
        return {
            **state,
            "keyframes": keyframes,
            "env_info": env_info,
            "perception_prompt": system_prompt,
            "stage_failures": stage_failures,
        }

    print(f"perception_time: {time.time() - start:.1f}s")
    return {
        **state,
        **media_updates,
        "keyframes": keyframes,
        "env_info": env_info,
        "perception_prompt": system_prompt,
    }
