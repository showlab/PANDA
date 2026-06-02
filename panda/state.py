"""PANDA Agent state — unified TypedDict and default factory."""

from typing import Any, Dict, List, Optional, TypedDict


class VadAgentState(TypedDict, total=False):
    """PANDA Agent state — unified definition for graph and agents."""

    # Core input
    video_path: str
    user_query: str

    # Video processing
    keyframes: List[str]
    all_windows: List[List[str]]
    current_window: List[str]
    enhance_frames: List[str]
    planning_preprocessing_window_index: Optional[int]
    planning_preprocessed_frames: List[str]
    planning_preprocessing_failures: List[Dict[str, Any]]

    # Agent outputs
    env_info: Dict[str, Any]
    planning_info: Dict[str, Any]
    analysis_result: List[Dict[str, Any]]
    current_prompt: str

    # Reflection
    insufficient_reason: str
    retry_count: int
    fail_logs: List[Any]
    enhancement_prompt_blocks: List[str]
    tools_already_used: List[str]
    reflection_attempts: List[Dict[str, Any]]
    reflection_failures: List[Dict[str, Any]]
    stage_failures: List[Dict[str, Any]]
    reflection_memory_path: str
    memory_backend: str
    long_memory: List[Dict[str, Any]]
    profile: str
    exhausted_score_policy: str
    exhausted_windows: List[int]
    tool_assets: Dict[str, Any]
    tool_diagnostics: List[Dict[str, Any]]
    output_dir: str
    planning_provider: str
    reflection_provider: str
    provider_diagnostics: Dict[str, Any]

    # Window control
    window_index: int

    # Config parameters
    vlm_api: str
    vlm_models: Dict[str, str]
    fps: float
    source_video_fps: Optional[float]
    mode: str
    chunk_idx: str
    dataset_name: str
    num_rules: int
    perception_frames_num: int
    perception_sampling: str
    allow_perception_fallback: bool
    vlm_temperature: float
    vlm_top_p: float

    # Prompt cache
    perception_prompt: str
    planning_prompt: str
    analysis_prompt: str
    reflection_prompt: str

    # RAG
    anomaly_rules: Dict[str, Any]
    rag_backend: str
    rag_diagnostics: Dict[str, Any]


def make_default_state(
    video_path: str,
    user_query: str,
    **overrides,
) -> dict:
    """Create a fully initialized state dict with sensible defaults."""
    state = {
        "video_path": video_path,
        "user_query": user_query,
        "keyframes": [],
        "all_windows": [],
        "current_window": [],
        "enhance_frames": [],
        "planning_preprocessing_window_index": None,
        "planning_preprocessed_frames": [],
        "planning_preprocessing_failures": [],
        "env_info": {},
        "planning_info": {},
        "analysis_result": [],
        "current_prompt": "",
        "insufficient_reason": "",
        "retry_count": 0,
        "fail_logs": [],
        "enhancement_prompt_blocks": [],
        "tools_already_used": [],
        "reflection_attempts": [],
        "reflection_failures": [],
        "stage_failures": [],
        "reflection_memory_path": "",
        "memory_backend": "none",
        "long_memory": [],
        "profile": "engineering",
        "exhausted_score_policy": "reject",
        "exhausted_windows": [],
        "tool_assets": {},
        "tool_diagnostics": [],
        "output_dir": "",
        "planning_provider": "gemini",
        "reflection_provider": "gemini",
        "provider_diagnostics": {},
        "window_index": 0,
        "vlm_api": "qwen2_5_7b_v0",
        "vlm_models": {},
        "fps": 1.0,
        "source_video_fps": None,
        "mode": "eval",
        "chunk_idx": "0",
        "dataset_name": "ucf",
        "num_rules": 5,
        "perception_frames_num": 300,
        "perception_sampling": "legacy_initial",
        "allow_perception_fallback": False,
        "vlm_temperature": 0.2,
        "vlm_top_p": 0.5,
        "perception_prompt": "",
        "planning_prompt": "",
        "analysis_prompt": "",
        "reflection_prompt": "",
        "anomaly_rules": {},
        "rag_backend": "lexical",
        "rag_diagnostics": {},
    }
    state.update(overrides)
    return state
