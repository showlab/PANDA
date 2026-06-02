"""Lazy OpenAI-compatible VLM endpoint registry.

Importing this module only parses configuration. Network clients are constructed
on the first :func:`call_vlm` invocation and cached thereafter.
"""

from __future__ import annotations

import json
import math
import os
from importlib import resources
from numbers import Real
from typing import Any, Dict, List, Optional

from panda.config import load_environment

load_environment()


def _load_endpoint_configs() -> Dict[str, Dict[str, str]]:
    """Load packaged defaults, then apply environment overrides."""
    configs: Dict[str, Dict[str, str]] = {}

    try:
        import yaml

        text = (
            resources.files("panda.resources")
            .joinpath("vlm_endpoints.yaml")
            .read_text(encoding="utf-8")
        )
        packaged = yaml.safe_load(text) or {}
        if isinstance(packaged, dict):
            configs.update(packaged)
    except (ImportError, FileNotFoundError, ModuleNotFoundError, OSError, ValueError):
        pass

    env_config = os.getenv("VLM_ENDPOINTS", "").strip()
    if env_config:
        try:
            parsed = json.loads(env_config)
        except json.JSONDecodeError as exc:
            raise ValueError("VLM_ENDPOINTS must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("VLM_ENDPOINTS must map endpoint names to configurations")
        configs.update(parsed)

    for i in range(8):
        url = os.getenv(f"VLM_ENDPOINT_{i}", "").strip()
        if url:
            configs[f"qwen2_5_7b_v{i}"] = {
                "base_url": url,
                "api_key": os.getenv("VLM_API_KEY", "EMPTY"),
                "model": os.getenv("VLM_MODEL_PATH", ""),
            }

    normalized: Dict[str, Dict[str, str]] = {}
    for name, raw in configs.items():
        if isinstance(raw, str):
            raw = {"base_url": raw}
        if not isinstance(raw, dict):
            raise ValueError(f"Endpoint '{name}' must be a mapping or URL string")
        base_url = str(raw.get("base_url", "")).strip()
        if not base_url:
            raise ValueError(f"Endpoint '{name}' is missing base_url")
        normalized[name] = {
            "base_url": base_url,
            "api_key": str(raw.get("api_key", os.getenv("VLM_API_KEY", "EMPTY"))),
            "model": str(raw.get("model", "")),
        }
    return normalized


def _with_perception_aliases(
    configs: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    registry = {name: dict(cfg) for name, cfg in configs.items()}
    for name, cfg in list(registry.items()):
        if not name.startswith("per-"):
            registry.setdefault(f"per-{name}", dict(cfg))
    return registry


VLM_ENDPOINT_REGISTRY: Dict[str, Dict[str, str]] = _with_perception_aliases(
    _load_endpoint_configs()
)
# Backward-compatible public name. Values are endpoint configuration dictionaries,
# never live clients.
VLM_CLIENT_REGISTRY = VLM_ENDPOINT_REGISTRY
_CLIENT_CACHE: Dict[str, Any] = {}


def resolve_vlm_model(vlm_api: str, model_path: str = "") -> str:
    """Resolve the exact model selector used by the client, without a request."""
    if vlm_api not in VLM_ENDPOINT_REGISTRY:
        raise ValueError(f"Unknown vlm_api '{vlm_api}'")
    model = (
        model_path
        or os.getenv("VLM_MODEL_PATH", "").strip()
        or VLM_ENDPOINT_REGISTRY[vlm_api].get("model", "")
    )
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"No model configured for endpoint '{vlm_api}'")
    return model.strip()


def _make_client(base_url: str, api_key: str = "EMPTY"):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Package 'openai' is required for VLM calls. Install PANDA with "
            "the agents extra: pip install -e '.[agents]'"
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url)


def _get_client(name: str):
    if name not in VLM_ENDPOINT_REGISTRY:
        valid = ", ".join(sorted(VLM_ENDPOINT_REGISTRY))
        raise ValueError(f"Unknown vlm_api '{name}'. Valid options: {valid}")
    if name not in _CLIENT_CACHE:
        cfg = VLM_ENDPOINT_REGISTRY[name]
        _CLIENT_CACHE[name] = _make_client(cfg["base_url"], cfg.get("api_key", "EMPTY"))
    return _CLIENT_CACHE[name]


def _flatten_frames(frames: List) -> List[str]:
    flat: List[str] = []
    for frame in frames or []:
        if isinstance(frame, (list, tuple)):
            flat.extend(_flatten_frames(list(frame)))
        elif frame is not None:
            flat.append(str(frame))
    return flat


def _make_video_content(frames: List[str]) -> List[dict]:
    """Return one standard OpenAI-compatible PNG image block per sampled frame."""
    flat = _flatten_frames(frames)
    if not flat:
        raise ValueError("At least one video frame is required")
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{frame}"},
        }
        for frame in flat
    ]


def _build_messages(
    history_frames: List,
    current_window: List,
    system_prompt: str,
    user_prompt: str,
    perception_mode: bool = False,
    *,
    media_format: str = "image_sequence",
) -> List[dict]:
    if media_format not in ("image_sequence", "native_video"):
        raise ValueError(
            "media_format must be exactly 'image_sequence' or 'native_video'"
        )
    if media_format == "native_video":
        current_frames = _flatten_frames(current_window)
        if not current_frames:
            raise ValueError("At least one current video frame is required")
        content = []
        history = _flatten_frames(history_frames)
        if history and not perception_mode:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/jpeg;base64," + ",".join(history)},
                }
            )
        content.append(
            {
                "type": "video_url",
                "video_url": {
                    "url": "data:video/jpeg;base64," + ",".join(current_frames)
                },
            }
        )
    else:
        frames: List[str] = []
        if not perception_mode:
            frames.extend(_flatten_frames(history_frames))
        frames.extend(_flatten_frames(current_window))
        content = _make_video_content(frames)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                *content,
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def call_vlm(
    vlm_api: str,
    history_frames: Optional[List] = None,
    current_window: Optional[List] = None,
    system_prompt: str = "",
    user_prompt: str = "",
    temperature: float = 0.2,
    top_p: float = 0.5,
    model_path: str = "",
    *,
    media_format: str = "image_sequence",
    video_fps: Optional[float] = None,
) -> str:
    """Call one configured VLM endpoint with sampled visual content."""
    if vlm_api not in VLM_ENDPOINT_REGISTRY:
        valid = ", ".join(sorted(VLM_ENDPOINT_REGISTRY))
        raise ValueError(f"Unknown vlm_api '{vlm_api}'. Valid options: {valid}")

    model = resolve_vlm_model(vlm_api, model_path)

    fps_value: Optional[float] = None
    if media_format == "native_video":
        if isinstance(video_fps, bool) or not isinstance(video_fps, Real):
            raise ValueError("video_fps must be an explicit positive finite number")
        fps_value = float(video_fps)
        if not math.isfinite(fps_value) or fps_value <= 0:
            raise ValueError("video_fps must be an explicit positive finite number")

    messages = _build_messages(
        history_frames=history_frames or [],
        current_window=current_window or [],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        perception_mode=vlm_api.startswith("per-"),
        media_format=media_format,
    )
    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "seed": 2025,
    }
    if media_format == "native_video":
        request_kwargs["extra_body"] = {"mm_processor_kwargs": {"fps": fps_value}}

    response = _get_client(vlm_api).chat.completions.create(**request_kwargs)
    return response.choices[0].message.content
