"""Strict registry and execution boundary for PANDA reflection tools."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Mapping

TOOL_KNOWLEDGE = {
    "object_detection": "Detect people, objects, and anomaly-relevant actions with an explicitly configured YOLO model.",
    "denoise_images": "Denoise the current base64 frames to improve visual clarity.",
    "deblur_images": "Sharpen blurred current frames using unsharp masking.",
    "enhance_brightness": "Improve brightness and local contrast in current frames.",
    "super_resolve_images": "Recover image details with an explicitly configured Real-ESRGAN model.",
    "retrieve_past_keyframes": "Retrieve query-relevant frames strictly from earlier video windows.",
    "zoom_images": "Magnify current frames with bicubic interpolation; supply an explicit numeric zoom_factor in params.",
    "web_search": "Search the web for contextual information using an environment-provided Tavily credential.",
}

TOOL_PARAMETER_HINTS = {
    "object_detection": {
        "inputs": {
            "frames": "list of base64 frames",
            "save_dir": "explicit output directory",
        },
        "required": ["frames", "save_dir"],
    },
    "denoise_images": {
        "inputs": {
            "frames": "list of base64 frames",
            "save_dir": "explicit output directory",
        },
        "required": ["frames", "save_dir"],
    },
    "deblur_images": {
        "inputs": {
            "frames": "list of base64 frames",
            "save_dir": "explicit output directory",
        },
        "required": ["frames", "save_dir"],
    },
    "enhance_brightness": {
        "inputs": {
            "frames": "list of base64 frames",
            "save_dir": "explicit output directory",
        },
        "required": ["frames", "save_dir"],
    },
    "super_resolve_images": {
        "inputs": {
            "frames": "list of base64 frames",
            "save_dir": "explicit output directory",
        },
        "required": ["frames", "save_dir"],
    },
    "retrieve_past_keyframes": {
        "inputs": {
            "frames": "flat list of all video frames",
            "query": "semantic retrieval query",
            "save_dir": "explicit output directory",
            "lookback": "number of previous windows",
            "top_k": "maximum returned frames",
            "current_window_index": "current window index",
            "window_size": "frames per window",
        },
        "required": [
            "frames",
            "query",
            "save_dir",
            "current_window_index",
            "window_size",
        ],
    },
    "web_search": {
        "inputs": {"query": "search query"},
        "required": ["query"],
    },
    "zoom_images": {
        "inputs": {
            "frames": "list of base64 frames",
            "save_dir": "explicit output directory",
            "zoom_factor": "finite positive magnification factor",
        },
        "required": ["frames", "save_dir", "zoom_factor"],
    },
}


# Eq. (5) planning is deliberately narrower than the reflection registry.
# Values are the only pixel transforms a planning response may execute.
PLANNING_PREPROCESSING_ALIASES = {
    "image brightness enhancement": "enhance_brightness",
    "image denoising": "denoise_images",
    "image denoise": "denoise_images",
    "image deblurring": "deblur_images",
    "image deblur": "deblur_images",
    "image super-resolution": "super_resolve_images",
    "brightness enhancement": "enhance_brightness",
    "enhance_brightness": "enhance_brightness",
    "video denoising": "denoise_images",
    "denoise_images": "denoise_images",
    "deblurring": "deblur_images",
    "deblur_images": "deblur_images",
    "super-resolution": "super_resolve_images",
    "super_resolve_images": "super_resolve_images",
}


def _lazy_tool(name: str) -> Callable[..., dict]:
    def invoke(**kwargs: Any) -> dict:
        module = import_module("panda.tools.image_tools")
        return getattr(module, name)(**kwargs)

    invoke.__name__ = name
    return invoke


TOOL_REGISTRY = {name: _lazy_tool(name) for name in TOOL_KNOWLEDGE}


class ToolManager:
    """Describe and execute a closed set of tools with strict keyword schemas."""

    def __init__(
        self,
        registry: Mapping[str, Callable[..., dict]],
        knowledge: Mapping[str, str],
        parameter_hints: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.registry = dict(registry)
        self.knowledge = dict(knowledge)
        self.parameter_hints = {
            name: dict(hint) for name, hint in parameter_hints.items()
        }

    def get_tool_function(self, tool_name: str):
        return self.registry.get(tool_name)

    def get_tool_description(self, tool_name: str) -> str:
        return self.knowledge.get(tool_name, "No description available.")

    def get_tool_parameter_hint(self, tool_name: str) -> dict:
        return dict(self.parameter_hints.get(tool_name, {}))

    def list_all_tools(self) -> list[str]:
        return list(self.registry)

    def execute(self, tool_name: str, **kwargs: Any) -> dict:
        if tool_name not in self.registry:
            raise ValueError(f"Unknown tool: {tool_name}")
        hint = self.parameter_hints.get(tool_name, {})
        allowed = set(hint.get("inputs", {}))
        required = set(hint.get("required", allowed))
        missing = sorted(required - kwargs.keys())
        if missing:
            raise ValueError(
                f"Tool {tool_name} missing required parameter(s): {', '.join(missing)}"
            )
        unexpected = sorted(kwargs.keys() - allowed)
        if unexpected:
            raise ValueError(
                f"Tool {tool_name} received unexpected parameter(s): {', '.join(unexpected)}"
            )
        result = self.registry[tool_name](**kwargs)
        if not isinstance(result, dict):
            raise TypeError(f"Tool {tool_name} must return a dict")
        return result


TOOL_MANAGER = ToolManager(TOOL_REGISTRY, TOOL_KNOWLEDGE, TOOL_PARAMETER_HINTS)
