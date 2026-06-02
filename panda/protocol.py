"""Named paper profiles and configurable engineering-mode behavior."""

from __future__ import annotations

import os
from pathlib import Path

PAPER_MODEL = "gemini-2.0-flash"
PAPER_VLM_MODEL = "Qwen2.5-VL-7B-Instruct"
PROFILES = ("engineering", "paper-offline", "paper-online")


def configure_profile(args):
    """Resolve a profile before decoding video or constructing model clients.

    Paper profiles specify sampling, retrieval, and model settings.
    ``last_score`` retains the final VLM score when reflection retries are exhausted.
    """
    profile = getattr(args, "profile", "engineering")
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    args.profile = profile
    args.model_substitution = False
    if profile == "engineering":
        args.memory_backend = getattr(args, "memory_backend", "none")
        args.exhausted_score_policy = "reject"
        return args
    for field, expected in (
        ("fps", 1.0),
        ("window", 5),
        ("num_rules", 5),
        ("vlm_temperature", 0.2),
        ("vlm_top_p", 0.5),
    ):
        if getattr(args, field, expected) != expected:
            raise ValueError(f"{profile} requires {field}={expected}")
        setattr(args, field, expected)
    if getattr(args, "allow_partial", False) or getattr(
        args, "allow_perception_fallback", False
    ):
        raise ValueError(
            "Paper profiles forbid partial coverage or perception fallback"
        )
    if getattr(args, "reflection_memory_path", ""):
        raise ValueError(
            "Paper memory is per-video; a shared JSONL memory path is forbidden"
        )
    for role in ("planning", "reflection"):
        provider = getattr(args, f"{role}_provider", "gemini")
        if provider not in ("gemini", "openai"):
            raise ValueError(f"Unsupported {role} provider: {provider}")
        setattr(args, f"{role}_provider", provider)
        if provider == "gemini":
            model = os.getenv("GEMINI_MODEL", PAPER_MODEL).strip()
            if not model:
                raise ValueError("GEMINI_MODEL must name an explicit model")
            args.model_substitution |= model != PAPER_MODEL
        else:
            # A different provider requires explicit substitution permission,
            # even when it advertises the same model name.
            args.model_substitution = True
    from panda.tools.vlm_client import resolve_vlm_model

    vlm_api = getattr(args, "vlm_api", "qwen2_5_7b_v0")
    args.vlm_models = {
        "perception": resolve_vlm_model(f"per-{vlm_api}"),
        "analysis": resolve_vlm_model(vlm_api),
    }
    # This is configured identity, not attestation of the weights served remotely.
    args.model_substitution |= any(
        not (
            model in (PAPER_VLM_MODEL, f"Qwen/{PAPER_VLM_MODEL}")
            or (Path(model).is_absolute() and Path(model).name == PAPER_VLM_MODEL)
        )
        for model in args.vlm_models.values()
    )
    if args.model_substitution and not getattr(args, "allow_model_substitution", False):
        raise ValueError(
            "The selected model/provider combination requires --allow-model-substitution"
        )
    args.memory_backend = "paper"
    args.exhausted_score_policy = "last_score"
    args.rag_backend = "faiss"
    args.perception_sampling = (
        "paper_uniform" if profile == "paper-offline" else "legacy_initial"
    )
    args.perception_frames_num = 300 if profile == "paper-offline" else 10
    return args


def validate_smoothing(profile: str, method: str) -> None:
    if profile == "paper-online" and method != "none":
        raise ValueError("paper-online forbids future-dependent offline smoothing")
