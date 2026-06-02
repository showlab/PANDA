"""Lazy, fail-closed image and context tools used by reflection.

The module itself imports only Python's standard library. Optional packages and
models are loaded on the first call that actually needs them.
"""

from __future__ import annotations

import base64
import json
import math
import os
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from panda.tool_assets import loaded_tool_identity

_YOLO_MODEL = None
_CLIP_COMPONENTS = None
_SR_MODEL = None
_MODEL_LOCK = threading.Lock()
_TOOL_MODEL_IDENTITIES: dict[str, dict] = {}

YOLO_CLASSES = [
    "a person",
    "a group of people",
    "a person running",
    "two people physically fighting",
    "a person pointing a gun",
    "a person holding a weapon",
    "a person lying on the ground",
    "a crashed vehicle",
    "fire",
    "smoke",
]


def _required_model_path(variable: str) -> str:
    value = os.getenv(variable, "").strip()
    if not value:
        raise ValueError(f"{variable} must be set before using this tool")
    if not Path(value).exists():
        raise ValueError(f"{variable} path does not exist")
    return value


def _decode_image(encoded: str):
    from PIL import Image

    raw = base64.b64decode(encoded, validate=True)
    image = Image.open(BytesIO(raw))
    image.load()
    return image.convert("RGB")


def _encode_image(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _output_directory(save_dir: str | os.PathLike[str]) -> Path:
    if not str(save_dir).strip():
        raise ValueError("save_dir must be an explicit non-empty path")
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _process_pillow_frames(
    *,
    frames: list[str],
    save_dir: str | os.PathLike[str],
    operation: Callable[[Any], Any],
    label: str,
) -> dict:
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty list of base64 images")
    directory = _output_directory(save_dir)
    enhanced: list[str] = []
    failures: list[dict] = []
    for index, encoded in enumerate(frames):
        try:
            output = operation(_decode_image(encoded))
            output.save(directory / f"frame_{index}.png", format="PNG")
            enhanced.append(_encode_image(output))
        except (
            Exception
        ) as exc:  # retain partial success but never report all-failed as success
            failures.append({"frame_index": index, "error_type": type(exc).__name__})
    if not enhanced:
        raise RuntimeError(f"{label} processed 0 of {len(frames)} frames")
    return {
        "summary": f"{label} processed {len(enhanced)} of {len(frames)} frames.",
        "enhanced_frames": enhanced,
        "frame_failures": failures,
    }


def _process_opencv_frames(
    *,
    frames: list[str],
    save_dir: str | os.PathLike[str],
    operation: Callable[[Any], Any],
    label: str,
) -> dict:
    """Process an ordered base64-PNG frame list entirely in memory."""
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty list of base64 images")
    if not str(save_dir).strip():
        raise ValueError("save_dir must be an explicit non-empty path")

    import cv2
    import numpy as np

    enhanced: list[str] = []
    for index, encoded in enumerate(frames):
        try:
            raw = base64.b64decode(encoded, validate=True)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("decoded data is not an image")
            output = operation(image)
            success, png = cv2.imencode(".png", output)
            if not success:
                raise ValueError("OpenCV could not encode output as PNG")
            enhanced.append(base64.b64encode(png.tobytes()).decode("ascii"))
        except Exception as exc:
            raise RuntimeError(
                f"{label} rejected invalid frame {index} ({type(exc).__name__})"
            ) from None
    return {
        "summary": f"{label} processed {len(enhanced)} of {len(frames)} frames.",
        "enhanced_frames": enhanced,
        "frame_failures": [],
    }


def denoise_images(*, frames: list[str], save_dir: str) -> dict:
    """Apply OpenCV colored non-local-means denoising."""
    import cv2

    return _process_opencv_frames(
        frames=frames,
        save_dir=save_dir,
        operation=lambda image: cv2.fastNlMeansDenoisingColored(
            image, None, 10, 10, 7, 21
        ),
        label="Denoising",
    )


def deblur_images(*, frames: list[str], save_dir: str) -> dict:
    """Apply OpenCV Gaussian unsharp masking."""
    import cv2

    def deblur(image):
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    return _process_opencv_frames(
        frames=frames,
        save_dir=save_dir,
        operation=deblur,
        label="Deblurring",
    )


def enhance_brightness(*, frames: list[str], save_dir: str) -> dict:
    """Apply CLAHE to the OpenCV LAB luminance channel."""
    import cv2

    def enhance(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_lab = cv2.merge((clahe.apply(luminance), channel_a, channel_b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return _process_opencv_frames(
        frames=frames,
        save_dir=save_dir,
        operation=enhance,
        label="Brightness enhancement",
    )


def zoom_images(*, frames: list[str], save_dir: str, zoom_factor: float) -> dict:
    """Magnify frames by an explicit factor using bicubic interpolation (paper D)."""
    if (
        isinstance(zoom_factor, bool)
        or not isinstance(zoom_factor, (int, float))
        or not math.isfinite(zoom_factor)
        or zoom_factor <= 0
    ):
        raise ValueError("zoom_factor must be a finite positive number")
    import cv2

    def zoom(image):
        height, width = image.shape[:2]
        size = (max(1, round(width * zoom_factor)), max(1, round(height * zoom_factor)))
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    return _process_opencv_frames(
        frames=frames, save_dir=save_dir, operation=zoom, label="Image zooming"
    )


def _get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    model_path = _required_model_path("PANDA_YOLO_MODEL_PATH")
    with _MODEL_LOCK:
        if _YOLO_MODEL is None:
            from ultralytics import YOLOWorld

            identity = loaded_tool_identity("object_detection")
            model = YOLOWorld(model_path)
            model.set_classes(YOLO_CLASSES)
            _TOOL_MODEL_IDENTITIES["object_detection"] = identity
            _YOLO_MODEL = model
    return _YOLO_MODEL


def _annotated_image(result):
    from PIL import Image

    plotted = result.plot()
    if isinstance(plotted, Image.Image):
        return plotted.convert("RGB")
    # Ultralytics plot() returns a BGR ndarray.
    return Image.fromarray(plotted[..., ::-1]).convert("RGB")


def _detection_counts(result) -> dict[str, int]:
    classes = getattr(getattr(result, "boxes", None), "cls", [])
    if hasattr(classes, "detach"):
        classes = classes.detach().cpu().tolist()
    elif hasattr(classes, "tolist"):
        classes = classes.tolist()
    names = getattr(result, "names", {})
    counts: dict[str, int] = {}
    for class_id in classes:
        key = int(class_id)
        label = names.get(key, str(key)) if hasattr(names, "get") else names[key]
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def object_detection(*, frames: list[str], save_dir: str) -> dict:
    """Run YOLOWorld and return the model's annotated frames, never originals."""
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty list of base64 images")
    model = _get_yolo_model()
    directory = _output_directory(save_dir)
    annotated_frames: list[str] = []
    summaries: list[str] = []
    failures: list[dict] = []
    for index, encoded in enumerate(frames):
        try:
            image = _decode_image(encoded)
            result = model(image)[0]
            annotated = _annotated_image(result)
            annotated.save(directory / f"frame_{index}.png", format="PNG")
            annotated_frames.append(_encode_image(annotated))
            counts = _detection_counts(result)
            details = ", ".join(
                f"{name}: {count}" for name, count in sorted(counts.items())
            )
            summaries.append(
                f"Frame {index}: {details or 'no configured objects detected'}"
            )
        except Exception as exc:
            failures.append({"frame_index": index, "error_type": type(exc).__name__})
    if not annotated_frames:
        raise RuntimeError(f"Object detection processed 0 of {len(frames)} frames")
    return {
        "summary": "Object detection results:\n" + "\n".join(summaries),
        "asset_identity": dict(_TOOL_MODEL_IDENTITIES.get("object_detection", {})),
        "enhanced_frames": annotated_frames,
        "frame_failures": failures,
    }


def _ensure_basicsr_torchvision_compat():
    """Bridge BasicSR's removed import to torchvision's public grayscale API."""
    import importlib
    import sys
    from types import ModuleType

    module_name = "torchvision.transforms.functional_tensor"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        from torchvision.transforms.functional import rgb_to_grayscale

        compatibility = ModuleType(module_name)
        setattr(compatibility, "rgb_to_grayscale", rgb_to_grayscale)
        sys.modules.setdefault(module_name, compatibility)


def _get_super_resolution_model():
    global _SR_MODEL
    if _SR_MODEL is not None:
        return _SR_MODEL
    model_path = _required_model_path("PANDA_SR_MODEL_PATH")
    with _MODEL_LOCK:
        if _SR_MODEL is None:
            import torch

            _ensure_basicsr_torchvision_compat()
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            network = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            identity = loaded_tool_identity("super_resolve_images")
            _SR_MODEL = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=network,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            _TOOL_MODEL_IDENTITIES["super_resolve_images"] = identity
    return _SR_MODEL


def super_resolve_images(*, frames: list[str], save_dir: str) -> dict:
    """Run Real-ESRGAN lazily and return actual super-resolved frames."""
    model = _get_super_resolution_model()
    import numpy as np
    from PIL import Image

    def enhance(image):
        rgb = np.asarray(image)
        bgr = rgb[..., ::-1]
        output_bgr, _ = model.enhance(bgr)
        return Image.fromarray(output_bgr[..., ::-1]).convert("RGB")

    result = _process_pillow_frames(
        frames=frames,
        save_dir=save_dir,
        operation=enhance,
        label="Super-resolution",
    )
    result["asset_identity"] = dict(
        _TOOL_MODEL_IDENTITIES.get("super_resolve_images", {})
    )
    return result


def _get_clip_components():
    global _CLIP_COMPONENTS
    if _CLIP_COMPONENTS is not None:
        return _CLIP_COMPONENTS
    model_path = _required_model_path("PANDA_CLIP_MODEL_PATH")
    with _MODEL_LOCK:
        if _CLIP_COMPONENTS is None:
            from transformers import CLIPModel, CLIPProcessor

            identity = loaded_tool_identity("retrieve_past_keyframes")
            model = CLIPModel.from_pretrained(model_path).eval()
            processor = CLIPProcessor.from_pretrained(model_path)
            _TOOL_MODEL_IDENTITIES["retrieve_past_keyframes"] = identity
            _CLIP_COMPONENTS = (model, processor)
    return _CLIP_COMPONENTS


def _score_frames_with_clip(query: str, frames: list[str]) -> list[float]:
    model, processor = _get_clip_components()
    import torch

    images = [_decode_image(frame) for frame in frames]
    inputs = processor(text=[query], images=images, return_tensors="pt", padding=True)
    device = getattr(model, "device", None)
    if device is not None and hasattr(inputs, "to"):
        inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits_per_text[0]
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu()
    return [float(value) for value in scores.tolist()]


def retrieve_past_keyframes(
    *,
    frames: list[str],
    query: str,
    save_dir: str,
    current_window_index: int,
    window_size: int,
    lookback: int = 3,
    top_k: int = 3,
) -> dict:
    """Retrieve top-k frames from only the requested earlier-window interval."""
    if not isinstance(frames, list):
        raise ValueError("frames must be a list")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if current_window_index < 1:
        raise ValueError("current_window_index must have at least one previous window")
    if window_size < 1 or lookback < 1 or top_k < 1:
        raise ValueError("window_size, lookback, and top_k must be positive")

    start_window = max(0, current_window_index - lookback)
    start = start_window * window_size
    stop = min(current_window_index * window_size, len(frames))
    candidates = list(frames[start:stop])
    if not candidates:
        raise ValueError(
            "No historical frames are available in the requested lookback window"
        )

    scores = _score_frames_with_clip(query.strip(), candidates)
    if len(scores) != len(candidates):
        raise RuntimeError(
            "CLIP scorer returned a score count inconsistent with candidate frames"
        )
    ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[:top_k]
    selected = [candidates[index] for index, _ in ranked]

    directory = _output_directory(save_dir)
    saved = 0
    for rank, encoded in enumerate(selected):
        try:
            _decode_image(encoded).save(
                directory / f"retrieved_frame_{rank}.png", "PNG"
            )
            saved += 1
        except Exception:
            continue
    if saved == 0:
        raise RuntimeError(
            f"Historical retrieval saved 0 of {len(selected)} selected frames"
        )
    return {
        "summary": (
            f"Retrieved {len(selected)} frame(s) from prior windows "
            f"[{start_window}, {current_window_index}) with top_k={top_k}."
        ),
        "past_keyframes": selected,
        "asset_identity": dict(
            _TOOL_MODEL_IDENTITIES.get("retrieve_past_keyframes", {})
        ),
    }


def web_search(query: str) -> dict:
    """Call Tavily without a global client and return sanitized structured results."""
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        raise ValueError("TAVILY_API_KEY must be set before using web_search")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    from urllib import error, request

    payload = json.dumps(
        {
            "api_key": key,
            "query": query.strip(),
            "max_results": 3,
            "search_depth": "basic",
        }
    ).encode("utf-8")
    req = request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(
            f"Tavily web search failed with HTTP status {exc.code}"
        ) from None
    except Exception as exc:
        raise RuntimeError(f"Tavily web search failed ({type(exc).__name__})") from None

    raw_results = parsed.get("results", []) if isinstance(parsed, dict) else []
    if not isinstance(raw_results, list):
        raise RuntimeError("Tavily web search returned an invalid result structure")
    results = []
    for item in raw_results[:3]:
        if isinstance(item, dict):
            results.append(
                {
                    "title": str(item.get("title", "")),
                    "content": str(item.get("content", "")),
                    "url": str(item.get("url", "")),
                }
            )
    summary = "\n\n".join(
        f"{index}. {item['title']}\n{item['content']}\n{item['url']}"
        for index, item in enumerate(results, 1)
    )
    return {"summary": summary or "No web results found.", "results": results}
