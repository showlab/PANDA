"""Video frame sampling and slicing utilities.

OpenCV/Pillow remain lazy imports. Sampling is driven by source/target timestamps,
not a rounded fixed-frame modulo interval.
"""

import math
from typing import Any, List, Optional, Sequence, Tuple


def _flatten_frames(frames: List) -> List[str]:
    """Flatten potentially nested frame lists into a flat list of strings."""
    flat: List[str] = []
    for frame in frames or []:
        if isinstance(frame, list):
            flat.extend(_flatten_frames(frame))
        else:
            flat.append(str(frame))
    return flat


def _positive_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def compute_sample_frame_indices(
    total_frames: int, video_fps: float, sampling_fps: float
) -> List[int]:
    """Return source-frame indices nearest at-or-before target timestamps.

    Target timestamps are ``n / sampling_fps`` and source frame ``i`` has
    timestamp ``i / video_fps``. Thus each index is
    ``floor(n * video_fps / sampling_fps)``. If the requested rate exceeds the
    source rate, duplicate timestamp mappings are removed, yielding every real
    source frame at most once.
    """
    if (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames <= 0
    ):
        raise ValueError("total_frames must be a positive integer")
    source_fps = _positive_finite_number(video_fps, "video_fps")
    target_fps = _positive_finite_number(sampling_fps, "sampling_fps")

    sample_count = int(math.ceil(total_frames * target_fps / source_fps))
    indices: List[int] = []
    previous = -1
    for sample_number in range(sample_count):
        index = int(math.floor(sample_number * source_fps / target_fps))
        if index >= total_frames:
            break
        if index > previous:
            indices.append(index)
            previous = index
    if not indices:
        raise ValueError("Sampling produced zero frame indices")
    return indices


def slice_frames_into_windows(frames: Sequence[Any], window: int) -> List[List[Any]]:
    """Split frames into non-overlapping windows, preserving a real short tail."""
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    return [
        list(frames[index : index + window]) for index in range(0, len(frames), window)
    ]


def select_perception_indices(
    sampled_indices: Sequence[int], max_frames: int, sampling: str
) -> List[int]:
    """Select at most ``max_frames`` sampled source indices for perception."""
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames <= 0
    ):
        raise ValueError("max_frames must be a positive integer frame count")
    if sampling not in {"legacy_initial", "paper_uniform"}:
        raise ValueError(
            "perception sampling must be 'legacy_initial' or 'paper_uniform'"
        )
    indices = list(sampled_indices)
    if not indices:
        raise ValueError("Cannot select perception frames from an empty sample")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("sampled_indices must contain integers")
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise ValueError("sampled_indices must be strictly increasing")

    count = min(max_frames, len(indices))
    if sampling == "legacy_initial" or count == len(indices):
        return indices[:count]
    if count == 1:
        return [indices[0]]

    # Integer half-up rounding over [0, len(indices)-1] is deterministic and,
    # because count <= len(indices), produces unique positions including both ends.
    span = len(indices) - 1
    denominator = count - 1
    positions = [
        (step * span * 2 + denominator) // (2 * denominator) for step in range(count)
    ]
    selected = [indices[position] for position in positions]
    if len(selected) != len(set(selected)):
        raise RuntimeError("Uniform perception selection produced duplicate indices")
    return selected


def _open_video_metadata(video_path: str):
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_raw = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(video_fps) or video_fps <= 0:
            raise ValueError(f"Video reports invalid FPS: {video_fps!r}")
        if not math.isfinite(total_raw) or total_raw <= 0:
            raise ValueError(f"Video reports zero/invalid frame count: {total_raw!r}")
        total_frames = int(total_raw)
        if total_frames <= 0:
            raise ValueError("Video reports zero frames")
        return cap, video_fps, total_frames
    except Exception:
        cap.release()
        raise


def _encode_selected_frames(
    cap: Any, selected_indices: Sequence[int], target_size: Tuple[int, int]
) -> List[str]:
    import base64
    from io import BytesIO

    import cv2
    from PIL import Image

    target_height, target_width = target_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_size dimensions must be positive")

    wanted_position = 0
    encoded: List[str] = []
    final_index = selected_indices[-1]
    frame_index = 0
    while frame_index <= final_index:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index == selected_indices[wanted_position]:
            height, width = frame.shape[:2]
            if (height, width) != (target_height, target_width):
                frame = cv2.resize(
                    frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
            wanted_position += 1
            if wanted_position == len(selected_indices):
                break
        frame_index += 1

    if len(encoded) != len(selected_indices):
        raise ValueError(
            f"Video decode ended early: requested {len(selected_indices)} frames, decoded {len(encoded)}"
        )
    return encoded


def extract_keyframes_base64(
    video_path: str,
    fps: float = 1,
    duration: float = 99999999,
    target_size: Tuple[int, int] = (480, 640),
) -> Tuple[List[str], float]:
    """Extract timestamp-sampled frames as base64 PNGs."""
    duration_value = _positive_finite_number(duration, "duration")
    cap, video_fps, total_frames = _open_video_metadata(video_path)
    try:
        duration_frames = min(total_frames, int(math.ceil(duration_value * video_fps)))
        if duration_frames <= 0:
            raise ValueError("duration produced zero source frames")
        indices = compute_sample_frame_indices(duration_frames, video_fps, fps)
        return _encode_selected_frames(cap, indices, target_size), video_fps
    finally:
        cap.release()


def extract_initial_keyframes_base64(
    video_path: str,
    fps: float = 1,
    duration: float = 10,
    target_size: Tuple[int, int] = (240, 320),
) -> List[str]:
    """Compatibility helper: sample only the requested initial duration."""
    frames, _ = extract_keyframes_base64(
        video_path, fps=fps, duration=duration, target_size=target_size
    )
    return frames


def extract_perception_keyframes_base64(
    video_path: str,
    *,
    fps: float = 1.0,
    max_frames: int,
    sampling: str,
    target_size: Tuple[int, int] = (240, 320),
    return_metadata: bool = False,
) -> Any:
    """Extract a perception sample, optionally returning its actual time mapping.

    Native video accepts a constant FPS. Uniform capped selection can have
    unequal source intervals; its endpoint-average FPS is explicitly marked as
    an approximation, not as exact per-frame timestamps.
    """
    cap, video_fps, total_frames = _open_video_metadata(video_path)
    try:
        sampled = compute_sample_frame_indices(total_frames, video_fps, fps)
        selected = select_perception_indices(sampled, max_frames, sampling)
        frames = _encode_selected_frames(cap, selected, target_size)
        if not return_metadata:
            return frames
        intervals = [right - left for left, right in zip(selected, selected[1:])]
        if intervals:
            effective_fps = (
                (len(selected) - 1) * video_fps / (selected[-1] - selected[0])
            )
            policy = (
                "constant_source_spacing"
                if len(set(intervals)) == 1
                else "mean_interval_approximation"
            )
        else:
            effective_fps = min(float(fps), video_fps)
            policy = "single_frame_no_motion"
        return frames, {
            "source_frame_indices": selected,
            "source_fps": video_fps,
            "source_timestamps_seconds": [index / video_fps for index in selected],
            "sampling_fps": float(fps),
            "video_fps": effective_fps,
            "time_policy": policy,
        }
    finally:
        cap.release()


def load_video_and_slice(
    video_path: str,
    save_dir: Optional[str] = None,
    fps: float = 1,
    window: int = 5,
) -> Tuple[List[List[str]], float]:
    """Extract frames and split them with the shared exact-tail window logic."""
    del save_dir
    frames, video_fps = extract_keyframes_base64(video_path, fps=fps, duration=99999999)
    return slice_frames_into_windows(frames, window), video_fps


# Re-export message builders for compatibility with the original public module.
from panda.tools.vlm_client import (  # noqa: E402, F401
    _build_messages,
    _make_video_content,
)
