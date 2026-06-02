"""Canonical annotation parsing and frame-label mapping for PANDA evaluation."""

from typing import Any, Dict, List, Sequence

import numpy as np

_DATASETS = {"ucf", "xd", "ub", "csad"}


def parse_annotation_line(line: str, dataset: str = "ucf") -> Dict[str, Any]:
    """Parse one non-empty annotation row.

    UCF rows are ``filename class start end ...``. XD/UB/CSAD rows are
    ``filename start end ...``. Segment endpoints are inclusive original-frame
    indices. ``-1 -1`` is a sentinel pair and is not an anomaly segment.
    """
    dataset = dataset.lower()
    if dataset not in _DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Valid options: {sorted(_DATASETS)}"
        )

    parts = line.strip().split()
    if not parts:
        raise ValueError("Annotation line must not be empty")

    if dataset == "ucf":
        if len(parts) < 2:
            raise ValueError("UCF annotation requires filename and class")
        video_name = parts[0].lstrip("@")
        label = parts[1]
        segments = _parse_segments(parts[2:])
        is_normal = label.lower() == "normal"
        if is_normal != (not segments):
            raise ValueError(f"UCF label/segments inconsistent for '{video_name}'")
        return {"video_name": video_name, "label": label, "segments": segments}

    video_name = parts[0]
    segments = _parse_segments(parts[1:])
    label = _infer_binary_label(video_name, segments, dataset)
    return {"video_name": video_name, "label": label, "segments": segments}


def _parse_segments(tokens: Sequence[str]) -> List[List[int]]:
    if len(tokens) % 2:
        raise ValueError(
            f"Segment endpoints must occur in start/end pairs; got {len(tokens)} integers"
        )
    try:
        values = [int(token) for token in tokens]
    except (TypeError, ValueError) as exc:
        raise ValueError("Segment endpoints must be integers") from exc

    segments: List[List[int]] = []
    for start, end in zip(values[::2], values[1::2]):
        if start == -1 and end == -1:
            continue
        if start < 0 or end < 0:
            raise ValueError(
                f"Invalid negative segment [{start}, {end}]; only -1 -1 is a sentinel"
            )
        if end < start:
            raise ValueError(f"Segment end {end} precedes start {start}")
        segments.append([start, end])
    return segments


def _infer_binary_label(
    video_name: str, segments: Sequence[Sequence[int]], dataset: str
) -> str:
    lower_name = video_name.lower()
    has_segments = bool(segments)

    if dataset == "xd":
        filename_label = "normal" if "_label_a" in lower_name else "abnormal"
    elif dataset == "ub":
        if lower_name.startswith("normal_"):
            filename_label = "normal"
        elif lower_name.startswith("abnormal_"):
            filename_label = "abnormal"
        else:
            raise ValueError(f"Cannot infer UB label from filename '{video_name}'")
    else:  # CSAD is a merged cross-dataset annotation file.
        if (
            lower_name.startswith("normal_")
            or lower_name.startswith("normal_videos_")
            or "_label_a" in lower_name
        ):
            filename_label = "normal"
        else:
            filename_label = "abnormal"

    segment_label = "abnormal" if has_segments else "normal"
    if filename_label != segment_label:
        raise ValueError(
            f"Filename and segments are inconsistent for '{video_name}': "
            f"filename={filename_label}, segments={segment_label}"
        )
    return filename_label


def build_frame_label(
    total_frames_sampled: int,
    segments: List[List[int]],
    fps_original: float = 30,
    fps_sampled: float = 1,
) -> np.ndarray:
    """Map inclusive original-frame segments to sampled-frame labels.

    Both annotation endpoints are included in the anomalous interval.
    Mapping uses the exact FPS ratio: sampled index ``floor(frame *
    fps_sampled / fps_original)``. The returned array is always exactly
    ``total_frames_sampled`` long, so a final short window is not padded.
    """
    if not isinstance(total_frames_sampled, int) or total_frames_sampled < 0:
        raise ValueError("total_frames_sampled must be a non-negative integer")
    if not np.isfinite(fps_original) or fps_original <= 0:
        raise ValueError("fps_original must be finite and positive")
    if not np.isfinite(fps_sampled) or fps_sampled <= 0:
        raise ValueError("fps_sampled must be finite and positive")

    label = np.zeros(total_frames_sampled, dtype=np.float64)
    scale = float(fps_sampled) / float(fps_original)
    for segment in segments:
        if len(segment) != 2:
            raise ValueError(f"Segment must contain two endpoints: {segment!r}")
        start, end = segment
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError(f"Segment endpoints must be finite: {segment!r}")
        if start < 0 or end < start:
            raise ValueError(f"Invalid inclusive segment: {segment!r}")
        start_idx = int(np.floor(start * scale))
        end_idx = int(np.floor(end * scale))
        if start_idx < total_frames_sampled:
            label[start_idx : min(end_idx + 1, total_frames_sampled)] = 1.0
    return label
