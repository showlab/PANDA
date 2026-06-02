"""Pure offline post-processing for complete frame-level score arrays."""

import math
from typing import Iterable, List


def _reflect_index(index: int, length: int) -> int:
    """SciPy ndimage ``mode='reflect'`` (half-sample symmetric) index map."""
    period = 2 * length
    reflected = index % period
    return reflected if reflected < length else period - reflected - 1


def _boundary_index(index: int, length: int, boundary: str) -> int:
    if boundary == "reflect":
        return _reflect_index(index, length)
    return min(max(index, 0), length - 1)


def apply_score_smoothing(
    scores: Iterable[float],
    method: str = "none",
    window: int = 5,
    boundary: str = "reflect",
) -> List[float]:
    """Smooth a complete score sequence with an explicit edge policy.

    ``mean`` uses the arithmetic mean. ``median`` uses the upper middle value
    for even windows, matching ``scipy.ndimage.median_filter``. The footprint
    offsets are ``[-window//2, ..., window-window//2-1]``, which also matches
    SciPy's default ``origin=0`` behavior. ``boundary`` defaults to ``reflect``;
    ``nearest`` repeats the endpoint values outside the sequence.
    """
    if method not in {"none", "mean", "median"}:
        raise ValueError("smoothing method must be one of: none, mean, median")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("smoothing window must be a positive integer")
    if boundary not in {"reflect", "nearest"}:
        raise ValueError("smoothing boundary must be one of: reflect, nearest")
    try:
        values = [float(value) for value in scores]
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be a one-dimensional numeric sequence") from exc
    if not values:
        raise ValueError("scores must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scores must contain only finite values")
    if method == "none":
        return list(values)

    start_offset = -(window // 2)
    result: List[float] = []
    for center in range(len(values)):
        footprint = [
            values[_boundary_index(center + start_offset + step, len(values), boundary)]
            for step in range(window)
        ]
        if method == "mean":
            result.append(sum(footprint) / window)
        else:
            result.append(sorted(footprint)[window // 2])
    return result
