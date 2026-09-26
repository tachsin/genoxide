"""Quality indicators of multi-objective fronts, computed in Rust.

A front is a 2-D array with a point per row and a column per objective, 2 to 6 of them, such as
``MultiResult.front_objectives``. ``objectives`` gives "minimize" or "maximize" per column, all
"minimize" by default::

    import numpy as np
    import genoxide as gx

    front = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    gx.indicators.hypervolume(front, [4.0, 4.0])  # 6.0: 3 + 2 + 1 unit squares
    gx.indicators.igd(front[:1], front)  # the mean distance of the points of front to front[0]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _genoxide

__all__ = ["hypervolume", "igd", "igd_plus", "gd", "spread"]


def _points(name: str, points: Any, columns: int | None) -> np.ndarray:
    """``points`` as a float64 array with a point per row; an empty front may be ``[]``."""
    array = np.ascontiguousarray(points, dtype=np.float64)
    if array.size == 0 and columns is not None:
        return array.reshape(0, columns)
    if array.ndim != 2:
        raise ValueError(
            f"{name} is a 2-D array, a point per row, not an array of shape {array.shape}"
        )
    return array


def _fronts(front: Any, reference_front: Any) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.ascontiguousarray(points, dtype=np.float64) for points in (front, reference_front)]
    # the number of objectives, from a 2-D array; an empty front or reference front may be []
    columns = next((array.shape[1] for array in arrays if array.ndim == 2), None)
    points = _points("front", front, columns)
    reference = _points("reference_front", reference_front, columns)
    if points.shape[1] != reference.shape[1]:
        raise ValueError(
            f"the front has {points.shape[1]} objectives and the reference front "
            f"{reference.shape[1]}"
        )
    return points, reference


def _objectives(objectives: Sequence[str] | None, count: int) -> list[str]:
    if objectives is None:
        return ["minimize"] * count
    objectives = list(objectives)
    if len(objectives) != count:
        raise ValueError(f"objectives has {len(objectives)} entries, for {count} objectives")
    return objectives


def hypervolume(
    front: Any, reference_point: Any, objectives: Sequence[str] | None = None
) -> float:
    """The hypervolume of ``front``: the volume of the region that its points dominate, bounded
    by ``reference_point``. Points that don't dominate the reference point add nothing. Larger is
    better, and a front that dominates another has a larger hypervolume.

    The computation is exact: fast for 2 to 4 objectives, slow for hundreds of points in 5 or
    more.

    Raises
    ------
    ValueError
        For arrays of the wrong shape, a number of objectives other than 2 to 6, or an objective
        other than "minimize" or "maximize".
    """
    reference = np.ascontiguousarray(reference_point, dtype=np.float64)
    if reference.ndim != 1:
        raise ValueError(f"reference_point is a 1-D array, not an array of shape {reference.shape}")
    points = _points("front", front, len(reference))
    if points.shape[1] != len(reference):
        raise ValueError(
            f"the front has {points.shape[1]} objectives and the reference point {len(reference)}"
        )
    return _genoxide.indicator(
        "hypervolume",
        points,
        reference.reshape(1, -1),
        _objectives(objectives, len(reference)),
    )


def igd(front: Any, reference_front: Any) -> float:
    """The inverted generational distance: the mean distance from each point of
    ``reference_front`` to the nearest point of ``front``. Smaller is better; 0 means that
    ``front`` covers every reference point. Not Pareto compliant: see :func:`igd_plus`.

    Infinite for an empty front, and NaN for an empty reference front.

    Raises
    ------
    ValueError
        For arrays of the wrong shape, or a number of objectives other than 2 to 6.
    """
    points, reference = _fronts(front, reference_front)
    return _genoxide.indicator("igd", points, reference, _objectives(None, points.shape[1]))


def gd(front: Any, reference_front: Any) -> float:
    """The generational distance: the mean distance from each point of ``front`` to the nearest
    point of ``reference_front``. Smaller is better; it measures convergence only.

    Infinite for an empty reference front, and NaN for an empty front.

    Raises
    ------
    ValueError
        For arrays of the wrong shape, or a number of objectives other than 2 to 6.
    """
    points, reference = _fronts(front, reference_front)
    return _genoxide.indicator("gd", points, reference, _objectives(None, points.shape[1]))


def igd_plus(
    front: Any, reference_front: Any, objectives: Sequence[str] | None = None
) -> float:
    """IGD+ (Ishibuchi et al., 2015): like :func:`igd`, but a point of ``front`` only counts as far
    from a reference point by how much it's worse in each objective. A front that dominates or
    equals every reference point scores 0, and a front that dominates another never scores worse.
    Smaller is better.

    Infinite for an empty front, and NaN for an empty reference front.

    Raises
    ------
    ValueError
        For arrays of the wrong shape, a number of objectives other than 2 to 6, or an objective
        other than "minimize" or "maximize".
    """
    points, reference = _fronts(front, reference_front)
    return _genoxide.indicator(
        "igd_plus", points, reference, _objectives(objectives, points.shape[1])
    )


def spread(
    front: Any, reference_front: Any, objectives: Sequence[str] | None = None
) -> float:
    """The generalized spread Δ (Zhou et al., 2006): how evenly ``front`` covers the reference
    front, from the distances of its points to their nearest neighbors and of the reference
    front's extremes to ``front``. 0 for evenly spaced points that reach the extremes; larger is
    worse. It says nothing about convergence.

    1 for fronts of fewer than 2 points, an empty reference front, or when every distance is 0.

    Raises
    ------
    ValueError
        For arrays of the wrong shape, a number of objectives other than 2 to 6, or an objective
        other than "minimize" or "maximize".
    """
    points, reference = _fronts(front, reference_front)
    return _genoxide.indicator(
        "spread", points, reference, _objectives(objectives, points.shape[1])
    )
