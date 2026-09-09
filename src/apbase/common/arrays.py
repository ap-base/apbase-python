from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import ArrayLike

from .exceptions import ValidationError


@dataclass(frozen=True)
class MaskedRegularGridSpec:
    min_x: float
    min_y: float
    dx: float
    dy: float
    full_nx: int
    full_ny: int
    row_indices: np.ndarray
    x_start_indices: np.ndarray
    counts: np.ndarray
    point_count: int


def as_float64_1d(values: ArrayLike, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must contain numeric values") from None

    if array.ndim != 1:
        raise ValidationError(f"{name} must be a 1D array")
    return np.ascontiguousarray(array)


def finite_mask_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of positions where x and y are both finite."""
    finite_mask = np.isfinite(x)
    finite_mask &= np.isfinite(y)
    return finite_mask


def filter_finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x, y filtered to rows where both are finite."""
    finite_mask = finite_mask_xy(x, y)
    if bool(np.all(finite_mask)):
        return x, y
    return (
        np.ascontiguousarray(x[finite_mask]),
        np.ascontiguousarray(y[finite_mask]),
    )


def validate_same_size_xy(x: np.ndarray, y: np.ndarray) -> None:
    """Raise ValidationError unless x, y are non-empty and equal length."""
    if x.size == 0:
        raise ValidationError("x and y must contain at least one point")
    if x.size != y.size:
        raise ValidationError("x and y must have the same size")


def finite_mask_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Boolean mask of positions where x, y, and z are all finite."""
    finite_mask = np.isfinite(x)
    finite_mask &= np.isfinite(y)
    finite_mask &= np.isfinite(z)
    return finite_mask


def filter_finite_xyz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, z filtered to rows where all three are finite."""
    finite_mask = finite_mask_xyz(x, y, z)
    if bool(np.all(finite_mask)):
        return x, y, z
    return (
        np.ascontiguousarray(x[finite_mask]),
        np.ascontiguousarray(y[finite_mask]),
        np.ascontiguousarray(z[finite_mask]),
    )


def validate_same_size_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Raise ValidationError unless x, y, z are non-empty and equal length."""
    if x.size == 0:
        raise ValidationError("x, y, and z must contain at least one point")
    if x.size != y.size or x.size != z.size:
        raise ValidationError("x, y, and z must have the same size")


def extract_target_xy(targets: object) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(targets, "geometry"):
        geometry = targets.geometry
        return (
            np.ascontiguousarray(geometry.x.to_numpy(dtype=np.float64)),
            np.ascontiguousarray(geometry.y.to_numpy(dtype=np.float64)),
        )

    if hasattr(targets, "points"):
        return extract_target_xy(targets.points)

    array = np.asarray(targets)
    if array.ndim == 2 and array.shape[1] == 2:
        return (
            np.ascontiguousarray(array[:, 0], dtype=np.float64),
            np.ascontiguousarray(array[:, 1], dtype=np.float64),
        )

    # GeoSeries or plain list/tuple of shapely Points: shapely.get_x/get_y are
    # vectorized (GEOS) ufuncs, avoiding a Python-level per-geometry .x/.y loop.
    # Unlike a bare get_x/get_y call, get_type_id is checked explicitly first --
    # get_x/get_y silently return NaN for non-Point geometries rather than
    # raising, which would otherwise turn a real input error into a silent NaN
    # target instead of the ValidationError callers expect.
    invalid_input_error = ValidationError(
        "targets must be a GeoDataFrame, GeoSeries/list of Points, or an (n, 2) array"
    )
    try:
        targets_array = np.asarray(targets, dtype=object)
    except TypeError:
        raise invalid_input_error from None

    if targets_array.ndim != 1 or targets_array.size == 0:
        raise invalid_input_error
    try:
        is_point = shapely.get_type_id(targets_array) == shapely.GeometryType.POINT
    except TypeError:
        raise invalid_input_error from None
    if not np.all(is_point):
        raise invalid_input_error

    target_x = shapely.get_x(targets_array)
    target_y = shapely.get_y(targets_array)
    if target_x.size != target_y.size:
        raise ValidationError("targets has inconsistent coordinate lengths")
    return (
        np.ascontiguousarray(target_x, dtype=np.float64),
        np.ascontiguousarray(target_y, dtype=np.float64),
    )


def normalize_masked_regular_grid_spec(grid: object) -> MaskedRegularGridSpec | None:
    array = np.asarray(grid)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        return None

    x = np.ascontiguousarray(array[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(array[:, 1], dtype=np.float64)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    if unique_x.size == 0 or unique_y.size == 0:
        return None

    dx = _axis_step(unique_x)
    dy = _axis_step(unique_y)
    if dx is None or dy is None:
        return None

    min_x = float(unique_x[0])
    min_y = float(unique_y[0])
    full_nx = int(round((float(unique_x[-1]) - min_x) / dx)) + 1
    full_ny = int(round((float(unique_y[-1]) - min_y) / dy)) + 1
    if full_nx <= 0 or full_ny <= 0:
        return None

    x_index = np.rint((x - min_x) / dx).astype(np.int64)
    y_index = np.rint((y - min_y) / dy).astype(np.int64)
    if np.any(x_index < 0) or np.any(x_index >= full_nx) or np.any(y_index < 0) or np.any(y_index >= full_ny):
        return None

    atol = max(abs(dx), abs(dy), 1.0) * 1.0e-8
    if not np.allclose(min_x + x_index.astype(np.float64) * dx, x, rtol=1.0e-10, atol=atol):
        return None
    if not np.allclose(min_y + y_index.astype(np.float64) * dy, y, rtol=1.0e-10, atol=atol):
        return None

    # Run-length-encode consecutive points (in input order) that share a row
    # and have strictly incrementing column indices, vectorized instead of a
    # point-by-point Python while loop: a new run starts at position 0, and
    # at any later position whose row differs from the previous point's row
    # or whose column is not exactly one more than the previous point's.
    point_count = int(array.shape[0])
    is_run_start = np.empty(point_count, dtype=bool)
    is_run_start[0] = True
    is_run_start[1:] = (y_index[1:] != y_index[:-1]) | (x_index[1:] != x_index[:-1] + 1)

    run_start_positions = np.flatnonzero(is_run_start)
    run_end_positions = np.empty(run_start_positions.size, dtype=np.int64)
    run_end_positions[:-1] = run_start_positions[1:]
    run_end_positions[-1] = point_count

    return MaskedRegularGridSpec(
        min_x=min_x,
        min_y=min_y,
        dx=float(dx),
        dy=float(dy),
        full_nx=full_nx,
        full_ny=full_ny,
        row_indices=np.ascontiguousarray(y_index[run_start_positions], dtype=np.int32),
        x_start_indices=np.ascontiguousarray(x_index[run_start_positions], dtype=np.int32),
        counts=np.ascontiguousarray(run_end_positions - run_start_positions, dtype=np.int32),
        point_count=point_count,
    )


def _axis_step(axis: np.ndarray) -> float | None:
    if axis.size == 1:
        return 1.0

    diffs = np.diff(axis)
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        return None

    step = float(np.min(positive))
    if not np.isfinite(step) or step <= 0.0:
        return None

    indices = np.rint((axis - float(axis[0])) / step)
    expected = float(axis[0]) + indices * step
    if not np.allclose(expected, axis, rtol=1.0e-10, atol=max(abs(step), 1.0) * 1.0e-8):
        return None
    return step


__all__ = [
    "MaskedRegularGridSpec",
    "as_float64_1d",
    "extract_target_xy",
    "filter_finite_xy",
    "filter_finite_xyz",
    "finite_mask_xy",
    "finite_mask_xyz",
    "normalize_masked_regular_grid_spec",
    "validate_same_size_xy",
    "validate_same_size_xyz",
]
