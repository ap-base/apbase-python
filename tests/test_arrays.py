from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from apbase.common.arrays import extract_target_xy, normalize_masked_regular_grid_spec
from apbase.common.exceptions import ValidationError


def test_extract_target_xy_from_ndarray() -> None:
    targets = np.array([[1.0, 2.0], [3.0, 4.0]])
    x, y = extract_target_xy(targets)
    np.testing.assert_array_equal(x, [1.0, 3.0])
    np.testing.assert_array_equal(y, [2.0, 4.0])


def test_extract_target_xy_from_list_of_points() -> None:
    targets = [Point(1.0, 2.0), Point(3.0, 4.0), Point(5.0, 6.0)]
    x, y = extract_target_xy(targets)
    np.testing.assert_array_equal(x, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(y, [2.0, 4.0, 6.0])
    assert x.dtype == np.float64
    assert y.dtype == np.float64


def test_extract_target_xy_from_geoseries() -> None:
    geopandas = pytest.importorskip("geopandas")
    targets = geopandas.GeoSeries([Point(1.0, 2.0), Point(3.0, 4.0)])
    x, y = extract_target_xy(targets)
    np.testing.assert_array_equal(x, [1.0, 3.0])
    np.testing.assert_array_equal(y, [2.0, 4.0])


def test_extract_target_xy_rejects_non_point_geometry() -> None:
    targets = [Point(1.0, 2.0), LineString([(0.0, 0.0), (1.0, 1.0)])]
    with pytest.raises(ValidationError):
        extract_target_xy(targets)


def test_extract_target_xy_rejects_non_geometry_objects() -> None:
    with pytest.raises(ValidationError):
        extract_target_xy(["not", "a", "geometry"])


def _reference_normalize_masked_regular_grid_spec(
    x_index: np.ndarray, y_index: np.ndarray
) -> tuple[list[int], list[int], list[int]]:
    # The point-by-point Python loop normalize_masked_regular_grid_spec used
    # before its vectorized np.flatnonzero rewrite, kept here only to fuzz
    # the rewrite against for exact equivalence.
    row_indices: list[int] = []
    x_start_indices: list[int] = []
    counts: list[int] = []
    pos = 0
    point_count = x_index.size
    while pos < point_count:
        row = int(y_index[pos])
        start_x = int(x_index[pos])
        count = 1
        pos += 1
        while pos < point_count and int(y_index[pos]) == row and int(x_index[pos]) == start_x + count:
            count += 1
            pos += 1
        row_indices.append(row)
        x_start_indices.append(start_x)
        counts.append(count)
    return row_indices, x_start_indices, counts


def test_normalize_masked_regular_grid_spec_matches_reference_rle_fuzz() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        ny = int(rng.integers(1, 6))
        nx = int(rng.integers(1, 10))
        dx = float(rng.uniform(0.5, 5.0))
        dy = float(rng.uniform(0.5, 5.0))
        min_x, min_y = 10.0, -20.0

        # Randomly mask out grid cells (simulating boundary exclusion), but
        # every row and every column must keep at least one point: the
        # function derives its own row/column indices from the *observed*
        # unique x/y values among survivors, which only lines up with this
        # test's virtual full-grid row/col positions if none of the
        # full grid's rows/columns are entirely masked out.
        keep = rng.random((ny, nx)) > 0.35
        empty_cols = np.flatnonzero(~keep.any(axis=0))
        keep[rng.integers(0, ny, size=empty_cols.size), empty_cols] = True
        empty_rows = np.flatnonzero(~keep.any(axis=1))
        keep[empty_rows, rng.integers(0, nx, size=empty_rows.size)] = True
        rows, cols = np.nonzero(keep)  # row-major order, matching Grid's own point order

        x = min_x + cols.astype(np.float64) * dx
        y = min_y + rows.astype(np.float64) * dy
        points = np.column_stack([x, y])

        spec = normalize_masked_regular_grid_spec(points)
        assert spec is not None

        expected_rows, expected_x_starts, expected_counts = _reference_normalize_masked_regular_grid_spec(
            cols.astype(np.int64), rows.astype(np.int64)
        )

        np.testing.assert_array_equal(spec.row_indices, expected_rows)
        np.testing.assert_array_equal(spec.x_start_indices, expected_x_starts)
        np.testing.assert_array_equal(spec.counts, expected_counts)
        assert spec.point_count == points.shape[0]
        assert int(np.sum(spec.counts)) == points.shape[0]


def test_normalize_masked_regular_grid_spec_single_point() -> None:
    spec = normalize_masked_regular_grid_spec(np.array([[5.0, 7.0]]))
    assert spec is not None
    assert spec.point_count == 1
    np.testing.assert_array_equal(spec.row_indices, [0])
    np.testing.assert_array_equal(spec.x_start_indices, [0])
    np.testing.assert_array_equal(spec.counts, [1])


if __name__ == "__main__":
    test_extract_target_xy_from_ndarray()
    test_extract_target_xy_from_list_of_points()
    test_extract_target_xy_from_geoseries()
    test_extract_target_xy_rejects_non_point_geometry()
    test_extract_target_xy_rejects_non_geometry_objects()
    test_normalize_masked_regular_grid_spec_matches_reference_rle_fuzz()
    test_normalize_masked_regular_grid_spec_single_point()
