from __future__ import annotations

import numpy as np

from apbase.idw import IDW


def _reference_idw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: np.ndarray,
    radius: float,
    power: float,
) -> np.ndarray:
    out = np.full(grid.shape[0], np.nan, dtype=np.float64)
    for i, (tx, ty) in enumerate(grid):
        d2 = (x - tx) ** 2 + (y - ty) ** 2
        exact = d2 <= 1.0e-24
        if np.any(exact):
            out[i] = z[np.argmax(exact)]
            continue

        mask = d2 <= radius * radius
        if not np.any(mask):
            continue

        weight = d2[mask] ** (-0.5 * power)
        out[i] = float(np.sum(weight * z[mask]) / np.sum(weight))
    return out


def test_idw_matches_reference() -> None:
    x = np.ascontiguousarray([500000.0, 500010.0, 500000.0, 500010.0])
    y = np.ascontiguousarray([7000000.0, 7000000.0, 7000010.0, 7000010.0])
    z = np.ascontiguousarray([0.0, 10.0, 20.0, 30.0])
    grid = np.ascontiguousarray(
        [
            [500005.0, 7000005.0],
            [500000.0, 7000000.0],
            [500020.0, 7000020.0],
        ],
    )

    result = IDW(
        radius=30.0,
        power=2.0,
        min_neighbors=1,
        max_neighbors=4,
    ).fit(x, y, z).interpolate(grid)
    expected = _reference_idw(x, y, z, grid, radius=30.0, power=2.0)

    assert np.allclose(result, expected, equal_nan=True)


def test_idw_matches_reference_with_sparse_cell_hash_fallback() -> None:
    # Several small clusters spread far apart relative to `radius` push the
    # native grid's cell count well past the dense-lookup cap, exercising the
    # hash-based sparse cell index instead of the direct dense-array path.
    centers = np.ascontiguousarray(
        [[0.0, 0.0], [50000.0, 0.0], [0.0, 50000.0], [50000.0, 50000.0], [25000.0, 25000.0]],
    )
    offsets = np.ascontiguousarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])

    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    for cluster_index, (cx, cy) in enumerate(centers):
        for offset_index, (ox, oy) in enumerate(offsets):
            x_values.append(cx + ox)
            y_values.append(cy + oy)
            z_values.append(float(cluster_index * 10 + offset_index))
    x = np.ascontiguousarray(x_values)
    y = np.ascontiguousarray(y_values)
    z = np.ascontiguousarray(z_values)
    grid = np.ascontiguousarray(centers + 1.0)
    radius = 10.0
    power = 2.0

    result = (
        IDW(radius=radius, power=power, min_neighbors=1, max_neighbors=40)
        .fit(x, y, z)
        .interpolate(grid)
    )
    expected = _reference_idw(x, y, z, grid, radius=radius, power=power)

    assert np.all(np.isfinite(expected))
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_idw_matches_reference()
    test_idw_matches_reference_with_sparse_cell_hash_fallback()
