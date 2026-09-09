from __future__ import annotations

import numpy as np

from apbase.kriging import Kriging, ordinary_kriging

MODEL_SPHERICAL = 1
MODEL_EXPONENTIAL = 2
MODEL_GAUSSIAN = 3


def _variogram_shape(h: np.ndarray, model_id: int, model_range: float) -> np.ndarray:
    ratio = h / model_range
    if model_id == MODEL_SPHERICAL:
        clipped = np.clip(ratio, 0.0, 1.0)
        return np.where(ratio >= 1.0, 1.0, 1.5 * clipped - 0.5 * clipped**3)
    if model_id == MODEL_EXPONENTIAL:
        return 1.0 - np.exp(-3.0 * ratio)
    if model_id == MODEL_GAUSSIAN:
        return 1.0 - np.exp(-3.0 * ratio**2)
    raise ValueError(f"unsupported model_id: {model_id}")


def _variogram_value(
    h: np.ndarray, model_id: int, nugget: float, partial_sill: float, model_range: float
) -> np.ndarray:
    return nugget + partial_sill * _variogram_shape(h, model_id, model_range)


def _reference_ordinary_kriging(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid: np.ndarray,
    nugget: float,
    partial_sill: float,
    model_range: float,
    model_id: int = MODEL_SPHERICAL,
) -> np.ndarray:
    sill = nugget + partial_sill
    n = x.size
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    h = np.sqrt(dx**2 + dy**2)
    gamma = _variogram_value(h, model_id, nugget, partial_sill, model_range)
    covariance = np.where(h <= 0.0, sill, sill - gamma)

    system = np.ones((n + 1, n + 1))
    system[:n, :n] = covariance
    system[n, n] = 0.0

    out = np.empty(grid.shape[0], dtype=np.float64)
    for i, (tx, ty) in enumerate(grid):
        h0 = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
        c0 = np.where(
            h0 <= 0.0, sill, sill - _variogram_value(h0, model_id, nugget, partial_sill, model_range)
        )
        rhs = np.append(c0, 1.0)
        weights = np.linalg.solve(system, rhs)
        out[i] = float(np.dot(weights[:n], z))
    return out


def _model_values(
    nugget: float, partial_sill: float, model_range: float, model_id: int = MODEL_SPHERICAL
) -> np.ndarray:
    return np.ascontiguousarray([float(model_id), nugget, partial_sill, model_range, 0.0])


def test_kriging_matches_reference_ordinary_kriging() -> None:
    x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0, 4.0, 7.0])
    y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0, 6.0, 2.0])
    z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0, 6.5, 6.0])
    grid = np.ascontiguousarray([[5.0, 5.0], [2.0, 8.0], [9.0, 1.0]])
    nugget, partial_sill, model_range = 0.5, 4.0, 30.0

    result = Kriging(
        model_values=_model_values(nugget, partial_sill, model_range),
        radius=model_range,
        max_neighbors=10,
        min_neighbors=1,
    ).fit(x, y, z).interpolate(grid)
    expected = _reference_ordinary_kriging(x, y, z, grid, nugget, partial_sill, model_range)

    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_kriging_matches_reference_for_exponential_and_gaussian_models() -> None:
    x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0, 4.0, 7.0])
    y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0, 6.0, 2.0])
    z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0, 6.5, 6.0])
    grid = np.ascontiguousarray([[5.0, 5.0], [2.0, 8.0], [9.0, 1.0]])
    nugget, partial_sill, model_range = 0.5, 4.0, 30.0

    for model_id in (MODEL_EXPONENTIAL, MODEL_GAUSSIAN):
        result = Kriging(
            model_values=_model_values(nugget, partial_sill, model_range, model_id),
            radius=model_range,
            max_neighbors=10,
            min_neighbors=1,
        ).fit(x, y, z).interpolate(grid)
        expected = _reference_ordinary_kriging(
            x, y, z, grid, nugget, partial_sill, model_range, model_id,
        )
        assert np.allclose(result, expected, rtol=1e-6, atol=1e-6), model_id


def test_kriging_matches_reference_with_sparse_cell_hash_fallback() -> None:
    # Several small clusters spread far apart relative to `radius` push the
    # native grid's cell count well past the dense-lookup cap, exercising the
    # hash-based sparse cell index instead of the direct dense-array path.
    # `_reference_ordinary_kriging` solves a *global* system (no radius
    # cutoff), so each cluster's expected values are computed independently
    # from only that cluster's points -- valid here because the clusters are
    # far enough apart that no other cluster's points ever fall within
    # `radius` of a target, matching what the local native kernel sees.
    centers = np.ascontiguousarray(
        [[0.0, 0.0], [50000.0, 0.0], [0.0, 50000.0], [50000.0, 50000.0], [25000.0, 25000.0]],
    )
    offsets = np.ascontiguousarray(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0], [1.0, 0.0]],
    )
    nugget, partial_sill, model_range = 0.5, 4.0, 8.0
    radius = model_range

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    expected_parts: list[np.ndarray] = []
    targets: list[list[float]] = []
    for cluster_index, (cx, cy) in enumerate(centers):
        cluster_x = np.ascontiguousarray(cx + offsets[:, 0])
        cluster_y = np.ascontiguousarray(cy + offsets[:, 1])
        cluster_z = np.ascontiguousarray(
            [float(cluster_index * 3 + offset_index) for offset_index in range(len(offsets))],
        )
        target = np.ascontiguousarray([[cx + 1.0, cy + 1.0]])
        expected_parts.append(
            _reference_ordinary_kriging(
                cluster_x, cluster_y, cluster_z, target, nugget, partial_sill, model_range,
            ),
        )
        x_parts.append(cluster_x)
        y_parts.append(cluster_y)
        z_parts.append(cluster_z)
        targets.append([cx + 1.0, cy + 1.0])

    x = np.ascontiguousarray(np.concatenate(x_parts))
    y = np.ascontiguousarray(np.concatenate(y_parts))
    z = np.ascontiguousarray(np.concatenate(z_parts))
    grid = np.ascontiguousarray(targets)
    expected = np.concatenate(expected_parts)

    result = (
        Kriging(
            model_values=_model_values(nugget, partial_sill, model_range),
            radius=radius,
            max_neighbors=10,
            min_neighbors=1,
        )
        .fit(x, y, z)
        .interpolate(grid)
    )

    assert np.all(np.isfinite(expected))
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_kriging_exact_match_returns_source_value() -> None:
    x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0])
    y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0])
    z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0])
    grid = np.ascontiguousarray([[10.0, 0.0], [3.0, 3.0]])

    result = Kriging(
        model_values=_model_values(1.0, 5.0, 30.0),
        radius=30.0,
        max_neighbors=4,
        min_neighbors=1,
    ).fit(x, y, z).interpolate(grid)

    assert result[0] == 7.0


def test_kriging_returns_nan_below_min_neighbors() -> None:
    x = np.ascontiguousarray([0.0, 100.0])
    y = np.ascontiguousarray([0.0, 100.0])
    z = np.ascontiguousarray([1.0, 2.0])
    grid = np.ascontiguousarray([[1.0, 1.0]])

    result = Kriging(
        model_values=_model_values(1.0, 5.0, 30.0),
        radius=5.0,
        max_neighbors=4,
        min_neighbors=2,
    ).fit(x, y, z).interpolate(grid)

    assert np.isnan(result[0])


def test_kriging_ignores_non_finite_source_rows() -> None:
    x = np.ascontiguousarray([0.0, 10.0, np.nan, 10.0])
    y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0])
    z = np.ascontiguousarray([5.0, 7.0, 8.0, np.inf])

    kriging = Kriging(
        model_values=_model_values(1.0, 5.0, 30.0),
        radius=30.0,
        max_neighbors=4,
        min_neighbors=1,
    ).fit(x, y, z)

    assert kriging._x.size == 2  # noqa: SLF001


def test_kriging_rejects_non_positive_radius() -> None:
    try:
        Kriging(
            model_values=_model_values(1.0, 5.0, 30.0),
            radius=0.0,
            min_neighbors=1,
        )
    except ValueError as exc:
        assert "radius" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-positive radius")


def test_ordinary_kriging_free_function_matches_class() -> None:
    x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0, 4.0, 7.0])
    y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0, 6.0, 2.0])
    z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0, 6.5, 6.0])
    grid = np.ascontiguousarray([[5.0, 5.0], [2.0, 8.0]])
    model_values = _model_values(0.5, 4.0, 30.0)

    from_function = ordinary_kriging(
        x, y, z, grid,
        model_values=model_values,
        radius=30.0,
        max_neighbors=10,
        min_neighbors=1,
    )
    from_class = Kriging(
        model_values=model_values,
        radius=30.0,
        max_neighbors=10,
        min_neighbors=1,
    ).fit(x, y, z).interpolate(grid)

    assert np.array_equal(from_function, from_class)


if __name__ == "__main__":
    test_kriging_matches_reference_ordinary_kriging()
    test_kriging_matches_reference_for_exponential_and_gaussian_models()
    test_kriging_exact_match_returns_source_value()
    test_kriging_returns_nan_below_min_neighbors()
    test_kriging_ignores_non_finite_source_rows()
    test_kriging_rejects_non_positive_radius()
    test_ordinary_kriging_free_function_matches_class()
