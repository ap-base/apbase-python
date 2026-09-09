from __future__ import annotations

import numpy as np
import shapely

import apbase
from apbase.mapping import Map, MapResult, create_map


def _synthetic_source(
    seed: int, n: int, extent: float = 100.0, offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = offset + rng.uniform(0.0, extent, n)
    y = offset + rng.uniform(0.0, extent, n)
    z = np.sin((x - offset) / 12.0) + np.cos((y - offset) / 12.0) + rng.normal(0.0, 0.05, n)
    return x, y, z


def test_create_map_returns_compact_point_list_with_expected_shapes() -> None:
    x, y, z = _synthetic_source(seed=0, n=300)

    result = create_map(x, y, z, resolution=8.0)

    assert isinstance(result, MapResult)
    assert result.method in ("idw", "kriging")
    assert result.x.ndim == 1 and result.y.ndim == 1 and result.z.ndim == 1
    assert result.x.size == result.y.size == result.z.size
    assert result.x.size > 0
    assert result.n_source_points <= x.size
    assert result.radius > 0.0
    assert result.resolution == 8.0


def test_create_map_requires_resolution() -> None:
    x, y, z = _synthetic_source(seed=1, n=10)
    try:
        create_map(x, y, z)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError when resolution is omitted")

    try:
        Map(x, y, z)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError when resolution is omitted for Map too")


def test_map_requires_x_y_z_together() -> None:
    x, y, z = _synthetic_source(seed=1, n=10)
    try:
        Map(x, resolution=10.0)
    except ValueError as exc:
        assert "together" in str(exc)
    else:
        raise AssertionError("expected ValueError when x is given without y/z")


def test_map_filter_toggle_changes_source_points_and_statistics() -> None:
    x, y, z = _synthetic_source(seed=2, n=200)
    z = z.copy()
    z[0] = 1000.0  # obvious outlier for the default global-IQR filter

    filtered = create_map(x, y, z, resolution=10.0, filter=True)
    unfiltered = create_map(x, y, z, resolution=10.0, filter=False)

    assert filtered.filter_statistics is not None
    assert unfiltered.filter_statistics is None
    assert unfiltered.n_source_points == x.size
    assert filtered.n_source_points <= x.size


def test_map_selects_method_matching_lower_cross_validation_rmse() -> None:
    x, y, z = _synthetic_source(seed=3, n=250, extent=300.0)

    result = create_map(x, y, z, resolution=15.0, filter=False)

    # Independently re-derive the same shared variogram/radius the pipeline
    # fits internally, and confirm the automatic pick matches the lower
    # leave-one-out RMSE from cross_validate on identical inputs.
    variogram = apbase.Variogram(n_lags=50, max_pairs=100_000, max_distance=0.0).fit(x, y, z)
    radius = float(variogram.model_params["range"]) / 3.0
    cv = apbase.cross_validate(
        x,
        y,
        z,
        radius=radius,
        power=2.0,
        max_neighbors=40,
        min_neighbors=3,
        model_values=variogram.model_values,
    )
    expected_method = "idw" if cv.idw.rmse <= cv.kriging.rmse else "kriging"

    assert result.method == expected_method
    assert result.cross_validation.idw.rmse == cv.idw.rmse
    assert result.cross_validation.kriging.rmse == cv.kriging.rmse


def test_map_bounds_accepts_geometry_wkt_and_wkb_equally() -> None:
    # origin offset keeps boundary/x/y out of the [-180, 180] x [-90, 90]
    # range guess_xy_coordinate_system() reads as geographic -- an
    # un-offset box(10, 10, 60, 60) sits entirely inside it and gets
    # flagged as a CRS mismatch against the (mostly) non-geographic data.
    origin = 10_000.0
    x, y, z = _synthetic_source(seed=4, n=150, offset=origin)
    boundary = shapely.box(origin + 10.0, origin + 10.0, origin + 60.0, origin + 60.0)

    from_geometry = create_map(x, y, z, resolution=8.0, bounds=boundary, filter=False)
    from_wkt = create_map(x, y, z, resolution=8.0, bounds=boundary.wkt, filter=False)
    from_wkb = create_map(x, y, z, resolution=8.0, bounds=shapely.to_wkb(boundary), filter=False)

    assert np.array_equal(from_geometry.x, from_wkt.x)
    assert np.array_equal(from_geometry.x, from_wkb.x)
    assert np.allclose(from_geometry.z, from_wkt.z, equal_nan=True)
    assert np.allclose(from_geometry.z, from_wkb.z, equal_nan=True)


def test_map_output_coordinates_preserve_input_metric_not_rebased() -> None:
    # Large, non-zero offset (UTM-like magnitude) so a silent rebase to a
    # local 0..max/pixel-index system would be immediately visible.
    offset = 500_000.0
    extent = 100.0
    boundary = shapely.box(offset, offset, offset + extent, offset + extent)
    x, y, z = _synthetic_source(seed=5, n=400, extent=extent, offset=offset)

    result = create_map(x, y, z, resolution=10.0, bounds=boundary, filter=False)

    assert result.x.min() >= offset - 10.0
    assert result.y.min() >= offset - 10.0
    assert result.x.max() <= offset + extent + 10.0
    assert result.y.max() <= offset + extent + 10.0

    x_raster, y_raster, _ = result.to_raster()
    assert np.nanmin(x_raster) >= offset - 10.0
    assert np.nanmin(y_raster) >= offset - 10.0


def test_map_to_raster_reconstructs_points_with_matching_coordinates_and_values() -> None:
    # Offset keeps x/y out of the [-180, 180] x [-90, 90] range that
    # guess_xy_coordinate_system() reads as geographic (lon/lat) -- an
    # in-range small extent like the un-offset [-5, 25] this used to produce
    # gets silently reprojected to UTM, turning resolution=5.0 into 5
    # *meters* over a ~2,000 km-wide domain (a ~450k x 450k point grid,
    # OOM). Same relative geometry as before (source overlaps the boundary
    # box with a 5-unit margin on each side), just shifted to an unambiguous
    # coordinate range.
    origin = 10_000.0
    boundary = shapely.box(origin, origin, origin + 20.0, origin + 20.0)
    x, y, z = _synthetic_source(seed=6, n=300, extent=30.0, offset=origin - 5.0)

    result = create_map(x, y, z, resolution=5.0, bounds=boundary, filter=False)
    x_raster, y_raster, z_raster = result.to_raster()

    assert x_raster.shape == y_raster.shape == z_raster.shape

    expected_points = set(
        zip(np.round(result.x, 6).tolist(), np.round(result.y, 6).tolist(), strict=True)
    )
    actual_points = set(
        zip(np.round(x_raster.ravel(), 6).tolist(), np.round(y_raster.ravel(), 6).tolist(), strict=True)
    )
    assert expected_points == actual_points

    point_to_z = {
        (round(float(px), 6), round(float(py), 6)): float(pz)
        for px, py, pz in zip(result.x.tolist(), result.y.tolist(), result.z.tolist(), strict=True)
    }
    for row in range(x_raster.shape[0]):
        for col in range(x_raster.shape[1]):
            key = (round(float(x_raster[row, col]), 6), round(float(y_raster[row, col]), 6))
            expected_z = point_to_z[key]
            if np.isnan(expected_z):
                assert np.isnan(z_raster[row, col])
            else:
                assert z_raster[row, col] == expected_z

    array3d = result.to_array3d()
    assert array3d.shape == (x_raster.shape[0], x_raster.shape[1], 3)
    assert np.allclose(array3d[..., 0], x_raster, equal_nan=True)
    assert np.allclose(array3d[..., 1], y_raster, equal_nan=True)
    assert np.allclose(array3d[..., 2], z_raster, equal_nan=True)


def test_map_eager_lazy_and_functional_forms_match() -> None:
    x, y, z = _synthetic_source(seed=7, n=150)

    eager = Map(x, y, z, resolution=6.0, filter=False)
    lazy = Map(resolution=6.0, filter=False).fit_generate(x, y, z)
    via_call = Map(resolution=6.0, filter=False)(x, y, z)
    via_function = create_map(x, y, z, resolution=6.0, filter=False)

    assert eager.result is not None
    assert np.array_equal(eager.result.x, lazy.x)
    assert np.array_equal(eager.result.x, via_call.x)
    assert np.array_equal(eager.result.x, via_function.x)
    assert np.allclose(eager.result.z, lazy.z, equal_nan=True)
    assert np.allclose(eager.result.z, via_call.z, equal_nan=True)
    assert np.allclose(eager.result.z, via_function.z, equal_nan=True)
    assert eager.result.method == lazy.method == via_call.method == via_function.method


if __name__ == "__main__":
    test_create_map_returns_compact_point_list_with_expected_shapes()
    test_create_map_requires_resolution()
    test_map_requires_x_y_z_together()
    test_map_filter_toggle_changes_source_points_and_statistics()
    test_map_selects_method_matching_lower_cross_validation_rmse()
    test_map_bounds_accepts_geometry_wkt_and_wkb_equally()
    test_map_output_coordinates_preserve_input_metric_not_rebased()
    test_map_to_raster_reconstructs_points_with_matching_coordinates_and_values()
    test_map_eager_lazy_and_functional_forms_match()
