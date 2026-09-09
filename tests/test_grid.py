from __future__ import annotations

import numpy as np
import shapely

from apbase.grid import Grid, create_point_grid


def test_grid_matches_regular_mesh_inside_box_boundary() -> None:
    boundary = shapely.box(0.0, 0.0, 10.0, 10.0)
    grid = Grid(boundary=boundary, resolution=5.0).fit()

    expected = np.array(
        [
            [0.0, 0.0], [5.0, 0.0], [10.0, 0.0],
            [0.0, 5.0], [5.0, 5.0], [10.0, 5.0],
            [0.0, 10.0], [5.0, 10.0], [10.0, 10.0],
        ],
    )
    assert np.array_equal(np.asarray(sorted(grid.points.tolist())), np.asarray(sorted(expected.tolist())))
    assert grid.points.flags["F_CONTIGUOUS"]


def test_grid_fit_ignores_xy_when_boundary_given() -> None:
    boundary = shapely.box(0.0, 0.0, 10.0, 10.0)
    no_xy = Grid(boundary=boundary, resolution=5.0).fit().points
    with_xy = Grid(boundary=boundary, resolution=5.0).fit(np.array([0.0]), np.array([0.0])).points

    assert np.array_equal(no_xy, with_xy)


def test_grid_fit_requires_xy_when_boundary_missing() -> None:
    try:
        Grid(resolution=5.0).fit()
    except ValueError as exc:
        assert "x and y" in str(exc)
    else:
        raise AssertionError("expected ValueError when neither boundary nor x/y are given")


def test_grid_point_count_is_independent_of_chunk_size() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1000, 200)
    y = rng.uniform(0, 1000, 200)

    small_chunks = Grid(resolution=10.0, chunk_size=50).fit(x, y).points
    large_chunks = Grid(resolution=10.0, chunk_size=1_000_000).fit(x, y).points

    assert small_chunks.shape[0] == large_chunks.shape[0]
    assert np.array_equal(small_chunks, large_chunks)


def test_grid_infers_concave_hull_boundary_when_none_given() -> None:
    x = np.ascontiguousarray([0.0, 100.0, 0.0, 100.0])
    y = np.ascontiguousarray([0.0, 0.0, 100.0, 100.0])

    grid = Grid(resolution=10.0).fit(x, y)

    assert grid.points.shape[0] > 0
    assert grid.fitted_boundary is not None
    # every source point must fall inside its own inferred (buffered) boundary
    assert all(shapely.intersects_xy(grid.fitted_boundary, x, y))


def test_create_point_grid_matches_grid_class() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 500, 100)
    y = rng.uniform(0, 500, 100)

    from_function = create_point_grid(x=x, y=y, resolution=25.0)
    from_class = Grid(resolution=25.0).fit(x, y).points

    assert np.array_equal(from_function, from_class)


def test_create_point_grid_from_boundary_without_xy() -> None:
    boundary = shapely.box(0.0, 0.0, 10.0, 10.0)

    from_function = create_point_grid(boundary=boundary, resolution=5.0)
    from_class = Grid(boundary=boundary, resolution=5.0).fit().points

    assert np.array_equal(from_function, from_class)


def test_grid_from_data_xy_matches_fit() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 500, 100)
    y = rng.uniform(0, 500, 100)

    from_classmethod = Grid.from_data_xy(x, y, resolution=25.0).points
    from_fit = Grid(resolution=25.0).fit(x, y).points

    assert np.array_equal(from_classmethod, from_fit)


def test_grid_from_wkt_matches_boundary_grid() -> None:
    boundary = shapely.box(0.0, 0.0, 10.0, 10.0)

    from_wkt = Grid.from_wkt(boundary.wkt, resolution=5.0).points
    from_boundary = Grid(boundary=boundary, resolution=5.0).fit().points

    assert np.array_equal(from_wkt, from_boundary)


def test_grid_from_wkb_matches_boundary_grid() -> None:
    boundary = shapely.box(0.0, 0.0, 10.0, 10.0)

    from_wkb = Grid.from_wkb(shapely.to_wkb(boundary), resolution=5.0).points
    from_boundary = Grid(boundary=boundary, resolution=5.0).fit().points

    assert np.array_equal(from_wkb, from_boundary)


if __name__ == "__main__":
    test_grid_matches_regular_mesh_inside_box_boundary()
    test_grid_fit_ignores_xy_when_boundary_given()
    test_grid_fit_requires_xy_when_boundary_missing()
    test_grid_point_count_is_independent_of_chunk_size()
    test_grid_infers_concave_hull_boundary_when_none_given()
    test_create_point_grid_matches_grid_class()
    test_create_point_grid_from_boundary_without_xy()
    test_grid_from_data_xy_matches_fit()
    test_grid_from_wkt_matches_boundary_grid()
    test_grid_from_wkb_matches_boundary_grid()
