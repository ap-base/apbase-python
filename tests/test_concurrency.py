"""Regression tests for concurrent top-level calls into the native extension.

idw_local, cross_validate_local, ordinary_kriging_local and
viz_zstats_radius_gridagg all reuse a `SAVE`d per-thread neighbor workspace
across calls for performance. Before the fix these tests guard, the OMP
critical section only protected the (re)allocation decision, not the actual
read/write usage in the parallel loop -- so two top-level calls overlapping
with matching `max_neighbors`/`thread_count` (no reallocation needed) could
write into the same shared columns and silently corrupt each other's result.
These tests call the same code path from multiple Python threads at once,
repeatedly, and check every result against a sequential reference.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from apbase.cross_validate import cross_validate
from apbase.idw import IDW

_ROUNDS = 5
_CALLS_PER_ROUND = 8
_MAX_WORKERS = 4


def test_idw_interpolate_is_safe_under_concurrent_top_level_calls() -> None:
    def make_model(seed: int, z_offset: float) -> tuple[IDW, np.ndarray]:
        local_rng = np.random.default_rng(seed)
        x = np.ascontiguousarray(local_rng.uniform(0.0, 1000.0, size=300))
        y = np.ascontiguousarray(local_rng.uniform(0.0, 1000.0, size=300))
        z = np.ascontiguousarray(local_rng.uniform(0.0, 50.0, size=300) + z_offset)
        grid = np.ascontiguousarray(local_rng.uniform(0.0, 1000.0, size=(200, 2)))
        # Same radius/max_neighbors/min_neighbors/n_threads across both models
        # so the shared native workspace is reused as-is (no reallocation) --
        # exactly the path that used to race.
        model = IDW(radius=80.0, power=2.0, min_neighbors=1, max_neighbors=16).fit(x, y, z)
        return model, grid

    model_a, grid_a = make_model(1, z_offset=0.0)
    model_b, grid_b = make_model(2, z_offset=1000.0)  # disjoint value range: corruption is obvious

    expected_a = model_a.interpolate(grid_a)
    expected_b = model_b.interpolate(grid_b)
    assert np.any(np.isfinite(expected_a))
    assert np.any(np.isfinite(expected_b))

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for _ in range(_ROUNDS):
            futures = [
                pool.submit(model_a.interpolate, grid_a)
                if i % 2 == 0
                else pool.submit(model_b.interpolate, grid_b)
                for i in range(_CALLS_PER_ROUND)
            ]
            results = [future.result() for future in futures]

            for i, result in enumerate(results):
                expected = expected_a if i % 2 == 0 else expected_b
                assert np.allclose(result, expected, equal_nan=True), (
                    "concurrent idw_local call diverged from the sequential reference -- "
                    "likely a data race on the shared neighbor workspace"
                )


def test_cross_validate_is_safe_under_concurrent_top_level_calls() -> None:
    def make_dataset(seed: int, z_offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_rng = np.random.default_rng(seed)
        x = np.ascontiguousarray(local_rng.uniform(0.0, 1000.0, size=200))
        y = np.ascontiguousarray(local_rng.uniform(0.0, 1000.0, size=200))
        z = np.ascontiguousarray(local_rng.uniform(0.0, 50.0, size=200) + z_offset)
        return x, y, z

    x_a, y_a, z_a = make_dataset(1, z_offset=0.0)
    x_b, y_b, z_b = make_dataset(2, z_offset=1000.0)

    # Same radius/max_neighbors/min_neighbors/power across both calls so the
    # shared native workspace is reused as-is (no reallocation). An explicit
    # radius also skips each call's internal variogram fit, keeping the two
    # calls' native argument shapes identical.
    common_kwargs = dict(radius=80.0, power=2.0, max_neighbors=16, min_neighbors=1, seed=0)

    def run_a():
        return cross_validate(x_a, y_a, z_a, **common_kwargs)

    def run_b():
        return cross_validate(x_b, y_b, z_b, **common_kwargs)

    expected_a = run_a()
    expected_b = run_b()

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for _ in range(_ROUNDS):
            futures = [
                pool.submit(run_a) if i % 2 == 0 else pool.submit(run_b)
                for i in range(_CALLS_PER_ROUND)
            ]
            results = [future.result() for future in futures]

            for i, result in enumerate(results):
                expected = expected_a if i % 2 == 0 else expected_b
                assert np.allclose(result.idw.rmse, expected.idw.rmse, equal_nan=True)
                assert np.allclose(result.kriging.rmse, expected.kriging.rmse, equal_nan=True)
                assert np.allclose(result.idw.predicted, expected.idw.predicted, equal_nan=True), (
                    "concurrent cross_validate_local call diverged from the sequential reference -- "
                    "likely a data race on the shared neighbor workspace"
                )


if __name__ == "__main__":
    test_idw_interpolate_is_safe_under_concurrent_top_level_calls()
    test_cross_validate_is_safe_under_concurrent_top_level_calls()
