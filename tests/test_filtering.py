from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pytest

from apbase.filtering import SpatialFilter


def test_spatial_filter_keeps_finite_constant_values_with_explicit_radius() -> None:
    x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0, np.nan])
    y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0, 10.0])
    z = np.ascontiguousarray([10.0, 10.0, 10.0, 10.0, 10.0])

    spatial_filter = SpatialFilter(
        radius=30.0,
        apply_global_iqr=False,
        max_neighbors=0,
        filter_level="light",
    ).fit(x, y, z)

    expected_mask = np.asarray([True, True, True, True, False])
    result_x, result_y, result_z = spatial_filter.filter()

    assert np.array_equal(spatial_filter.mask(), expected_mask)
    assert np.array_equal(result_x, x[:4])
    assert np.array_equal(result_y, y[:4])
    assert np.array_equal(result_z, z[:4])


def test_spatial_filter_dedup_radius_fixes_duplicate_clump_nan() -> None:
    # A stalled flow sensor emits many bit-identical readings at nearly the
    # same spot (a "duplicate clump"). When a nearby point's k-NN
    # neighborhood is dominated by that clump, local variance collapses to
    # exactly 0; since the point's own value is (almost) never bit-equal to
    # the collapsed mean, its local_z/local_prob come back NaN and it gets
    # dropped regardless of filter_level. dedup_radius collapses the clump
    # to one representative in the neighbor pool, restoring real variance.
    rng = np.random.default_rng(0)

    n_strip = 40
    x_strip = np.arange(n_strip, dtype=np.float64) * 5.0
    y_strip = np.zeros(n_strip)
    z_strip = 5.0 + rng.normal(0, 0.3, n_strip)

    n_clump = 25
    cx, cy = x_strip[20], y_strip[20]
    x_clump = cx + rng.uniform(-0.15, 0.15, n_clump)
    y_clump = cy + rng.uniform(-0.15, 0.15, n_clump)
    z_clump = np.full(n_clump, 5.4)

    x = np.ascontiguousarray(np.concatenate([x_strip, x_clump]))
    y = np.ascontiguousarray(np.concatenate([y_strip, y_clump]))
    z = np.ascontiguousarray(np.concatenate([z_strip, z_clump]))

    no_dedup = SpatialFilter(
        radius=15.0, apply_global_iqr=False, max_neighbors=20, filter_level="light"
    ).fit(x, y, z)
    _, local_z_no_dedup, _ = no_dedup.statistics
    assert np.isnan(local_z_no_dedup).any()

    with_dedup = SpatialFilter(
        radius=15.0,
        apply_global_iqr=False,
        max_neighbors=20,
        filter_level="light",
        dedup_radius=0.5,
    ).fit(x, y, z)
    _, local_z_with_dedup, _ = with_dedup.statistics
    assert not np.isnan(local_z_with_dedup).any()


def test_spatial_filter_dedup_radius_zero_matches_disabled_default() -> None:
    x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0, 5.0])
    y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0, 5.0])
    z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0, 6.0])

    baseline = SpatialFilter(radius=20.0, apply_global_iqr=False, max_neighbors=0).fit(x, y, z)
    explicit_zero = SpatialFilter(
        radius=20.0, apply_global_iqr=False, max_neighbors=0, dedup_radius=0.0
    ).fit(x, y, z)

    np.testing.assert_array_equal(baseline.mask(), explicit_zero.mask())
    baseline_stats = baseline.statistics
    zero_stats = explicit_zero.statistics
    for baseline_arr, zero_arr in zip(baseline_stats, zero_stats, strict=True):
        np.testing.assert_array_equal(baseline_arr, zero_arr)


def test_spatial_filter_dedup_radius_with_full_radius_neighbor_scan() -> None:
    # max_neighbors=0 takes the dense per-cell-sum aggregate path (a
    # different branch than the k-NN heap), which needed its own
    # dedup-aware neighbor count/sum bookkeeping (cell_rep_count) and a
    # gated self-subtraction so a non-representative target point doesn't
    # get double-subtracted from a sum it was never added to.
    x = np.ascontiguousarray([0.0, 0.05, 0.05, 20.0, 40.0])
    y = np.ascontiguousarray([0.0, 0.0, 0.05, 0.0, 0.0])
    z = np.ascontiguousarray([5.0, 5.0, 5.0, 9.0, 5.2])

    filtered = SpatialFilter(
        radius=100.0,
        apply_global_iqr=False,
        max_neighbors=0,
        filter_level="light",
        dedup_radius=1.0,
    ).fit(x, y, z)

    pct_diff, local_z, local_prob = filtered.statistics
    assert np.isfinite(pct_diff).all()
    assert np.isfinite(local_z).all()
    assert np.isfinite(local_prob).all()


def test_spatial_filter_uses_shared_variogram_radius() -> None:
    gx, gy = np.meshgrid(np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64))
    x = np.ascontiguousarray(gx.ravel())
    y = np.ascontiguousarray(gy.ravel())
    z = np.ascontiguousarray(np.sin(x * 0.3) + np.cos(y * 0.2))

    spatial_filter = SpatialFilter(
        apply_global_iqr=False,
        max_neighbors=16,
        filter_level="light",
    ).fit(x, y, z)

    assert spatial_filter.model_params["range"] > 0.0
    assert spatial_filter.mask().shape == x.shape
    assert spatial_filter.mask().dtype == np.bool_


def test_spatial_filter_knn_local_stats_stable_for_large_mean_small_variance() -> None:
    # Yield-monitor-scale values (mean ~150) with a tiny local spread
    # (std ~1e-4): mean**2 (~22500) dwarfs the true variance (~1e-8) enough
    # that a naive sum_sq/count - mean**2 computation loses essentially all
    # significant digits -- and once the (correct, if tiny) variance falls
    # below the old cancellation-tolerance floor (1e-12 * mean**2 ~ 2.25e-8
    # here), it used to get clamped straight to zero, which for a non-exact
    # match between z(point) and the mean left local_z/local_prob at their
    # NaN initial value entirely (the same "duplicate clump" failure mode as
    # test_spatial_filter_dedup_radius_fixes_duplicate_clump_nan above).
    # Welford's online algorithm (mean + M2), used on the k-NN heap path,
    # does not have this problem: it should match a straightforward NumPy
    # population-variance reference closely.
    #
    # The extent (5900) must exceed radius (250) so the native code takes
    # the spatial-grid k-NN heap path, not the "whole dataset fits in one
    # radius" global-sum shortcut (a separate, still sum_sq-based code path
    # this test does not cover).
    rng = np.random.default_rng(7)
    n = 60
    x = np.arange(n, dtype=np.float64) * 100.0
    y = np.zeros(n)
    z = np.ascontiguousarray(150.0 + rng.normal(0.0, 1.0e-4, n))
    radius = 250.0

    spatial_filter = SpatialFilter(
        radius=radius,
        apply_global_iqr=False,
        max_neighbors=8,  # comfortably above the <=4 neighbors any point actually has within radius
        filter_level="light",
    ).fit(x, y, z)
    _, local_z, local_prob = spatial_filter.statistics

    assert not np.isnan(local_z).any(), "local_z collapsed to NaN under cancellation"
    assert not np.isnan(local_prob).any()

    for i in range(n):
        within = np.abs(x - x[i]) <= radius
        within[i] = False
        neighbor_z = z[within]
        assert 0 < neighbor_z.size <= 4  # sanity: never hits the max_neighbors=8 cap
        expected_mean = float(neighbor_z.mean())
        expected_var = float(neighbor_z.var())  # population variance (ddof=0), matches the native formula
        expected_z = (z[i] - expected_mean) / sqrt(expected_var)
        expected_prob = erfc(abs(expected_z) / sqrt(2.0))
        assert local_z[i] == pytest.approx(expected_z, rel=1e-6, abs=1e-9)
        assert local_prob[i] == pytest.approx(expected_prob, rel=1e-6, abs=1e-9)


if __name__ == "__main__":
    test_spatial_filter_keeps_finite_constant_values_with_explicit_radius()
    test_spatial_filter_uses_shared_variogram_radius()
    test_spatial_filter_knn_local_stats_stable_for_large_mean_small_variance()
