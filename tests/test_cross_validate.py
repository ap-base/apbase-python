from __future__ import annotations

import time

import numpy as np

from apbase.cross_validate import (
    BestModelResult,
    CrossValidationResult,
    MethodCrossValidation,
    MultiMethodBestModelResult,
    cross_validate,
    cross_validate_cokriging,
    select_best_model,
)

MODEL_SPHERICAL = 1


def _variogram_shape(h: np.ndarray, model_range: float) -> np.ndarray:
    ratio = np.clip(h / model_range, 0.0, 1.0)
    return np.where(h / model_range >= 1.0, 1.0, 1.5 * ratio - 0.5 * ratio**3)


def _variogram_value(h: np.ndarray, nugget: float, partial_sill: float, model_range: float) -> np.ndarray:
    return nugget + partial_sill * _variogram_shape(h, model_range)


def _reference_kriging_point(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, tx: float, ty: float,
    nugget: float, partial_sill: float, model_range: float,
) -> float:
    sill = nugget + partial_sill
    n = x.size
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    h = np.sqrt(dx**2 + dy**2)
    covariance = np.where(h <= 0.0, sill, sill - _variogram_value(h, nugget, partial_sill, model_range))

    system = np.ones((n + 1, n + 1))
    system[:n, :n] = covariance
    system[n, n] = 0.0

    h0 = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
    c0 = np.where(h0 <= 0.0, sill, sill - _variogram_value(h0, nugget, partial_sill, model_range))
    rhs = np.append(c0, 1.0)
    weights = np.linalg.solve(system, rhs)
    return float(np.dot(weights[:n], z))


def _reference_loocv(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, radius: float, power: float,
    nugget: float, partial_sill: float, model_range: float, min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.size
    idw_out = np.full(n, np.nan)
    krig_out = np.full(n, np.nan)
    for i in range(n):
        mask = np.arange(n) != i
        xs, ys, zs = x[mask], y[mask], z[mask]
        d2 = (xs - x[i]) ** 2 + (ys - y[i]) ** 2
        within = d2 <= radius * radius
        if np.count_nonzero(within) < min_neighbors:
            continue

        exact = within & (d2 <= 1.0e-24)
        if np.any(exact):
            value = float(zs[np.argmax(exact)])
            idw_out[i] = value
            krig_out[i] = value
            continue

        d2_sel = d2[within]
        z_sel = zs[within]
        weight = d2_sel ** (-0.5 * power)
        idw_out[i] = float(np.sum(weight * z_sel) / np.sum(weight))

        krig_out[i] = _reference_kriging_point(
            xs[within], ys[within], z_sel, x[i], y[i], nugget, partial_sill, model_range,
        )
    return idw_out, krig_out


def _model_values(nugget: float, partial_sill: float, model_range: float) -> np.ndarray:
    return np.ascontiguousarray([float(MODEL_SPHERICAL), nugget, partial_sill, model_range, 0.0])


def test_cross_validate_matches_reference_loocv() -> None:
    rng = np.random.default_rng(0)
    n = 30
    x = np.ascontiguousarray(rng.uniform(0.0, 100.0, n))
    y = np.ascontiguousarray(rng.uniform(0.0, 100.0, n))
    z = np.ascontiguousarray(5.0 + 0.1 * x + 0.2 * y + rng.normal(0.0, 0.5, n))
    nugget, partial_sill, model_range = 0.5, 4.0, 200.0
    radius = 200.0

    result = cross_validate(
        x, y, z,
        radius=radius,
        power=2.0,
        max_neighbors=n,
        min_neighbors=1,
        model_values=_model_values(nugget, partial_sill, model_range),
    )

    expected_idw, expected_krig = _reference_loocv(
        x, y, z, radius, 2.0, nugget, partial_sill, model_range, min_neighbors=1,
    )

    # cross_validate uses all n points as samples since n <= default max_points,
    # but sample order comes from arange(1, n+1), matching index order 0..n-1.
    assert result.n_samples == n
    assert np.allclose(result.idw.predicted, expected_idw, rtol=1e-6, atol=1e-6)
    assert np.allclose(result.kriging.predicted, expected_krig, rtol=1e-6, atol=1e-6)

    expected_idw_mae = float(np.mean(np.abs(expected_idw - z)))
    expected_idw_rmse = float(np.sqrt(np.mean((expected_idw - z) ** 2)))
    assert np.isclose(result.idw.mae, expected_idw_mae, rtol=1e-6)
    assert np.isclose(result.idw.rmse, expected_idw_rmse, rtol=1e-6)

    expected_krig_mae = float(np.mean(np.abs(expected_krig - z)))
    expected_krig_rmse = float(np.sqrt(np.mean((expected_krig - z) ** 2)))
    assert np.isclose(result.kriging.mae, expected_krig_mae, rtol=1e-6)
    assert np.isclose(result.kriging.rmse, expected_krig_rmse, rtol=1e-6)

    ss_tot = float(np.sum((z - z.mean()) ** 2))
    expected_idw_r2 = 1.0 - float(np.sum((expected_idw - z) ** 2)) / ss_tot
    expected_krig_r2 = 1.0 - float(np.sum((expected_krig - z) ** 2)) / ss_tot
    assert np.isclose(result.idw.r2, expected_idw_r2, rtol=1e-6)
    assert np.isclose(result.kriging.r2, expected_krig_r2, rtol=1e-6)


def test_cross_validate_respects_min_neighbors_nan() -> None:
    # Two isolated points far from everything else: with a small radius and
    # min_neighbors=2, every point (including the isolated pair) should end
    # up with < min_neighbors neighbors in at least some folds.
    x = np.ascontiguousarray([0.0, 1.0, 2.0, 1000.0, 1001.0])
    y = np.ascontiguousarray([0.0, 0.0, 0.0, 0.0, 0.0])
    z = np.ascontiguousarray([1.0, 2.0, 3.0, 4.0, 5.0])

    result = cross_validate(
        x, y, z,
        radius=5.0,
        max_neighbors=4,
        min_neighbors=2,
        model_values=_model_values(0.5, 4.0, 30.0),
    )

    # The two far points (indices 3, 4) only have each other within radius=5,
    # so leave-one-out drops each below min_neighbors=2 -> NaN prediction.
    assert np.isnan(result.idw.predicted[3])
    assert np.isnan(result.idw.predicted[4])
    assert np.isnan(result.kriging.predicted[3])
    assert np.isnan(result.kriging.predicted[4])
    assert result.idw.n_valid == 3
    assert result.kriging.n_valid == 3


def test_cross_validate_subsamples_to_max_points() -> None:
    rng = np.random.default_rng(1)
    n = 800
    x = np.ascontiguousarray(rng.uniform(0.0, 1000.0, n))
    y = np.ascontiguousarray(rng.uniform(0.0, 1000.0, n))
    z = np.ascontiguousarray(rng.normal(0.0, 1.0, n))

    result = cross_validate(x, y, z, max_points=200, seed=0, max_neighbors=20, min_neighbors=3)

    assert result.n_samples == 200
    assert result.idw.predicted.size == 200
    assert result.kriging.predicted.size == 200
    assert result.idw.actual.size == 200


def test_cross_validate_uses_all_points_below_max_points() -> None:
    rng = np.random.default_rng(2)
    n = 50
    x = np.ascontiguousarray(rng.uniform(0.0, 100.0, n))
    y = np.ascontiguousarray(rng.uniform(0.0, 100.0, n))
    z = np.ascontiguousarray(rng.normal(0.0, 1.0, n))

    result = cross_validate(x, y, z, max_points=500, max_neighbors=20, min_neighbors=3)

    assert result.n_samples == n
    assert np.array_equal(result.idw.actual, z)


def test_cross_validate_is_fast_for_500_points() -> None:
    rng = np.random.default_rng(3)
    n = 500
    x = np.ascontiguousarray(rng.uniform(0.0, 5000.0, n))
    y = np.ascontiguousarray(rng.uniform(0.0, 5000.0, n))
    z = np.ascontiguousarray(rng.normal(50.0, 10.0, n))

    # Warm-up call: excludes one-time extension load / grid-build workspace
    # allocation from the timed budget, matching how the native kernel is
    # actually used in a long-running process.
    cross_validate(x, y, z, max_points=500, max_neighbors=20, min_neighbors=3)

    start = time.perf_counter()
    cross_validate(x, y, z, max_points=500, max_neighbors=20, min_neighbors=3)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1, f"cross_validate took {elapsed:.4f}s for 500 points"


def test_cross_validate_rejects_non_positive_radius() -> None:
    x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0])
    y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0])
    z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0])

    try:
        cross_validate(
            x, y, z,
            radius=0.0,
            min_neighbors=1,
            model_values=_model_values(1.0, 5.0, 30.0),
        )
    except ValueError as exc:
        assert "radius" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-positive radius")


def _make_result(
    idw_pred: np.ndarray, krig_pred: np.ndarray, actual: np.ndarray, *, radius: float = 10.0,
) -> CrossValidationResult:
    actual = np.ascontiguousarray(actual, dtype=np.float64)
    idw_pred = np.ascontiguousarray(idw_pred, dtype=np.float64)
    krig_pred = np.ascontiguousarray(krig_pred, dtype=np.float64)

    def _metrics(pred: np.ndarray) -> tuple[float, float, int]:
        valid = np.isfinite(pred)
        if not valid.any():
            return float("nan"), float("nan"), 0
        err = pred[valid] - actual[valid]
        return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2))), int(valid.sum())

    idw_mae, idw_rmse, idw_n = _metrics(idw_pred)
    krig_mae, krig_rmse, krig_n = _metrics(krig_pred)
    return CrossValidationResult(
        idw=MethodCrossValidation(
            mae=idw_mae, rmse=idw_rmse, r2=0.0, n_valid=idw_n, predicted=idw_pred, actual=actual,
        ),
        kriging=MethodCrossValidation(
            mae=krig_mae, rmse=krig_rmse, r2=0.0, n_valid=krig_n, predicted=krig_pred, actual=actual,
        ),
        n_samples=actual.size,
        radius=radius,
        model_params={},
    )


def test_select_best_model_picks_lower_rmse_method_and_flags_significance() -> None:
    actual = np.ascontiguousarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    idw_pred = actual + np.ascontiguousarray([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0])
    krig_pred = actual + np.ascontiguousarray([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    result = _make_result(idw_pred, krig_pred, actual)

    best: BestModelResult = select_best_model(result, seed=0)

    assert best.method == "kriging"
    assert best.kriging_score < best.idw_score
    assert best.significant is True
    assert best.ci_low > 0.0  # idw - kriging consistently positive: kriging is better


def test_select_best_model_exact_tie_prefers_idw_and_is_not_significant() -> None:
    actual = np.ascontiguousarray([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = actual + np.ascontiguousarray([0.5, -0.3, 0.2, -0.1, 0.4])
    result = _make_result(pred, pred.copy(), actual)

    best = select_best_model(result, seed=1)

    assert best.method == "idw"
    assert best.margin == 0.0
    assert best.significant is False
    assert best.ci_low == 0.0
    assert best.ci_high == 0.0


def test_select_best_model_trivial_winner_when_one_method_is_all_nan() -> None:
    actual = np.ascontiguousarray([1.0, 2.0, 3.0, 4.0])
    idw_pred = actual + np.ascontiguousarray([0.5, -0.5, 0.2, -0.2])
    krig_pred = np.full(4, np.nan)
    result = _make_result(idw_pred, krig_pred, actual)

    best = select_best_model(result)

    assert best.method == "idw"
    assert best.n_common_valid == 0
    assert best.significant is True
    assert np.isnan(best.kriging_score)
    assert np.isnan(best.ci_low) and np.isnan(best.ci_high)


def test_select_best_model_raises_when_both_methods_are_all_nan() -> None:
    actual = np.ascontiguousarray([1.0, 2.0, 3.0])
    nan_pred = np.full(3, np.nan)
    result = _make_result(nan_pred, nan_pred.copy(), actual)
    try:
        select_best_model(result)
    except ValueError as exc:
        assert "cannot select a best model" in str(exc)
    else:
        raise AssertionError("expected ValueError when neither method has a finite prediction")


def test_select_best_model_rejects_invalid_parameters() -> None:
    actual = np.ascontiguousarray([1.0, 2.0, 3.0, 4.0])
    pred = actual + 0.1
    result = _make_result(pred, pred.copy(), actual)
    bad_kwargs = [
        {"metric": "r2"},
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
        {"n_bootstrap": 0},
    ]
    for kwargs in bad_kwargs:
        try:
            select_best_model(result, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_select_best_model_is_reproducible_with_seed() -> None:
    rng = np.random.default_rng(7)
    actual = np.ascontiguousarray(rng.normal(size=40))
    idw_pred = actual + rng.normal(0.0, 0.3, size=40)
    krig_pred = actual + rng.normal(0.0, 0.5, size=40)
    result = _make_result(idw_pred, krig_pred, actual)

    first = select_best_model(result, seed=42)
    second = select_best_model(result, seed=42)

    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_select_best_model_rejects_mismatched_actual_arrays() -> None:
    actual_a = np.ascontiguousarray([1.0, 2.0, 3.0])
    actual_b = np.ascontiguousarray([1.0, 2.0, 99.0])
    result = CrossValidationResult(
        idw=MethodCrossValidation(
            mae=0.0, rmse=0.0, r2=0.0, n_valid=3, predicted=actual_a + 0.1, actual=actual_a,
        ),
        kriging=MethodCrossValidation(
            mae=0.0, rmse=0.0, r2=0.0, n_valid=3, predicted=actual_b + 0.1, actual=actual_b,
        ),
        n_samples=3,
        radius=10.0,
        model_params={},
    )
    try:
        select_best_model(result)
    except ValueError as exc:
        assert "actual" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched actual arrays")


def _smooth_field(
    x: np.ndarray, y: np.ndarray, seed: int, n_centers: int = 8, length_scale: float = 25.0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.0, 100.0, (n_centers, 2))
    amps = rng.normal(0.0, 1.0, n_centers)
    field = np.zeros_like(x)
    for amp, (cx, cy) in zip(amps, centers, strict=True):
        field = field + amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * length_scale**2))
    return field


def _cokriging_loo_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Same admissible (PSD) seed/noise profile as test_cokriging.py's ICM
    # K1 fixture -- see the note there on why this seed was chosen.
    rng = np.random.default_rng(6)
    n = 60
    x0 = rng.uniform(0.0, 100.0, n)
    y0 = rng.uniform(0.0, 100.0, n)
    latent = _smooth_field(x0, y0, seed=1006)
    z0 = latent + rng.normal(0.0, 0.15, n)
    zu = 1.4 * latent + rng.normal(0.0, 0.25, n)
    return x0, y0, z0, zu


def test_cross_validate_cokriging_matches_manual_loo() -> None:
    """Batch LOO from the shared native kernel must match holding out each
    point by hand (remove it from every variable's source arrays, then
    interpolate with the regular fixed-model kernels at its coordinates).
    The model itself is fit once from the FULL dataset and held fixed --
    only the interpolation step is leave-one-out, so this reference removes
    the point from the source arrays but reuses the CoKriging instances'
    already-fitted matrices/vectors, not a refit.
    """
    from apbase.cokriging import CoKriging
    from apbase.cokriging._native import run_co_kriging_icm, run_co_kriging_lmc, run_collocated_cokriging

    x0, y0, z0, zu = _cokriging_loo_fixture()
    secondaries = {"secondary": (x0, y0, zu)}
    radius, max_neighbors, min_neighbors = 40.0, 12, 1

    collocated = CoKriging(
        method="collocated", radius=radius, max_neighbors=max_neighbors, min_neighbors=min_neighbors
    ).fit(x0, y0, z0, secondaries)
    icm = CoKriging(
        method="icm", radius=radius, max_neighbors=max_neighbors, min_neighbors=min_neighbors
    ).fit(x0, y0, z0, secondaries)
    lmc = CoKriging(
        method="lmc", radius=radius, max_neighbors=max_neighbors, min_neighbors=min_neighbors
    ).fit(x0, y0, z0, secondaries)

    result = cross_validate_cokriging(
        x0, y0, z0, secondaries, radius=radius, max_neighbors=max_neighbors, min_neighbors=min_neighbors,
    )
    assert result.n_samples == x0.size

    # LOO here is on the PRIMARY only -- secondaries and the coregionalization
    # model are fixed (see the checklist's Milestone 5 note), so the
    # reference calls below hold out `pos` from x0/y0/z0 alone and reuse the
    # full, untouched secondary arrays for every method, matching the CV
    # kernel exactly.
    full_su = collocated._secondary_x, collocated._secondary_y, collocated._secondary_z
    full_offset = np.array([0, full_su[0].size], dtype=np.int32)
    check_positions = [0, 5, 17, 30, 45, x0.size - 1]
    for pos in check_positions:
        mask = np.arange(x0.size) != pos
        held_x0, held_y0, held_z0 = x0[mask], y0[mask], z0[mask]
        tx, ty = np.array([x0[pos]]), np.array([y0[pos]])

        collocated_pred, status = run_collocated_cokriging(
            held_x0, held_y0, held_z0, tx, ty, collocated.model_values, *full_su, full_offset,
            radius, max_neighbors, min_neighbors, 1,
        )
        assert status == 0
        icm_pred, status = run_co_kriging_icm(
            held_x0, held_y0, held_z0, tx, ty,
            icm._fit.model_id, icm._fit.model_range, icm._fit.nugget_matrix, icm._fit.sill_matrix,
            *full_su, full_offset, radius, max_neighbors, min_neighbors, 1,
        )
        assert status == 0
        lmc_pred, status = run_co_kriging_lmc(
            held_x0, held_y0, held_z0, tx, ty,
            lmc._fit.model_id, lmc._fit.structure_ranges, lmc._fit.coefficient_matrices,
            *full_su, full_offset, radius, max_neighbors, min_neighbors, 1,
        )
        assert status == 0

        assert np.allclose(
            result.by_method["collocated"].predicted[pos], collocated_pred[0], rtol=1e-6, atol=1e-6
        )
        assert np.allclose(result.by_method["icm"].predicted[pos], icm_pred[0], rtol=1e-6, atol=1e-6)
        assert np.allclose(result.by_method["lmc"].predicted[pos], lmc_pred[0], rtol=1e-6, atol=1e-6)


def test_cross_validate_cokriging_metrics_match_predictions() -> None:
    x0, y0, z0, zu = _cokriging_loo_fixture()
    result = cross_validate_cokriging(
        x0, y0, z0, {"secondary": (x0, y0, zu)}, max_neighbors=12, min_neighbors=1
    )

    for name, mcv in result.by_method.items():
        valid = np.isfinite(mcv.predicted)
        assert mcv.n_valid == int(np.count_nonzero(valid))
        err = mcv.predicted[valid] - mcv.actual[valid]
        assert np.isclose(mcv.mae, float(np.mean(np.abs(err))), rtol=1e-6), name
        assert np.isclose(mcv.rmse, float(np.sqrt(np.mean(err**2))), rtol=1e-6), name
        assert np.array_equal(mcv.actual, z0)


def test_select_best_model_generalized_picks_correct_winner() -> None:
    actual = np.ascontiguousarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    methods = {
        "collocated": MethodCrossValidation(
            mae=0.0, rmse=0.0, r2=0.0, n_valid=8,
            predicted=actual + np.ascontiguousarray([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0]),
            actual=actual,
        ),
        "icm": MethodCrossValidation(
            mae=0.0, rmse=0.0, r2=0.0, n_valid=8,
            predicted=actual + np.ascontiguousarray([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]),
            actual=actual,
        ),
        "lmc": MethodCrossValidation(
            mae=0.0, rmse=0.0, r2=0.0, n_valid=8,
            predicted=actual + np.ascontiguousarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
            actual=actual,
        ),
    }

    best = select_best_model(methods, seed=0)

    assert isinstance(best, MultiMethodBestModelResult)
    assert best.method == "icm"
    assert best.runner_up == "lmc"
    assert best.scores["icm"] < best.scores["lmc"] < best.scores["collocated"]
    assert best.significant is True
    assert best.n_common_valid == 8


def test_select_best_model_generalized_requires_at_least_two_methods() -> None:
    actual = np.ascontiguousarray([1.0, 2.0, 3.0])
    methods = {
        "only": MethodCrossValidation(mae=0.0, rmse=0.0, r2=0.0, n_valid=3, predicted=actual, actual=actual),
    }
    try:
        select_best_model(methods)
    except ValueError as exc:
        assert "at least two methods" in str(exc)
    else:
        raise AssertionError("expected ValueError for a single-method mapping")


def test_select_best_model_legacy_two_method_call_unchanged() -> None:
    """The classic CrossValidationResult call keeps returning BestModelResult
    with its original field names (idw_score/kriging_score), not the
    generalized MultiMethodBestModelResult -- see the isinstance dispatch
    in select_best_model.
    """
    actual = np.ascontiguousarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    idw_pred = actual + np.ascontiguousarray([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0])
    krig_pred = actual + np.ascontiguousarray([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    result = _make_result(idw_pred, krig_pred, actual)

    best = select_best_model(result, seed=0)

    assert isinstance(best, BestModelResult)
    assert not isinstance(best, MultiMethodBestModelResult)
    assert best.method == "kriging"
    assert best.kriging_score < best.idw_score


if __name__ == "__main__":
    test_cross_validate_matches_reference_loocv()
    test_cross_validate_respects_min_neighbors_nan()
    test_cross_validate_subsamples_to_max_points()
    test_cross_validate_uses_all_points_below_max_points()
    test_cross_validate_is_fast_for_500_points()
    test_cross_validate_rejects_non_positive_radius()
    test_select_best_model_picks_lower_rmse_method_and_flags_significance()
    test_select_best_model_exact_tie_prefers_idw_and_is_not_significant()
    test_select_best_model_trivial_winner_when_one_method_is_all_nan()
    test_select_best_model_raises_when_both_methods_are_all_nan()
    test_select_best_model_rejects_invalid_parameters()
    test_select_best_model_is_reproducible_with_seed()
    test_select_best_model_rejects_mismatched_actual_arrays()
