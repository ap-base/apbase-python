"""High-level leave-one-out cross-validation for IDW and ordinary kriging.

:func:`select_best_model` then picks the statistically better of the two
from a :class:`CrossValidationResult`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from apbase.common.arrays import as_float64_1d, filter_finite_xyz, validate_same_size_xyz
from apbase.common.exceptions import (
    COKRIGING_STATUS_ERRORS,
    CROSS_VALIDATE_STATUS_ERRORS,
    raise_status_error,
)
from apbase.common.validation import require_neighbor_bounds
from apbase.config import resolve_max_neighbors_ceiling, resolve_n_threads, resolve_variogram_profile
from apbase.variogram import Variogram

from ._native import run_cokriging_cross_validate, run_cross_validate

if TYPE_CHECKING:
    from apbase.cokriging import SecondaryInput


@dataclass(frozen=True)
class MethodCrossValidation:
    """Leave-one-out results for a single interpolation method."""

    mae: float
    rmse: float
    r2: float
    n_valid: int
    predicted: np.ndarray
    actual: np.ndarray


@dataclass(frozen=True)
class CrossValidationResult:
    """Leave-one-out cross-validation results for IDW and ordinary kriging."""

    idw: MethodCrossValidation
    kriging: MethodCrossValidation
    n_samples: int
    radius: float
    model_params: dict[str, float | int | str]


@dataclass(frozen=True)
class CokrigingCrossValidationResult:
    """Leave-one-out cross-validation results for collocated/ICM/LMC cokriging.

    See :func:`cross_validate_cokriging`.
    """

    by_method: dict[str, MethodCrossValidation]
    n_samples: int
    radius: float
    model_params: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class BestModelResult:
    """Statistically-supported IDW-vs-kriging pick from a `CrossValidationResult`."""

    method: Literal["idw", "kriging"]
    metric: Literal["rmse", "mae"]
    idw_score: float
    kriging_score: float
    margin: float
    relative_margin: float
    significant: bool
    confidence_level: float
    ci_low: float
    ci_high: float
    n_common_valid: int


@dataclass(frozen=True)
class MultiMethodBestModelResult:
    """Statistically-supported best-of-N pick from a named method mapping.

    Generalizes :class:`BestModelResult` to more than two methods (e.g.
    :attr:`CokrigingCrossValidationResult.by_method`) -- see
    :func:`select_best_model`.
    """

    method: str
    runner_up: str
    metric: Literal["rmse", "mae"]
    scores: dict[str, float]
    margin: float
    relative_margin: float
    significant: bool
    confidence_level: float
    ci_low: float
    ci_high: float
    n_common_valid: int


def cross_validate(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    max_points: int = 500,
    seed: int | None = None,
    radius: float | None = None,
    power: float = 2.0,
    max_neighbors: int = 40,
    min_neighbors: int = 3,
    model_values: ArrayLike | None = None,
) -> CrossValidationResult:
    """Leave-one-out cross-validate local IDW and local ordinary kriging.

    Each held-out point is removed one at a time and re-estimated from the
    remaining source points, reusing one variogram/radius and one spatial
    index for both methods. Evaluation is capped at ``max_points`` held-out
    samples; see ``max_points`` for the sampling behavior on larger
    datasets.

    Parameters
    ----------
    x, y, z:
        One-dimensional source coordinates and values. Rows where any of
        them are ``NaN`` or infinite are ignored.
    max_points:
        Maximum number of source points evaluated by leave-one-out. If the
        (finite) dataset has more points than this, a random subsample of
        this size is drawn instead of evaluating every point.
    seed:
        Seed for the subsample draw when ``x`` has more than ``max_points``
        finite points. ``None`` uses fresh entropy.
    radius:
        Local search radius shared by both methods. If ``None``, derived as
        ``range / 3`` from the fitted (or supplied) variogram, matching the
        default used by :class:`~apbase.idw.IDW` and
        :class:`~apbase.kriging.Kriging`.
    power:
        IDW distance exponent.
    max_neighbors, min_neighbors:
        Neighbor bounds shared by both methods. ``max_neighbors`` is capped at
        ``apbase.config["max_neighbors_ceiling"]`` (default ``500``).
    model_values:
        Optional native variogram model vector (5 values). If omitted, a
        variogram is fit from ``x``, ``y``, ``z``.

    Returns
    -------
    CrossValidationResult
        MAE, RMSE, and R2 for both methods, plus per-sample predictions and
        the held-out actual values.

    Examples
    --------
    >>> result = cross_validate(x, y, z, max_points=300)
    >>> result.idw.rmse, result.kriging.rmse
    """
    x_array = as_float64_1d(x, "x")
    y_array = as_float64_1d(y, "y")
    z_array = as_float64_1d(z, "z")
    validate_same_size_xyz(x_array, y_array, z_array)
    _validate_config(max_points, power, max_neighbors, min_neighbors)

    x_valid, y_valid, z_valid = filter_finite_xyz(x_array, y_array, z_array)
    if x_valid.size < min_neighbors + 1:
        raise ValueError("cross_validate requires at least min_neighbors + 1 finite points")

    if model_values is None:
        n_lags, max_pairs, max_distance = resolve_variogram_profile()
        variogram = Variogram(
            n_lags=n_lags,
            max_pairs=max_pairs,
            max_distance=max_distance,
        ).fit(x_valid, y_valid, z_valid)
    else:
        variogram = Variogram.from_model(model_values)

    radius_value = radius
    if radius_value is None:
        radius_value = float(variogram.model_params["range"]) / 3.0
    if not np.isfinite(radius_value) or radius_value <= 0.0:
        raise ValueError("radius must be finite and greater than zero")

    sample_idx = _select_sample_idx(x_valid.size, max_points, seed)
    n_threads = resolve_n_threads()

    (
        idw_pred,
        krig_pred,
        idw_mae,
        idw_rmse,
        idw_r2,
        idw_n_valid,
        krig_mae,
        krig_rmse,
        krig_r2,
        krig_n_valid,
        status,
    ) = run_cross_validate(
        x_valid,
        y_valid,
        z_valid,
        sample_idx,
        radius_value,
        power,
        variogram.model_values,
        max_neighbors,
        min_neighbors,
        n_threads,
    )
    if status != 0:
        raise_status_error(status, CROSS_VALIDATE_STATUS_ERRORS, "native cross-validation execution failed")

    actual = z_valid[sample_idx - 1]
    return CrossValidationResult(
        idw=MethodCrossValidation(
            mae=idw_mae, rmse=idw_rmse, r2=idw_r2, n_valid=idw_n_valid,
            predicted=idw_pred, actual=actual,
        ),
        kriging=MethodCrossValidation(
            mae=krig_mae, rmse=krig_rmse, r2=krig_r2, n_valid=krig_n_valid,
            predicted=krig_pred, actual=actual,
        ),
        n_samples=int(sample_idx.size),
        radius=float(radius_value),
        model_params=variogram.model_params,
    )


def cross_validate_cokriging(
    x0: ArrayLike,
    y0: ArrayLike,
    z0: ArrayLike,
    secondaries: SecondaryInput,
    *,
    max_points: int = 500,
    seed: int | None = None,
    radius: float | None = None,
    max_neighbors: int = 40,
    min_neighbors: int = 3,
    n_structures: int = 2,
) -> CokrigingCrossValidationResult:
    """Leave-one-out cross-validate collocated, ICM, and LMC cokriging together.

    Each of the three coregionalization models is fit once from the full
    dataset (same as :meth:`~apbase.cokriging.CoKriging.fit`); only the
    primary is then held out one point at a time and re-estimated, sharing
    one primary spatial index and one radius across all three methods --
    see :func:`cross_validate` for the analogous IDW/kriging function.

    Parameters
    ----------
    x0, y0, z0:
        Primary variable source coordinates and values.
    secondaries:
        Same accepted forms as :meth:`~apbase.cokriging.CoKriging.fit`.
    max_points:
        Maximum number of source points evaluated by leave-one-out; see
        :func:`cross_validate`.
    seed:
        Seed for the subsample draw when there are more than ``max_points``
        finite points. ``None`` uses fresh entropy.
    radius:
        Local search radius shared by all three methods and by every
        variable's neighbor search. If ``None``, derived as ``range / 3``
        from a variogram fit on the primary alone (independent of any one
        method's own joint range fit).
    max_neighbors, min_neighbors:
        Neighbor bounds shared by all three methods.
    n_structures:
        Number of nested LMC structures (S), see
        :class:`~apbase.cokriging.CoKriging`.

    Returns
    -------
    CokrigingCrossValidationResult
        MAE, RMSE, and R2 per method under ``by_method``
        (``"collocated"``, ``"icm"``, ``"lmc"``), plus per-sample
        predictions and the held-out actual values (identical across
        methods, evaluated on the same held-out samples).

    Examples
    --------
    >>> result = cross_validate_cokriging(x0, y0, z0, {"ndvi": (xu, yu, zu)})
    >>> result.by_method["icm"].rmse
    """
    from apbase.cokriging import CoKriging

    x0_array = as_float64_1d(x0, "x0")
    y0_array = as_float64_1d(y0, "y0")
    z0_array = as_float64_1d(z0, "z0")
    validate_same_size_xyz(x0_array, y0_array, z0_array)
    if max_points <= 0:
        raise ValueError("max_points must be greater than zero")
    require_neighbor_bounds(
        max_neighbors, min_neighbors, max_neighbors_ceiling=resolve_max_neighbors_ceiling()
    )

    collocated = CoKriging(
        method="collocated", radius=radius, max_neighbors=max_neighbors, min_neighbors=min_neighbors
    ).fit(x0, y0, z0, secondaries)
    icm = CoKriging(
        method="icm", radius=radius, max_neighbors=max_neighbors, min_neighbors=min_neighbors
    ).fit(x0, y0, z0, secondaries)
    lmc = CoKriging(
        method="lmc", radius=radius, max_neighbors=max_neighbors, min_neighbors=min_neighbors,
        n_structures=n_structures,
    ).fit(x0, y0, z0, secondaries)

    x0_valid, y0_valid, z0_valid = collocated._x0, collocated._y0, collocated._z0
    secondary_x, secondary_y, secondary_z = (
        collocated._secondary_x, collocated._secondary_y, collocated._secondary_z
    )
    secondary_offset = collocated._secondary_offset
    assert x0_valid is not None and y0_valid is not None and z0_valid is not None
    assert secondary_x is not None and secondary_y is not None and secondary_z is not None
    assert secondary_offset is not None

    if radius is None:
        n_lags, max_pairs, max_distance = resolve_variogram_profile()
        variogram = Variogram(n_lags=n_lags, max_pairs=max_pairs, max_distance=max_distance).fit(
            x0_valid, y0_valid, z0_valid
        )
        radius_value = float(variogram.model_params["range"]) / 3.0
        if not np.isfinite(radius_value) or radius_value <= 0.0:
            raise ValueError("radius must be finite and greater than zero")
    else:
        radius_value = float(radius)

    sample_idx = _select_sample_idx(x0_valid.size, max_points, seed)
    n_threads = resolve_n_threads()

    from apbase.cokriging._interpolation import _ICMFit, _LMCFit

    assert isinstance(icm._fit, _ICMFit)
    assert isinstance(lmc._fit, _LMCFit)

    (
        collocated_pred, icm_pred, lmc_pred,
        collocated_mae, collocated_rmse, collocated_r2, collocated_n_valid,
        icm_mae, icm_rmse, icm_r2, icm_n_valid,
        lmc_mae, lmc_rmse, lmc_r2, lmc_n_valid,
        status,
    ) = run_cokriging_cross_validate(
        x0_valid, y0_valid, z0_valid, sample_idx,
        secondary_x, secondary_y, secondary_z, secondary_offset,
        collocated.model_values,
        icm._fit.model_id, icm._fit.model_range, icm._fit.nugget_matrix, icm._fit.sill_matrix,
        lmc._fit.model_id, lmc._fit.structure_ranges, lmc._fit.coefficient_matrices,
        radius_value, max_neighbors, min_neighbors, n_threads,
    )
    if status != 0:
        raise_status_error(
            status, COKRIGING_STATUS_ERRORS, "native cokriging cross-validation execution failed"
        )

    actual = z0_valid[sample_idx - 1]
    by_method = {
        "collocated": MethodCrossValidation(
            mae=collocated_mae, rmse=collocated_rmse, r2=collocated_r2, n_valid=collocated_n_valid,
            predicted=collocated_pred, actual=actual,
        ),
        "icm": MethodCrossValidation(
            mae=icm_mae, rmse=icm_rmse, r2=icm_r2, n_valid=icm_n_valid,
            predicted=icm_pred, actual=actual,
        ),
        "lmc": MethodCrossValidation(
            mae=lmc_mae, rmse=lmc_rmse, r2=lmc_r2, n_valid=lmc_n_valid,
            predicted=lmc_pred, actual=actual,
        ),
    }
    return CokrigingCrossValidationResult(
        by_method=by_method,
        n_samples=int(sample_idx.size),
        radius=float(radius_value),
        model_params={
            "collocated": collocated.model_params,
            "icm": icm.model_params,
            "lmc": lmc.model_params,
        },
    )


@overload
def select_best_model(
    result: CrossValidationResult,
    *,
    metric: Literal["rmse", "mae"] = "rmse",
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BestModelResult: ...


@overload
def select_best_model(
    result: Mapping[str, MethodCrossValidation],
    *,
    metric: Literal["rmse", "mae"] = "rmse",
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> MultiMethodBestModelResult: ...


def select_best_model(
    result: CrossValidationResult | Mapping[str, MethodCrossValidation],
    *,
    metric: Literal["rmse", "mae"] = "rmse",
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BestModelResult | MultiMethodBestModelResult:
    """Pick the statistically better method from two or more evaluated methods.

    Given a :class:`CrossValidationResult` (IDW vs. kriging), this keeps its
    original two-method behavior and return type unchanged -- see below.
    Given any other named mapping of :class:`MethodCrossValidation` (e.g.
    :attr:`CokrigingCrossValidationResult.by_method`, or a hand-built
    ``{"a": ..., "b": ..., "c": ...}``), it generalizes to a best-vs-runner-up
    comparison across all of them and returns a
    :class:`MultiMethodBestModelResult` instead.

    Pick the better of IDW/kriging from a :class:`CrossValidationResult`.

    Compares IDW against ordinary kriging on the held-out samples where
    *both* methods produced a finite leave-one-out prediction (some samples
    can be finite for one method and NaN for the other -- e.g. a singular
    local kriging system -- so the two error series stay paired
    point-for-point). The lower-``metric`` method wins (ties go to
    ``"idw"``, matching :func:`~apbase.mapping.create_map`'s selection
    rule). A paired bootstrap over that common set then estimates a
    confidence interval for the score difference (idw minus kriging);
    ``significant`` is ``True`` when that interval excludes zero, i.e. the
    win is unlikely to be an artifact of the particular held-out sample
    draw.

    Parameters
    ----------
    result:
        Output of :func:`cross_validate`.
    metric:
        Error metric used both for the point estimate and for each
        bootstrap replicate. ``"rmse"`` matches the metric already used by
        :func:`~apbase.mapping.create_map` to auto-select a method. R2 is
        intentionally not offered here: it is monotonic in RMSE for a fixed
        actual-value set, but its normalization (``1 - SS_res/SS_tot``)
        depends on the evaluated subset's variance, so it does not compare
        as cleanly once the two methods' NaN patterns diverge.
    n_bootstrap:
        Number of paired bootstrap resamples. Memory and time scale as
        ``O(n_bootstrap * n_common_valid)``; the default is fast (well
        under a second) up to ``cross_validate``'s default
        ``max_points=500``.
    confidence_level:
        Two-sided confidence level for the score-difference interval, e.g.
        ``0.95`` for a 95% CI.
    seed:
        Seed for the bootstrap resample draw. ``None`` uses fresh entropy.

    Returns
    -------
    BestModelResult
        The selected method plus the paired scores, margin, and bootstrap
        confidence interval backing that choice. ``ci_low``/``ci_high``
        bound the idw-minus-kriging score difference: a positive interval
        means kriging scored better, a negative interval means idw scored
        better.

    Notes
    -----
    This uses a percentile bootstrap on the paired per-sample errors
    instead of a parametric test (e.g. a paired t-test), so it needs no
    distributional assumption and no SciPy dependency: ``n_bootstrap``
    index draws are taken once via
    :meth:`numpy.random.Generator.integers`, and every replicate's
    RMSE/MAE is computed in one vectorized reduction, not a Python loop.
    Resampling reuses the *same* draw for both methods on each replicate,
    which preserves the point-to-point pairing/correlation between IDW and
    kriging errors -- essential for a meaningful difference interval.

    Any held-out sample where ``actual`` itself is non-finite is also
    excluded (this cannot happen from ``cross_validate``'s own output,
    which is filtered upfront, but protects hand-built
    ``CrossValidationResult`` inputs).

    If one method has zero finite predictions on the common set while the
    other has at least one, that method wins outright (``significant=True``,
    ``margin=inf``, no bootstrap -- there is nothing to resample). If
    neither method has any finite prediction, or the two methods' finite
    predictions never overlap on the same held-out sample, this raises
    ``ValueError``: unlike ``create_map``'s plain RMSE tie-break, this
    function refuses to guess when there is no shared, paired evidence to
    compare.

    Examples
    --------
    >>> result = cross_validate(x, y, z, max_points=300)
    >>> best = select_best_model(result)
    >>> best.method
    'kriging'
    >>> best.significant
    True
    """
    if not isinstance(result, CrossValidationResult):
        return _select_best_of_methods(
            dict(result),
            metric=metric,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )
    if metric not in ("rmse", "mae"):
        raise ValueError('metric must be "rmse" or "mae"')
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be greater than zero")

    idw, krig = result.idw, result.kriging
    if idw.predicted.shape != idw.actual.shape or krig.predicted.shape != krig.actual.shape:
        raise ValueError("predicted and actual arrays must share one shape per method")
    if idw.actual.shape != krig.actual.shape or not np.array_equal(idw.actual, krig.actual):
        raise ValueError(
            "idw.actual and kriging.actual must be identical: cross_validate() always "
            "evaluates both methods on the same held-out samples"
        )

    actual = idw.actual
    actual_finite = np.isfinite(actual)
    idw_valid = np.isfinite(idw.predicted) & actual_finite
    krig_valid = np.isfinite(krig.predicted) & actual_finite
    mask = idw_valid & krig_valid
    n_common_valid = int(np.count_nonzero(mask))

    if n_common_valid == 0:
        return _select_without_overlap(metric, confidence_level, idw, krig, actual, idw_valid, krig_valid)

    idw_err = idw.predicted[mask] - actual[mask]
    krig_err = krig.predicted[mask] - actual[mask]
    idw_score = float(_method_score(metric, idw_err))
    krig_score = float(_method_score(metric, krig_err))
    method: Literal["idw", "kriging"] = "idw" if idw_score <= krig_score else "kriging"

    loser_score = max(idw_score, krig_score)
    margin = abs(idw_score - krig_score)
    relative_margin = 0.0 if loser_score == 0.0 else margin / loser_score

    if n_common_valid < 2:
        return BestModelResult(
            method=method,
            metric=metric,
            idw_score=idw_score,
            kriging_score=krig_score,
            margin=margin,
            relative_margin=relative_margin,
            significant=False,
            confidence_level=confidence_level,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_common_valid=n_common_valid,
        )

    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n_common_valid, size=(n_bootstrap, n_common_valid))
    idw_boot_score = _method_score(metric, idw_err[resample_idx])
    krig_boot_score = _method_score(metric, krig_err[resample_idx])
    diff = idw_boot_score - krig_boot_score
    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(diff, [alpha / 2.0, 1.0 - alpha / 2.0])
    significant = bool(ci_low > 0.0 or ci_high < 0.0)

    return BestModelResult(
        method=method,
        metric=metric,
        idw_score=idw_score,
        kriging_score=krig_score,
        margin=margin,
        relative_margin=relative_margin,
        significant=significant,
        confidence_level=confidence_level,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_common_valid=n_common_valid,
    )


def _method_score(metric: Literal["rmse", "mae"], error: np.ndarray) -> np.ndarray:
    """RMSE or MAE reduced along the last axis (scalar-like for 1D input)."""
    if metric == "rmse":
        return np.sqrt(np.mean(error * error, axis=-1))
    return np.mean(np.abs(error), axis=-1)


def _select_best_of_methods(
    methods: dict[str, MethodCrossValidation],
    *,
    metric: Literal["rmse", "mae"],
    n_bootstrap: int,
    confidence_level: float,
    seed: int | None,
) -> MultiMethodBestModelResult:
    """Generalized N-method core behind :func:`select_best_model`.

    Fixes the winner and runner-up from the point estimate on the samples
    valid across *every* method, then bootstraps only that pair's paired
    score difference -- a direct generalization of the two-method logic
    below (which always compares exactly idw vs. kriging), not a new
    statistical procedure.
    """
    if metric not in ("rmse", "mae"):
        raise ValueError('metric must be "rmse" or "mae"')
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be greater than zero")
    if len(methods) < 2:
        raise ValueError("select_best_model requires at least two methods to compare")

    names = list(methods)
    actual = methods[names[0]].actual
    for name in names:
        m = methods[name]
        if m.predicted.shape != m.actual.shape:
            raise ValueError("predicted and actual arrays must share one shape per method")
        if m.actual.shape != actual.shape or not np.array_equal(m.actual, actual):
            raise ValueError(
                "all methods must share identical held-out actual values -- "
                "cross_validate_cokriging() always evaluates every method on the same held-out samples"
            )

    actual_finite = np.isfinite(actual)
    valid_masks = {name: (np.isfinite(m.predicted) & actual_finite) for name, m in methods.items()}
    common_mask = actual_finite.copy()
    for mask in valid_masks.values():
        common_mask &= mask
    n_common_valid = int(np.count_nonzero(common_mask))

    if n_common_valid == 0:
        return _select_best_of_methods_without_overlap(metric, confidence_level, methods, valid_masks, actual)

    errors = {name: methods[name].predicted[common_mask] - actual[common_mask] for name in names}
    scores = {name: float(_method_score(metric, err)) for name, err in errors.items()}
    ranked = sorted(names, key=lambda n: scores[n])
    winner, runner_up = ranked[0], ranked[1]
    winner_score, runner_up_score = scores[winner], scores[runner_up]
    margin = runner_up_score - winner_score
    relative_margin = 0.0 if runner_up_score == 0.0 else margin / runner_up_score

    if n_common_valid < 2:
        return MultiMethodBestModelResult(
            method=winner,
            runner_up=runner_up,
            metric=metric,
            scores=scores,
            margin=margin,
            relative_margin=relative_margin,
            significant=False,
            confidence_level=confidence_level,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_common_valid=n_common_valid,
        )

    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n_common_valid, size=(n_bootstrap, n_common_valid))
    winner_boot = _method_score(metric, errors[winner][resample_idx])
    runner_up_boot = _method_score(metric, errors[runner_up][resample_idx])
    diff = runner_up_boot - winner_boot
    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(diff, [alpha / 2.0, 1.0 - alpha / 2.0])
    significant = bool(ci_low > 0.0 or ci_high < 0.0)

    return MultiMethodBestModelResult(
        method=winner,
        runner_up=runner_up,
        metric=metric,
        scores=scores,
        margin=margin,
        relative_margin=relative_margin,
        significant=significant,
        confidence_level=confidence_level,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_common_valid=n_common_valid,
    )


def _select_best_of_methods_without_overlap(
    metric: Literal["rmse", "mae"],
    confidence_level: float,
    methods: dict[str, MethodCrossValidation],
    valid_masks: dict[str, np.ndarray],
    actual: np.ndarray,
) -> MultiMethodBestModelResult:
    names_with_valid = [name for name, mask in valid_masks.items() if np.count_nonzero(mask) > 0]
    if not names_with_valid:
        raise ValueError(
            "no method produced a finite leave-one-out prediction; cannot select a best model"
        )
    if len(names_with_valid) > 1:
        raise ValueError(
            "no held-out sample has a finite prediction from every method; cannot form a paired comparison"
        )

    winner = names_with_valid[0]
    mask = valid_masks[winner]
    winner_score = float(_method_score(metric, methods[winner].predicted[mask] - actual[mask]))
    scores = {name: (winner_score if name == winner else float("nan")) for name in methods}
    runner_up = next(name for name in methods if name != winner)

    return MultiMethodBestModelResult(
        method=winner,
        runner_up=runner_up,
        metric=metric,
        scores=scores,
        margin=float("inf"),
        relative_margin=float("inf"),
        significant=True,
        confidence_level=confidence_level,
        ci_low=float("nan"),
        ci_high=float("nan"),
        n_common_valid=0,
    )


def _select_without_overlap(
    metric: Literal["rmse", "mae"],
    confidence_level: float,
    idw: MethodCrossValidation,
    krig: MethodCrossValidation,
    actual: np.ndarray,
    idw_valid: np.ndarray,
    krig_valid: np.ndarray,
) -> BestModelResult:
    n_idw_valid = int(np.count_nonzero(idw_valid))
    n_krig_valid = int(np.count_nonzero(krig_valid))
    if n_idw_valid == 0 and n_krig_valid == 0:
        raise ValueError(
            "neither idw nor kriging produced a finite leave-one-out prediction; "
            "cannot select a best model"
        )
    if n_idw_valid > 0 and n_krig_valid > 0:
        raise ValueError(
            "idw and kriging have no held-out sample with both a finite idw and a "
            "finite kriging prediction; cannot form a paired comparison"
        )

    method: Literal["idw", "kriging"]
    if n_idw_valid == 0:
        winner_score = float(_method_score(metric, krig.predicted[krig_valid] - actual[krig_valid]))
        idw_score, krig_score, method = float("nan"), winner_score, "kriging"
    else:
        winner_score = float(_method_score(metric, idw.predicted[idw_valid] - actual[idw_valid]))
        idw_score, krig_score, method = winner_score, float("nan"), "idw"

    return BestModelResult(
        method=method,
        metric=metric,
        idw_score=idw_score,
        kriging_score=krig_score,
        margin=float("inf"),
        relative_margin=float("inf"),
        significant=True,
        confidence_level=confidence_level,
        ci_low=float("nan"),
        ci_high=float("nan"),
        n_common_valid=0,
    )


def _select_sample_idx(n: int, max_points: int, seed: int | None) -> np.ndarray:
    if n <= max_points:
        return np.arange(1, n + 1, dtype=np.int32)
    rng = np.random.default_rng(seed)
    positions = rng.choice(n, size=max_points, replace=False)
    return np.ascontiguousarray(positions + 1, dtype=np.int32)


def _validate_config(max_points: int, power: float, max_neighbors: int, min_neighbors: int) -> None:
    if max_points <= 0:
        raise ValueError("max_points must be greater than zero")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("power must be finite and greater than zero")
    require_neighbor_bounds(
        max_neighbors,
        min_neighbors,
        max_neighbors_ceiling=resolve_max_neighbors_ceiling(),
    )


__all__ = [
    "BestModelResult",
    "CokrigingCrossValidationResult",
    "CrossValidationResult",
    "MethodCrossValidation",
    "MultiMethodBestModelResult",
    "cross_validate",
    "cross_validate_cokriging",
    "select_best_model",
]
