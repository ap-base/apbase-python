"""Statistical screening of candidate secondary variables for cokriging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from apbase.common.arrays import as_float64_1d, filter_finite_xyz, validate_same_size_xyz
from apbase.variogram import Variogram

from ._cross_variogram import CrossVariogram
from ._native import run_screen_covariate

_STATUS_MESSAGES: dict[int, str] = {
    0: "ok",
    1: "amostra insuficiente",
    2: "modelo auto-variograma invalido",
    3: "modelo cross-variograma invalido",
}


@dataclass(frozen=True)
class CovariateReport:
    """Screening outcome for one candidate secondary variable."""

    rho: float
    p_value: float
    admissible_fraction: float
    n_pairs: int
    decision: bool
    status: str
    model_params: dict[str, float | int | str] | None


@dataclass(frozen=True)
class ScreeningResult:
    """Result of :func:`screen_secondary_variables`."""

    selected: dict[str, tuple[ArrayLike, ArrayLike, ArrayLike]]
    report: dict[str, CovariateReport]


def screen_secondary_variables(
    x0: ArrayLike,
    y0: ArrayLike,
    z0: ArrayLike,
    candidates: dict[str, tuple[ArrayLike, ArrayLike, ArrayLike]],
    *,
    alpha: float = 0.05,
    correction: Literal["bonferroni", "none"] = "bonferroni",
    min_abs_rho: float = 0.3,
    max_violation_fraction: float = 0.1,
    max_colocation_dist: float | None = None,
    n_lags: int = 50,
    max_pairs: int = 100_000,
    max_distance: float = 0.0,
) -> ScreeningResult:
    """Test each candidate secondary variable for admissible use in cokriging.

    Runs, per candidate: a Fisher z-test on the primary<->candidate correlation
    (``Fisher1921``) and a Cauchy-Schwarz admissibility check on the fitted
    cross-variogram (``Journel1978``, ``Myers1982``). A candidate is kept only
    if both the correlation is significant (and practically relevant, via
    ``min_abs_rho``) and the cross-variogram is admissible.

    Parameters
    ----------
    x0, y0, z0:
        Primary variable source coordinates and values.
    candidates:
        Mapping of candidate name to ``(x, y, z)`` source arrays.
    alpha:
        Significance level for the Fisher z-test, per candidate before
        ``correction``.
    correction:
        ``"bonferroni"`` (default when ``len(candidates) > 1``) divides
        ``alpha`` by the candidate count to control the family-wise
        false-positive rate; ``"none"`` tests each candidate at ``alpha``
        directly.
    min_abs_rho:
        Practical-significance threshold: a candidate is rejected even with a
        significant p-value if ``|rho| < min_abs_rho``.
    max_violation_fraction:
        Maximum fraction of evaluated lags allowed to violate the
        Cauchy-Schwarz bound before a candidate is rejected as inadmissible.
    max_colocation_dist:
        Passed through to :class:`CrossVariogram` for the primary<->candidate
        pairing. ``None`` requires an exact coincidence.
    n_lags, max_pairs, max_distance:
        Passed through to both the primary/candidate :class:`~apbase.variogram.
        Variogram` fits and the :class:`CrossVariogram` fit.

    Returns
    -------
    ScreeningResult
        ``selected``: subset of ``candidates`` that passed both tests, ready
        for ``CoKriging(...).fit(x0, y0, z0, secondaries=result.selected)``.
        ``report``: per-candidate diagnostics (rho, p-value, admissible
        fraction, decision, status) for every candidate, kept or not.

    Notes
    -----
    A candidate rejected for "amostra insuficiente" (fewer than 4 colocated
    pairs) or an invalid fitted model is recorded in ``report`` with
    ``decision=False`` and the reason in ``status`` -- it does not raise.

    Examples
    --------
    >>> result = screen_secondary_variables(x0, y0, z0, {"ndvi": (xn, yn, zn)})
    >>> result.selected  # doctest: +SKIP
    {'ndvi': (xn, yn, zn)}
    """
    if not candidates:
        raise ValueError("candidates must contain at least one entry")
    if correction not in ("bonferroni", "none"):
        raise ValueError('correction must be "bonferroni" or "none"')
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")
    if not np.isfinite(min_abs_rho) or not (0.0 <= min_abs_rho <= 1.0):
        raise ValueError("min_abs_rho must be between 0 and 1")
    if not np.isfinite(max_violation_fraction) or not (0.0 <= max_violation_fraction <= 1.0):
        raise ValueError("max_violation_fraction must be between 0 and 1")

    x0_array = as_float64_1d(x0, "x0")
    y0_array = as_float64_1d(y0, "y0")
    z0_array = as_float64_1d(z0, "z0")
    validate_same_size_xyz(x0_array, y0_array, z0_array)
    x0_valid, y0_valid, z0_valid = filter_finite_xyz(x0_array, y0_array, z0_array)
    if x0_valid.size == 0:
        raise ValueError("x0, y0, z0 require at least one finite point")

    primary_variogram = Variogram(n_lags=n_lags, max_pairs=max_pairs, max_distance=max_distance).fit(
        x0_valid, y0_valid, z0_valid
    )

    use_bonferroni = correction == "bonferroni" and len(candidates) > 1
    effective_alpha = alpha / len(candidates) if use_bonferroni else alpha

    selected: dict[str, tuple[ArrayLike, ArrayLike, ArrayLike]] = {}
    report: dict[str, CovariateReport] = {}

    for name, (xc, yc, zc) in candidates.items():
        xc_array = as_float64_1d(xc, f"candidates[{name!r}][0]")
        yc_array = as_float64_1d(yc, f"candidates[{name!r}][1]")
        zc_array = as_float64_1d(zc, f"candidates[{name!r}][2]")
        validate_same_size_xyz(xc_array, yc_array, zc_array)
        xc_valid, yc_valid, zc_valid = filter_finite_xyz(xc_array, yc_array, zc_array)

        if xc_valid.size == 0:
            report[name] = CovariateReport(
                rho=float("nan"),
                p_value=float("nan"),
                admissible_fraction=float("nan"),
                n_pairs=0,
                decision=False,
                status=_STATUS_MESSAGES[1],
                model_params=None,
            )
            continue

        candidate_variogram = Variogram(n_lags=n_lags, max_pairs=max_pairs, max_distance=max_distance).fit(
            xc_valid, yc_valid, zc_valid
        )
        cross = CrossVariogram(
            n_lags=n_lags,
            max_pairs=max_pairs,
            max_distance=max_distance,
            max_colocation_dist=max_colocation_dist,
        ).fit(x0_valid, y0_valid, z0_valid, xc_valid, yc_valid, zc_valid)

        decision, p_value, admissible_fraction, status = run_screen_covariate(
            primary_variogram.model_values,
            candidate_variogram.model_values,
            cross.model_values,
            cross.rho,
            cross.n_pairs,
            effective_alpha,
            min_abs_rho,
            max_violation_fraction,
        )

        kept = status == 0 and decision
        report[name] = CovariateReport(
            rho=cross.rho,
            p_value=p_value,
            admissible_fraction=admissible_fraction,
            n_pairs=cross.n_pairs,
            decision=kept,
            status=_STATUS_MESSAGES.get(status, f"status desconhecido ({status})"),
            model_params=cross.model_params if status == 0 else None,
        )
        if kept:
            selected[name] = (xc, yc, zc)

    return ScreeningResult(selected=selected, report=report)


__all__ = ["CovariateReport", "ScreeningResult", "screen_secondary_variables"]
