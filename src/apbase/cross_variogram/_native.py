"""Thin Python wrappers around the cross-variogram native extension."""

from __future__ import annotations

from types import ModuleType

import numpy as np

from apbase._native import lib as _shared_lib
from apbase.common.exceptions import NativeExecutionError, NativeExtensionError

REQUIRED_SYMBOLS = (
    "fit_cross_variogram",
    "screen_covariate",
)


def _as_float64_contiguous(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.float64)


def load_cross_variogram_module() -> ModuleType:
    """Return the loaded cross-variogram native extension."""
    module = _shared_lib.load()
    missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(module, symbol)]
    if missing:
        raise NativeExtensionError(f"cross_variogram extension is unavailable: missing {', '.join(missing)}")
    return module


def run_fit_cross_variogram(
    xa: np.ndarray,
    ya: np.ndarray,
    za: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    zb: np.ndarray,
    max_colocation_dist: float,
    n_lags: int,
    max_pairs: int,
    max_distance: float,
    n_threads: int,
) -> tuple[np.ndarray, float, int]:
    """Fit a generic cross-variogram model between two (possibly heterotopic) variables."""
    module = load_cross_variogram_module()
    try:
        model_info, rho, n_pairs = module.fit_cross_variogram(
            _as_float64_contiguous(xa),
            _as_float64_contiguous(ya),
            _as_float64_contiguous(za),
            _as_float64_contiguous(xb),
            _as_float64_contiguous(yb),
            _as_float64_contiguous(zb),
            float(max_colocation_dist),
            int(n_lags),
            int(max_pairs),
            float(max_distance),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native cross_variogram extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"error fitting cross-variogram: {exc}") from exc
    return np.asarray(model_info, dtype=np.float64), float(rho), int(n_pairs)


def run_screen_covariate(
    model_00: np.ndarray,
    model_uu: np.ndarray,
    model_0u: np.ndarray,
    rho: float,
    n_pairs: int,
    alpha: float,
    min_abs_rho: float,
    max_violation_fraction: float,
) -> tuple[bool, float, float, int]:
    """Run the Fisher z-test + Cauchy-Schwarz admissibility decision for one candidate.

    Unlike most ``run_*`` wrappers, a non-zero ``status`` here is not raised as an
    exception: ``status != 0`` (e.g. an insufficient paired sample) is an expected,
    reportable screening outcome, not a native execution failure -- see
    ``_screening.py``.
    """
    module = load_cross_variogram_module()
    try:
        decision, p_value, admissible_fraction, status = module.screen_covariate(
            _as_float64_contiguous(model_00),
            _as_float64_contiguous(model_uu),
            _as_float64_contiguous(model_0u),
            float(rho),
            int(n_pairs),
            float(alpha),
            float(min_abs_rho),
            float(max_violation_fraction),
        )
    except AttributeError:
        raise NativeExtensionError("native cross_variogram extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"error screening covariate: {exc}") from exc
    return bool(int(decision) != 0), float(p_value), float(admissible_fraction), int(status)


__all__ = [
    "load_cross_variogram_module",
    "run_fit_cross_variogram",
    "run_screen_covariate",
]
