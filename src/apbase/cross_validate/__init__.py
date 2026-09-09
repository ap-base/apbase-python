"""Leave-one-out cross-validation for IDW, ordinary kriging, and cokriging.

This package exports the public functions used to evaluate IDW/kriging and
collocated/ICM/LMC cokriging accuracy (MAE, RMSE, R2) on the same source
dataset, plus ``select_best_model`` to automatically pick the statistically
better method from either result.
"""

from __future__ import annotations

from ._cross_validation import (
    BestModelResult,
    CokrigingCrossValidationResult,
    CrossValidationResult,
    MethodCrossValidation,
    MultiMethodBestModelResult,
    cross_validate,
    cross_validate_cokriging,
    select_best_model,
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
