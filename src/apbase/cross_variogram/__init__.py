"""Cross-variogram fitting and secondary-variable screening for cokriging.

Runs *before* :mod:`apbase.cokriging`: fits a generic cross-variogram model
between any two variables, and screens candidate secondary variables against
a primary via a Fisher z-test (significance) and a Cauchy-Schwarz check
(admissibility).
"""

from __future__ import annotations

from ._cross_variogram import CrossVariogram
from ._screening import CovariateReport, ScreeningResult, screen_secondary_variables

__all__ = [
    "CovariateReport",
    "CrossVariogram",
    "ScreeningResult",
    "screen_secondary_variables",
]
