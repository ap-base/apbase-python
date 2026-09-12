from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from apbase.common.arrays import as_float64_1d, filter_finite_xyz, validate_same_size_xyz
from apbase.config import resolve_n_threads, resolve_variogram_max_pairs_ceiling
from apbase.variogram import Variogram


class CrossVariogram:
    """Fit an empirical cross-variogram model between two variables.

    Symmetric in ``a``/``b`` -- no fixed primary/secondary role. Used by
    :func:`screen_secondary_variables` and by :class:`~apbase.cokriging.
    CoKriging`'s ICM/LMC fit for every variable pair.

    ``a`` and ``b`` may be sampled at different coordinates: the native fit
    pairs them via nearest-neighbor join, accepting a match within
    ``max_colocation_dist``.

    Parameters
    ----------
    n_lags, max_pairs, max_distance:
        Same meaning as :class:`~apbase.variogram.Variogram`, applied to the
        paired sample.
    max_colocation_dist:
        Maximum distance accepted by the nearest-neighbor join. ``None``
        requires an exact coincidence.

    Examples
    --------
    >>> import numpy as np
    >>> from apbase.cross_variogram import CrossVariogram
    >>> xa = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0, 5.0])
    >>> ya = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0, 5.0])
    >>> za = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0, 6.5])
    >>> zb = np.ascontiguousarray([1.2, 1.6, 1.7, 2.1, 1.4])
    >>> cross = CrossVariogram(n_lags=4).fit(xa, ya, za, xa, ya, zb)
    >>> cross.rho > 0.0
    True
    """

    __slots__ = (
        "_n_lags",
        "_max_pairs",
        "_max_distance",
        "_max_colocation_dist",
        "_rho",
        "_n_pairs",
        "_model_values",
        "_fitted",
    )

    def __init__(
        self,
        *,
        n_lags: int = 50,
        max_pairs: int = 100_000,
        max_distance: float = 0.0,
        max_colocation_dist: float | None = None,
    ) -> None:
        self._n_lags = int(n_lags)
        self._max_pairs = int(max_pairs)
        self._max_distance = float(max_distance)
        self._max_colocation_dist = None if max_colocation_dist is None else float(max_colocation_dist)
        self._validate_config()

        self._rho: float | None = None
        self._n_pairs: int | None = None
        self._model_values: np.ndarray | None = None
        self._fitted = False

    @property
    def n_lags(self) -> int:
        """Number of lag bins used by the native fitter.

        Returns
        -------
        int
            Lag bin count.
        """
        return self._n_lags

    @property
    def max_pairs(self) -> int:
        """Maximum sampled colocated point pairs.

        Returns
        -------
        int
            Pair sampling cap; ``0`` means all eligible pairs subject to the
            native ceiling.
        """
        return self._max_pairs

    @property
    def max_distance(self) -> float:
        """Maximum pair distance considered by the fitter.

        Returns
        -------
        float
            Distance cutoff in coordinate units; ``0.0`` uses the native
            default extent.
        """
        return self._max_distance

    @property
    def max_colocation_dist(self) -> float | None:
        """Maximum nearest-neighbor distance accepted for colocating samples.

        Returns
        -------
        float or None
            Distance tolerance, or ``None`` for exact coordinate coincidence.
        """
        return self._max_colocation_dist

    @property
    def n_threads(self) -> int:
        """OpenMP thread count used by native cross-variogram routines.

        Returns
        -------
        int
            Current value resolved from ``apbase.config``.
        """
        return resolve_n_threads()

    @classmethod
    def from_data(
        cls,
        xa: ArrayLike,
        ya: ArrayLike,
        za: ArrayLike,
        xb: ArrayLike,
        yb: ArrayLike,
        zb: ArrayLike,
        **kwargs: Any,
    ) -> CrossVariogram:
        """Create and fit a cross-variogram from source data.

        Parameters
        ----------
        xa, ya, za : array_like
            First variable coordinates and values.
        xb, yb, zb : array_like
            Second variable coordinates and values.
        **kwargs
            Keyword arguments forwarded to :class:`CrossVariogram`.

        Returns
        -------
        CrossVariogram
            Fitted cross-variogram.
        """
        return cls(**kwargs).fit(xa, ya, za, xb, yb, zb)

    @property
    def is_fitted(self) -> bool:
        """Whether a cross-variogram fit is available.

        Returns
        -------
        bool
            ``True`` after :meth:`fit` succeeds.
        """
        return self._fitted

    @property
    def model_values(self) -> np.ndarray:
        """Native model vector: ``[model_id, nugget, partial_sill, range, sse]``.

        ``nugget``/``partial_sill`` may be negative for a negatively-correlated
        pair -- unlike an auto-variogram, a cross-variogram is not required to
        be a valid direct covariance on its own. Available whenever the fit
        found at least 3 populated lag bins, even if the winning model later
        turns out invalid for :attr:`model_params`/:meth:`evaluate` (e.g.
        ``range`` collapsing to 0) -- see :attr:`n_pairs` to check whether the
        colocated sample was even large enough to attempt a fit.

        Returns
        -------
        numpy.ndarray
            Native cross-variogram model vector.
        """
        self._require_fitted()
        assert self._model_values is not None
        return self._model_values

    @property
    def model_params(self) -> dict[str, float | int | str]:
        """Readable parameters for the selected cross-variogram model.

        Raises the same way :class:`~apbase.variogram.Variogram` does when
        the fit degenerated (e.g. too few colocated pairs to populate 3 lag
        bins). Callers that need to handle a degenerate fit without raising
        (like :func:`~apbase.cross_variogram.screen_secondary_variables`)
        should check :attr:`n_pairs` before reading this.

        Returns
        -------
        dict
            Model id/name, nugget, partial sill, sill, range, and SSE.
        """
        return self._as_variogram().model_params

    @property
    def rho(self) -> float:
        """Pearson correlation over the paired colocated sample.

        Returns
        -------
        float
            Correlation coefficient.
        """
        self._require_fitted()
        assert self._rho is not None
        return self._rho

    @property
    def n_pairs(self) -> int:
        """Number of colocated pairs found by the nearest-neighbor join.

        Returns
        -------
        int
            Count of paired finite samples.
        """
        self._require_fitted()
        assert self._n_pairs is not None
        return self._n_pairs

    def fit(
        self,
        xa: ArrayLike,
        ya: ArrayLike,
        za: ArrayLike,
        xb: ArrayLike,
        yb: ArrayLike,
        zb: ArrayLike,
    ) -> CrossVariogram:
        """Fit the cross-variogram model between ``a`` and ``b``.

        Parameters
        ----------
        xa, ya, za : array_like
            First variable coordinates and values.
        xb, yb, zb : array_like
            Second variable coordinates and values.

        Returns
        -------
        CrossVariogram
            Fitted cross-variogram. Non-finite rows are ignored separately on
            each variable before colocated pairing.

        Raises
        ------
        ValueError
            If either variable has invalid sizes or no finite rows.

        Notes
        -----
        Rows where ``xa``/``ya``/``za`` (or ``xb``/``yb``/``zb``) are ``NaN`` or
        infinite are ignored on their respective side before pairing.
        """
        xa_array = as_float64_1d(xa, "xa")
        ya_array = as_float64_1d(ya, "ya")
        za_array = as_float64_1d(za, "za")
        validate_same_size_xyz(xa_array, ya_array, za_array)
        xb_array = as_float64_1d(xb, "xb")
        yb_array = as_float64_1d(yb, "yb")
        zb_array = as_float64_1d(zb, "zb")
        validate_same_size_xyz(xb_array, yb_array, zb_array)

        xa_valid, ya_valid, za_valid = filter_finite_xyz(xa_array, ya_array, za_array)
        xb_valid, yb_valid, zb_valid = filter_finite_xyz(xb_array, yb_array, zb_array)
        if xa_valid.size == 0 or xb_valid.size == 0:
            raise ValueError("cross-variogram requires at least one finite point on each side")

        from ._native import run_fit_cross_variogram

        native_colocation_dist = 0.0 if self._max_colocation_dist is None else self._max_colocation_dist
        model_values, rho, n_pairs = run_fit_cross_variogram(
            xa_valid,
            ya_valid,
            za_valid,
            xb_valid,
            yb_valid,
            zb_valid,
            native_colocation_dist,
            self.n_lags,
            self.max_pairs,
            self.max_distance,
            self.n_threads,
        )

        self._rho = rho
        self._n_pairs = n_pairs
        self._model_values = model_values
        self._fitted = True
        return self

    def evaluate(self, distance: ArrayLike) -> np.ndarray:
        """Evaluate the fitted cross-semivariance at one or more distances.

        Parameters
        ----------
        distance : array_like
            Distances in the same coordinate units used during fitting.

        Returns
        -------
        numpy.ndarray
            Cross-semivariance values with the same shape as ``distance``.
        """
        return self._as_variogram().evaluate(distance)

    def __call__(self, distance: ArrayLike) -> np.ndarray:
        """Evaluate the fitted cross-semivariance.

        Parameters
        ----------
        distance : array_like
            Distances in the same coordinate units used during fitting.

        Returns
        -------
        numpy.ndarray
            Cross-semivariance values with the same shape as ``distance``.
        """
        return self.evaluate(distance)

    def _as_variogram(self) -> Variogram:
        """Build a :class:`~apbase.variogram.Variogram` from the fitted model.

        Not cached: :meth:`~apbase.variogram.Variogram.from_model` is cheap
        (pure arithmetic, no spatial search), and building it lazily -- only
        when a reader actually wants model_params/evaluate -- is what lets
        :attr:`rho`/:attr:`n_pairs`/:attr:`model_values` stay available even
        when the fit degenerated (see :attr:`model_values`).
        """
        self._require_fitted()
        assert self._model_values is not None
        return Variogram.from_model(self._model_values)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("call fit(xa, ya, za, xb, yb, zb) before using the cross-variogram")

    def _validate_config(self) -> None:
        if self.n_lags <= 0:
            raise ValueError("n_lags must be greater than zero")
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be >= 0")
        max_pairs_ceiling = resolve_variogram_max_pairs_ceiling()
        if self.max_pairs > max_pairs_ceiling:
            raise ValueError(f"max_pairs must be <= {max_pairs_ceiling}")
        if not np.isfinite(self.max_distance) or self.max_distance < 0.0:
            raise ValueError("max_distance must be finite and >= 0")
        if self._max_colocation_dist is not None and (
            not np.isfinite(self._max_colocation_dist) or self._max_colocation_dist <= 0.0
        ):
            raise ValueError("max_colocation_dist must be None or finite and greater than zero")


__all__ = ["CrossVariogram"]
