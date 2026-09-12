from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from apbase.common.arrays import as_float64_1d, filter_finite_xyz, validate_same_size_xyz
from apbase.common.exceptions import SELECT_STATUS_ERRORS, raise_status_error
from apbase.config import resolve_n_threads, resolve_variogram_max_pairs_ceiling

from ._native import run_evaluate_variogram, run_select_variogram, run_variogram

VARIOGRAM_NAMES = {
    1: "spherical",
    2: "exponential",
    3: "gaussian",
}


class Variogram:
    """Fit, select, and evaluate a native variogram model.

    Fits candidate semivariogram models from finite source samples and
    selects the best native model for evaluation or kriging. Coordinates
    are not projected internally; pass coordinates in the same unit
    expected by downstream distances.

    Parameters
    ----------
    n_lags:
        Number of lag bins used by the native fitter.
    max_pairs:
        Maximum sampled point pairs. ``0`` uses every eligible pair, capped
        internally at ``1_000_000``. An explicit value is capped at
        ``apbase.config["variogram_max_pairs_ceiling"]`` (default
        ``200_000``).
    max_distance:
        Maximum pair distance considered by the fitter. ``0`` uses the
        native default extent.

    Examples
    --------
    >>> import numpy as np
    >>> from apbase.variogram import Variogram
    >>> x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0])
    >>> y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0])
    >>> z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0])
    >>> variogram = Variogram(n_lags=4, max_pairs=0).fit(x, y, z)
    >>> gamma = variogram.evaluate(np.ascontiguousarray([5.0, 10.0]))
    """

    __slots__ = (
        "_n_lags",
        "_max_pairs",
        "_max_distance",
        "_x",
        "_y",
        "_z",
        "_model_values",
        "_model_params",
        "_fitted",
    )

    def __init__(
        self,
        *,
        n_lags: int = 50,
        max_pairs: int = 100_000,
        max_distance: float = 0.0,
    ) -> None:
        self._n_lags = int(n_lags)
        self._max_pairs = int(max_pairs)
        self._max_distance = float(max_distance)
        self._validate_config()

        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._z: np.ndarray | None = None
        self._model_values: np.ndarray | None = None
        self._model_params: dict[str, float | int | str] | None = None
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
        """Maximum sampled point pairs.

        Returns
        -------
        int
            Pair sampling cap; ``0`` means every eligible pair subject to the
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
    def n_threads(self) -> int:
        """Explicit OpenMP thread count used when evaluating the model.

        Resolved live from ``apbase.config["n_threads"]`` (see the class
        docstring) on every access -- not cached at construction time.

        Returns
        -------
        int
            Current OpenMP thread count.
        """
        return resolve_n_threads()

    @classmethod
    def from_data(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        **kwargs: Any,
    ) -> Variogram:
        """Create and fit a variogram from source data.

        Parameters
        ----------
        x, y, z : array_like
            Source coordinates and values.
        **kwargs
            Keyword arguments forwarded to :class:`Variogram`.

        Returns
        -------
        Variogram
            Fitted variogram.
        """
        return cls(**kwargs).fit(x, y, z)

    @classmethod
    def from_model(cls, model_values: ArrayLike) -> Variogram:
        """Create a variogram from an existing native model vector.

        Parameters
        ----------
        model_values : array_like
            Native model vector with five values:
            ``[model_id, nugget, partial_sill, range, sse]``.

        Returns
        -------
        Variogram
            Variogram instance backed by ``model_values``.
        """
        variogram = cls()
        variogram.set_model(model_values)
        return variogram

    @property
    def is_fitted(self) -> bool:
        """Whether a model is already available.

        Returns
        -------
        bool
            ``True`` after :meth:`fit` or :meth:`set_model` succeeds.
        """
        return self._fitted

    @property
    def model_values(self) -> np.ndarray:
        """Native model vector used by kriging and evaluation kernels.

        Returns
        -------
        numpy.ndarray
            Model vector ``[model_id, nugget, partial_sill, range, sse]``.
        """
        self._require_fitted()
        assert self._model_values is not None
        return self._model_values

    @property
    def model_params(self) -> dict[str, float | int | str]:
        """Readable parameters for the selected variogram model.

        Returns
        -------
        dict
            Model id/name, nugget, partial sill, sill, range, and SSE.
        """
        self._require_fitted()
        assert self._model_params is not None
        return dict(self._model_params)

    @property
    def range(self) -> float:
        """Selected model range.

        Returns
        -------
        float
            Fitted variogram range in coordinate units.
        """
        return float(self.model_params["range"])

    def fit(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Variogram:
        """Fit candidate variogram models from source samples.

        Parameters
        ----------
        x:
            One-dimensional source x coordinates.
        y:
            One-dimensional source y coordinates. Must have the same length as
            ``x``.
        z:
            One-dimensional source values. Must have the same length as
            ``x`` and ``y``.

        Returns
        -------
        Variogram
            The fitted variogram. Rows where ``x``, ``y``, or ``z`` are
            ``NaN`` or infinite are ignored.

        Raises
        ------
        ValueError
            If input arrays have inconsistent size or no finite rows.
        NativeExecutionError
            If the native variogram fit reports a failure status.

        Examples
        --------
        >>> variogram = Variogram(n_lags=8, max_pairs=0).fit(x, y, z)
        >>> variogram.model_params["model_name"]
        'spherical'
        """
        x_array = as_float64_1d(x, "x")
        y_array = as_float64_1d(y, "y")
        z_array = as_float64_1d(z, "z")
        validate_same_size_xyz(x_array, y_array, z_array)

        x_valid, y_valid, z_valid = filter_finite_xyz(x_array, y_array, z_array)
        if x_valid.size == 0:
            raise ValueError("variogram requires at least one finite point")

        model_values = run_variogram(
            x_valid,
            y_valid,
            z_valid,
            self.n_lags,
            self.max_pairs,
            self.max_distance,
            self.n_threads,
        )
        model_params = self._select_model_params(model_values)

        self._x = x_valid
        self._y = y_valid
        self._z = z_valid
        self._model_values = model_values
        self._model_params = model_params
        self._fitted = True
        return self

    def set_model(self, model_values: ArrayLike) -> Variogram:
        """Attach an existing native model vector and select its parameters.

        Parameters
        ----------
        model_values : array_like
            Native model vector with five values:
            ``[model_id, nugget, partial_sill, range, sse]``.

        Returns
        -------
        Variogram
            This instance, marked as fitted.

        Raises
        ------
        ValueError
            If the model vector shape or contents are invalid.
        """
        model_array = self._normalize_model_values(model_values)
        model_params = self._select_model_params(model_array)
        self._x = None
        self._y = None
        self._z = None
        self._model_values = model_array
        self._model_params = model_params
        self._fitted = True
        return self

    def evaluate(self, distance: ArrayLike) -> np.ndarray:
        """Evaluate semivariance for distances.

        Parameters
        ----------
        distance:
            Scalar-like or array-like distances in the same coordinate unit
            used during :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            Semivariance values with the same shape as ``distance``.

        Examples
        --------
        >>> gamma = variogram.evaluate(np.asarray([5.0, 10.0]))
        """
        self._require_fitted()
        gamma, status = run_evaluate_variogram(self.model_values, distance, self.n_threads)
        if status != 0:
            raise_status_error(status, SELECT_STATUS_ERRORS, "error evaluating variogram")
        return gamma

    def __call__(self, distance: ArrayLike) -> np.ndarray:
        """Evaluate semivariance for distances.

        Parameters
        ----------
        distance : array_like
            Distances in the same coordinate unit used during :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            Semivariance values with the same shape as ``distance``.
        """
        return self.evaluate(distance)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("call fit(x, y, z) or set_model(model_values) before using the variogram")

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

    @staticmethod
    def _normalize_model_values(model_values: ArrayLike) -> np.ndarray:
        model_array = as_float64_1d(model_values, "model_values")
        if model_array.size != 5:
            raise ValueError("model_values must contain 5 values")
        return np.ascontiguousarray(model_array, dtype=np.float64)

    @staticmethod
    def _select_model_params(model_values: np.ndarray) -> dict[str, float | int | str]:
        params_array, status = run_select_variogram(model_values)
        if status != 0:
            raise_status_error(status, SELECT_STATUS_ERRORS, "error selecting variogram")

        model_id = int(round(float(params_array[0])))
        try:
            model_name = VARIOGRAM_NAMES[model_id]
        except KeyError:
            raise ValueError(f"invalid model returned by the native extension: {model_id}") from None

        return {
            "model_id": model_id,
            "model_name": model_name,
            "nugget": float(params_array[1]),
            "partial_sill": float(params_array[2]),
            "sill": float(params_array[3]),
            "range": float(params_array[4]),
            "sse": float(params_array[5]),
        }


__all__ = ["Variogram"]
