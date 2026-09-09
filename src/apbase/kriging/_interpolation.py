from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from apbase.common.arrays import (
    as_float64_1d,
    extract_target_xy,
    filter_finite_xyz,
    validate_same_size_xyz,
)
from apbase.common.exceptions import KRIGING_STATUS_ERRORS, raise_status_error
from apbase.common.validation import require_neighbor_bounds
from apbase.config import resolve_max_neighbors_ceiling, resolve_n_threads, resolve_variogram_profile
from apbase.variogram import Variogram

from ._native import run_kriging

if TYPE_CHECKING:
    from apbase.grid import Grid


class Kriging:
    """Fit a variogram model and interpolate values with local ordinary kriging.

    Stores finite source samples, fits or accepts a variogram model, and
    estimates target values with a local ordinary kriging kernel.
    Coordinates are not projected internally; pass projected/metric
    coordinates when ``radius`` is metric.

    Parameters
    ----------
    radius:
        Local search radius. If ``None``, kriging uses ``range / 3`` from
        the selected variogram.
    max_neighbors:
        Maximum source points used per target coordinate. Capped at
        ``apbase.config["max_neighbors_ceiling"]`` (default ``500``).
    min_neighbors:
        Minimum source points required to produce a finite estimate.
    model_values:
        Optional native variogram model vector with five values. If
        provided, ``fit`` uses it instead of fitting a variogram from
        ``x``, ``y``, and ``z``.

    Examples
    --------
    >>> import numpy as np
    >>> from apbase.kriging import Kriging
    >>> x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0])
    >>> y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0])
    >>> z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0])
    >>> targets = np.ascontiguousarray([[5.0, 5.0]])
    >>> values = Kriging(radius=20.0, min_neighbors=3).fit(x, y, z).interpolate(targets)
    """

    __slots__ = (
        "_radius",
        "_max_neighbors",
        "_min_neighbors",
        "_initial_model_values",
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
        radius: float | None = None,
        max_neighbors: int = 40,
        min_neighbors: int = 3,
        model_values: ArrayLike | None = None,
    ) -> None:
        self._radius = None if radius is None else float(radius)
        self._max_neighbors = int(max_neighbors)
        self._min_neighbors = int(min_neighbors)
        self._initial_model_values = (
            None
            if model_values is None
            else Variogram._normalize_model_values(model_values)
        )

        self._validate_config(self._max_neighbors, self._min_neighbors)
        if self.radius is not None and (not np.isfinite(self.radius) or self.radius <= 0.0):
            raise ValueError("radius must be finite and greater than zero")

        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._z: np.ndarray | None = None
        self._model_values: np.ndarray | None = None
        self._model_params: dict[str, float | int | str] | None = None
        self._fitted = False

    @property
    def radius(self) -> float | None:
        """Local search radius, or ``None`` to derive it from the variogram range."""
        return self._radius

    @property
    def max_neighbors(self) -> int:
        """Maximum source points used per target coordinate."""
        return self._max_neighbors

    @property
    def min_neighbors(self) -> int:
        """Minimum source points required to produce a finite estimate."""
        return self._min_neighbors

    @property
    def n_threads(self) -> int:
        """OpenMP thread count passed to native routines.

        Resolved live from ``apbase.config["n_threads"]`` (see the class
        docstring) on every access -- not cached at construction time.
        """
        return resolve_n_threads()

    @classmethod
    def from_data(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        **kwargs: Any,
    ) -> Kriging:
        """Create a kriging interpolator and fit it from source data."""
        return cls(**kwargs).fit(x, y, z)

    @property
    def is_fitted(self) -> bool:
        """Whether source data and a variogram model are available."""
        return self._fitted

    @property
    def model_values(self) -> np.ndarray:
        """Best fitted variogram model values."""
        self._require_fitted()
        assert self._model_values is not None
        return self._model_values

    @property
    def model_params(self) -> dict[str, float | int | str]:
        """Readable parameters for the selected variogram model."""
        self._require_fitted()
        assert self._model_params is not None
        return dict(self._model_params)

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
    ) -> Kriging:
        """Store finite source data and fit or attach a variogram model.

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
        Kriging
            The fitted kriging interpolator. Rows where ``x``, ``y``, or ``z``
            are ``NaN`` or infinite are ignored.

        Examples
        --------
        >>> kriging = Kriging(radius=30.0, min_neighbors=3)
        >>> kriging.fit(x, y, z)
        >>> kriging.model_params["range"] > 0.0
        True
        >>> kriging = Kriging(model_values=model_values, radius=30.0).fit(x, y, z)
        """
        x_array = as_float64_1d(x, "x")
        y_array = as_float64_1d(y, "y")
        z_array = as_float64_1d(z, "z")
        validate_same_size_xyz(x_array, y_array, z_array)

        x_valid, y_valid, z_valid = filter_finite_xyz(x_array, y_array, z_array)
        if x_valid.size < self.min_neighbors:
            raise ValueError("kriging requires at least min_neighbors finite points")

        if self._initial_model_values is None:
            n_lags, max_pairs, max_distance = resolve_variogram_profile()
            variogram = Variogram(
                n_lags=n_lags,
                max_pairs=max_pairs,
                max_distance=max_distance,
            ).fit(
                x_valid,
                y_valid,
                z_valid,
            )
        else:
            variogram = Variogram.from_model(self._initial_model_values)

        self._x = x_valid
        self._y = y_valid
        self._z = z_valid
        self._model_values = variogram.model_values
        self._model_params = variogram.model_params
        self._fitted = True
        return self

    def interpolate(
        self,
        targets: Grid | ArrayLike,
    ) -> np.ndarray:
        """Estimate values at target coordinates.

        Parameters
        ----------
        targets:
            Target coordinates as an ``(n_targets, 2)`` array, where column
            ``0`` is x and column ``1`` is y. A fitted :class:`Grid`,
            GeoDataFrame/GeoSeries, or list of ``Point``-like objects is also
            accepted; these are parsed by the shared ``extract_target_xy`` helper.

        Returns
        -------
        numpy.ndarray
            One-dimensional ``float64`` array with one kriging estimate per
            target coordinate. Targets without enough valid neighbors are
            returned as ``NaN``.

        Examples
        --------
        >>> targets = np.ascontiguousarray([[5.0, 5.0], [20.0, 20.0]])
        >>> values = kriging.interpolate(targets)
        """
        x_array, y_array, z_array, model_array, model_params = self._get_fitted_state()

        target_x, target_y = extract_target_xy(targets)
        radius_value = self._resolve_radius(model_params)
        estimates_array, status = run_kriging(
            x_array,
            y_array,
            z_array,
            target_x,
            target_y,
            model_array,
            radius_value,
            self.max_neighbors,
            self.min_neighbors,
            self.n_threads,
        )
        if status != 0:
            raise_status_error(status, KRIGING_STATUS_ERRORS, "native kriging execution failed")

        return estimates_array

    def predict(self, targets: Grid | ArrayLike) -> np.ndarray:
        """Alias for :meth:`interpolate`."""
        return self.interpolate(targets)

    def fit_interpolate(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        targets: Grid | ArrayLike,
    ) -> np.ndarray:
        """Fit the interpolator and immediately estimate target values."""
        return self.fit(x, y, z).interpolate(targets)

    def __call__(
        self,
        targets: Grid | ArrayLike,
    ) -> np.ndarray:
        """Alias for :meth:`interpolate`."""
        return self.interpolate(targets)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("call fit(x, y, z) before interpolate(targets)")

    def _get_fitted_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
        self._require_fitted()
        assert self._x is not None
        assert self._y is not None
        assert self._z is not None
        assert self._model_values is not None
        assert self._model_params is not None
        return self._x, self._y, self._z, self._model_values, self._model_params

    def _resolve_radius(
        self,
        model_params: dict[str, float | int | str],
    ) -> float:
        if self.radius is not None:
            return self.radius

        radius_value = float(model_params["range"]) / 3.0
        if not np.isfinite(radius_value) or radius_value <= 0.0:
            raise ValueError("radius must be finite and greater than zero")
        return radius_value

    @staticmethod
    def _validate_config(max_neighbors: int, min_neighbors: int) -> None:
        require_neighbor_bounds(
            max_neighbors,
            min_neighbors,
            max_neighbors_ceiling=resolve_max_neighbors_ceiling(),
        )


__all__ = ["Kriging"]
