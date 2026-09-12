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
from apbase.common.exceptions import IDW_STATUS_ERRORS, raise_status_error
from apbase.common.validation import require_neighbor_bounds
from apbase.config import resolve_n_threads, resolve_variogram_profile
from apbase.variogram import Variogram

from ._native import run_idw

if TYPE_CHECKING:
    from apbase.grid import Grid


class IDW:
    """Interpolate projected coordinates with inverse distance weighting.

    Stores finite source samples with coordinates ``x``/``y`` and values
    ``z``, then estimates values for target coordinates using inverse
    distance weights. Coordinates are not projected internally; pass
    metric/projected coordinates when ``radius`` is in meters.

    Parameters
    ----------
    radius:
        Local search radius. If ``None``, ``fit`` estimates a variogram and
        uses ``range / 3``.
    power:
        Exponent applied to distance in the IDW weights. ``2.0`` is the
        common inverse-square weighting.
    max_neighbors:
        Maximum source points used per target coordinate.
    min_neighbors:
        Minimum source points required to produce a finite estimate.

    Examples
    --------
    >>> import numpy as np
    >>> from apbase.idw import IDW
    >>> x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0])
    >>> y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0])
    >>> z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0])
    >>> targets = np.ascontiguousarray([[5.0, 5.0], [10.0, 10.0]])
    >>> values = IDW(radius=20.0, min_neighbors=1).fit(x, y, z).interpolate(targets)
    """

    __slots__ = (
        "_radius",
        "_power",
        "_max_neighbors",
        "_min_neighbors",
        "_x",
        "_y",
        "_z",
        "_model_params",
        "_fitted",
    )

    def __init__(
        self,
        *,
        radius: float | None = None,
        power: float = 2.0,
        max_neighbors: int = 40,
        min_neighbors: int = 3,
    ) -> None:
        self._radius = None if radius is None else float(radius)
        self._power = float(power)
        self._max_neighbors = int(max_neighbors)
        self._min_neighbors = int(min_neighbors)
        self._validate_config(
            self._radius,
            self._power,
            self._max_neighbors,
            self._min_neighbors,
        )

        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._z: np.ndarray | None = None
        self._model_params: dict[str, float | int | str] | None = None
        self._fitted = False

    @property
    def radius(self) -> float | None:
        """Local search radius.

        Returns
        -------
        float or None
            Explicit radius, or ``None`` to derive it from the variogram
            range during :meth:`fit`.
        """
        return self._radius

    @property
    def power(self) -> float:
        """Exponent applied to distance in the IDW weights.

        Returns
        -------
        float
            Positive distance exponent.
        """
        return self._power

    @property
    def max_neighbors(self) -> int:
        """Maximum source points used per target coordinate.

        Returns
        -------
        int
            Upper bound on local neighbors.
        """
        return self._max_neighbors

    @property
    def min_neighbors(self) -> int:
        """Minimum source points required to produce a finite estimate.

        Returns
        -------
        int
            Lower bound on local neighbors.
        """
        return self._min_neighbors

    @property
    def n_threads(self) -> int:
        """OpenMP thread count passed to native routines.

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
    ) -> IDW:
        """Create and fit an IDW interpolator.

        Parameters
        ----------
        x, y, z : array_like
            Source coordinates and values.
        **kwargs
            Keyword arguments forwarded to :class:`IDW`.

        Returns
        -------
        IDW
            Fitted IDW interpolator.
        """
        return cls(**kwargs).fit(x, y, z)

    @property
    def is_fitted(self) -> bool:
        """Whether finite source data are available.

        Returns
        -------
        bool
            ``True`` after :meth:`fit` succeeds.
        """
        return self._fitted

    @property
    def model_params(self) -> dict[str, float | int | str]:
        """Readable parameters for the variogram used by the default radius.

        Returns
        -------
        dict
            Variogram parameters used to derive the implicit radius.

        Raises
        ------
        ValueError
            If the interpolator is not fitted or was fitted with an explicit
            radius, so no variogram was estimated.
        """
        self._require_fitted()
        if self._model_params is None:
            raise ValueError(
                "IDW was fit with an explicit radius; no variogram model was fitted"
            )
        return dict(self._model_params)

    def fit(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> IDW:
        """Store finite source samples used by later interpolation.

        Parameters
        ----------
        x:
            One-dimensional source x coordinates. Use projected/metric units
            when ``radius`` is metric.
        y:
            One-dimensional source y coordinates. Must have the same length as
            ``x``.
        z:
            One-dimensional source values to interpolate. Must have the same
            length as ``x`` and ``y``.

        Returns
        -------
        IDW
            The fitted interpolator. Rows where ``x``, ``y``, or ``z`` are
            ``NaN`` or infinite are ignored.

        Raises
        ------
        ValueError
            If input arrays have inconsistent size or fewer than
            ``min_neighbors`` finite rows.

        Examples
        --------
        >>> idw = IDW(radius=30.0, min_neighbors=1)
        >>> idw.fit(x, y, z)
        >>> idw.is_fitted
        True
        """
        x_array = as_float64_1d(x, "x")
        y_array = as_float64_1d(y, "y")
        z_array = as_float64_1d(z, "z")
        validate_same_size_xyz(x_array, y_array, z_array)

        x_valid, y_valid, z_valid = filter_finite_xyz(x_array, y_array, z_array)
        if x_valid.size < self.min_neighbors:
            raise ValueError("IDW requires at least min_neighbors finite points")

        model_params = None
        if self.radius is None:
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
            model_params = variogram.model_params

        self._x = x_valid
        self._y = y_valid
        self._z = z_valid
        self._model_params = model_params
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
            ``0`` is x and column ``1`` is y. A fitted :class:`~apbase.grid.Grid`,
            GeoDataFrame/GeoSeries, or list of ``Point``-like objects is also
            accepted; these are parsed by the shared ``extract_target_xy`` helper.

        Returns
        -------
        numpy.ndarray
            One-dimensional ``float64`` array with one interpolated value per
            target coordinate. Targets without enough neighbors are returned
            as ``NaN``.

        Examples
        --------
        >>> targets = np.ascontiguousarray([[5.0, 5.0], [20.0, 20.0]])
        >>> values = idw.interpolate(targets)
        """
        x_array, y_array, z_array, model_params = self._get_fitted_state()

        radius_value = self._resolve_radius(model_params)
        if x_array.size < self.min_neighbors:
            raise ValueError("IDW requires at least min_neighbors finite points")

        estimates_array, status = self._run_targets(
            targets,
            x_array,
            y_array,
            z_array,
            radius_value,
            self.power,
            self.max_neighbors,
            self.min_neighbors,
            self.n_threads,
        )
        if status != 0:
            raise_status_error(status, IDW_STATUS_ERRORS, "native IDW execution failed")
        return estimates_array

    def predict(self, targets: Grid | ArrayLike) -> np.ndarray:
        """Estimate values at target coordinates.

        Parameters
        ----------
        targets : Grid or array_like
            Target coordinates accepted by :meth:`interpolate`.

        Returns
        -------
        numpy.ndarray
            One interpolated value per target coordinate.
        """
        return self.interpolate(targets)

    def __call__(self, targets: Grid | ArrayLike) -> np.ndarray:
        """Estimate values at target coordinates.

        Parameters
        ----------
        targets : Grid or array_like
            Target coordinates accepted by :meth:`interpolate`.

        Returns
        -------
        numpy.ndarray
            One interpolated value per target coordinate.
        """
        return self.interpolate(targets)

    def _run_targets(
        self,
        targets: Grid | ArrayLike,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        radius: float,
        power: float,
        max_neighbors: int,
        min_neighbors: int,
        n_threads: int,
    ) -> tuple[np.ndarray, int]:
        target_x, target_y = extract_target_xy(targets)
        if target_x.size == 0:
            raise ValueError("targets must contain at least one target point")
        return run_idw(
            x,
            y,
            z,
            target_x,
            target_y,
            radius,
            power,
            max_neighbors,
            min_neighbors,
            n_threads,
        )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("call fit(x, y, z) before interpolate(targets)")

    def _get_fitted_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str] | None]:
        self._require_fitted()
        assert self._x is not None
        assert self._y is not None
        assert self._z is not None
        return self._x, self._y, self._z, self._model_params

    def _resolve_radius(
        self,
        model_params: dict[str, float | int | str] | None,
    ) -> float:
        if self.radius is not None:
            return self.radius

        if model_params is None:
            raise ValueError("IDW requires a fitted variogram or explicit radius")
        radius_value = float(model_params["range"]) / 3.0
        if not np.isfinite(radius_value) or radius_value <= 0.0:
            raise ValueError("radius must be finite and greater than zero")
        return radius_value

    @staticmethod
    def _validate_config(
        radius: float | None,
        power: float,
        max_neighbors: int,
        min_neighbors: int,
    ) -> None:
        if radius is not None and (not np.isfinite(radius) or radius <= 0.0):
            raise ValueError("radius must be finite and greater than zero")
        if not np.isfinite(power) or power <= 0.0:
            raise ValueError("power must be finite and greater than zero")
        require_neighbor_bounds(max_neighbors, min_neighbors)


__all__ = ["IDW"]
