from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from apbase.common.arrays import as_float64_1d, finite_mask_xyz, validate_same_size_xyz
from apbase.common.exceptions import FILTER_STATUS_ERRORS, raise_status_error
from apbase.config import resolve_n_threads
from apbase.variogram import Variogram

from ._native import run_filter_mask, run_local_zstats

DEFAULT_MAX_NEIGHBORS = 120
DEFAULT_CELL_FACTOR = 3
DEFAULT_GLOBAL_IQR_WHISKER = 3.0

# Fixed variogram-fitting configuration used to derive radius when it is not
# passed explicitly. Not user-configurable; see Variogram for the underlying
# fitter if a different profile is ever needed.
_VARIOGRAM_N_LAGS = 20
_VARIOGRAM_MAX_PAIRS = 100_000
_VARIOGRAM_MAX_DISTANCE = 0.0

# Ordered loosest -> strictest; values are the native threshold profile ids
# consumed by viz_filter_mask (see filter_mask.f90's filter_thresholds).
FILTER_LEVELS = {
    "light": 1,
    "moderate": 2,
    "strict": 3,
    "aggressive": 4,
}


class FilterStatistics(NamedTuple):
    """Per-point local filtering statistics returned by :attr:`SpatialFilter.statistics`."""

    pct_diff: np.ndarray
    local_z: np.ndarray
    local_prob: np.ndarray


class SpatialFilter:
    """Filter projected spatial samples with global IQR and local z-statistics.

    Removes invalid rows, optionally removes global IQR outliers, and
    applies a local native filter using spatial neighborhoods. Coordinates
    are not projected internally; pass projected/metric coordinates when
    ``radius`` is metric.

    Parameters
    ----------
    radius:
        Local search radius. Defaults to ``range / 3`` from a fitted
        variogram.
    filter_level:
        Strictness, loosest to strictest: ``"light"``, ``"moderate"``,
        ``"strict"``, or ``"aggressive"``.
    apply_global_iqr:
        Whether to remove global IQR outliers before local filtering.
    max_neighbors:
        Maximum local neighbors per source point. ``0`` uses every
        neighbor within ``radius``, uncapped.
    candidate_fraction:
        Fraction (0, 1] of in-radius candidates considered for the local
        sample.
    cell_factor:
        Spatial grid factor used by the local kernel.
    global_iqr_whisker:
        IQR whisker for the optional global filter; lower removes more
        points. Ignored when ``apply_global_iqr=False``.
    dedup_radius:
        Distance below which two points are treated as spatial duplicates
        and collapsed to one representative. ``0.0`` (default) disables
        this.
    dedup_z_tol:
        Extra value-closeness gate combined with ``dedup_radius``: points
        only merge if also within this value difference. ``0.0`` (default)
        means proximity alone is sufficient.

    Examples
    --------
    >>> import numpy as np
    >>> from apbase.filtering import SpatialFilter
    >>> x = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0])
    >>> y = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0])
    >>> z = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0])
    >>> filt = SpatialFilter(radius=20.0, apply_global_iqr=False).fit(x, y, z)
    >>> x_keep, y_keep, z_keep = filt.filter()
    """

    __slots__ = (
        "_radius",
        "_filter_level",
        "_filter_level_code",
        "_apply_global_iqr",
        "_max_neighbors",
        "_candidate_fraction",
        "_cell_factor",
        "_global_iqr_whisker",
        "_dedup_radius",
        "_dedup_z_tol",
        "_x",
        "_y",
        "_z",
        "_keep_mask",
        "_pct_diff",
        "_local_z",
        "_local_prob",
        "_model_params",
        "_fitted",
    )

    def __init__(
        self,
        *,
        radius: float | None = None,
        filter_level: str = "light",
        apply_global_iqr: bool = True,
        max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
        candidate_fraction: float = 1.0,
        cell_factor: int = DEFAULT_CELL_FACTOR,
        global_iqr_whisker: float = DEFAULT_GLOBAL_IQR_WHISKER,
        dedup_radius: float = 0.0,
        dedup_z_tol: float = 0.0,
    ) -> None:
        self._radius = None if radius is None else float(radius)
        self._filter_level = filter_level
        self._apply_global_iqr = bool(apply_global_iqr)
        self._max_neighbors = int(max_neighbors)
        self._candidate_fraction = float(candidate_fraction)
        self._cell_factor = int(cell_factor)
        self._global_iqr_whisker = float(global_iqr_whisker)
        self._dedup_radius = float(dedup_radius)
        self._dedup_z_tol = float(dedup_z_tol)
        self._filter_level_code = self._validate_config(
            self._radius,
            self._filter_level,
            self._max_neighbors,
            self._candidate_fraction,
            self._cell_factor,
            self._global_iqr_whisker,
            self._dedup_radius,
            self._dedup_z_tol,
        )

        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._z: np.ndarray | None = None
        self._keep_mask: np.ndarray | None = None
        self._pct_diff: np.ndarray | None = None
        self._local_z: np.ndarray | None = None
        self._local_prob: np.ndarray | None = None
        self._model_params: dict[str, float | int | str] | None = None
        self._fitted = False

    @property
    def radius(self) -> float | None:
        """Local search radius, or ``None`` to derive it from the variogram range."""
        return self._radius

    @property
    def filter_level(self) -> str:
        """Filtering strictness: ``"light"``, ``"moderate"``, ``"strict"``, or ``"aggressive"``."""
        return self._filter_level

    @property
    def apply_global_iqr(self) -> bool:
        """Whether global IQR outliers are removed before local filtering."""
        return self._apply_global_iqr

    @property
    def max_neighbors(self) -> int:
        """Maximum local neighbors used per source point."""
        return self._max_neighbors

    @property
    def candidate_fraction(self) -> float:
        """Fraction of in-radius candidates given a chance at the local sample."""
        return self._candidate_fraction

    @property
    def cell_factor(self) -> int:
        """Spatial grid factor used by the native local kernel."""
        return self._cell_factor

    @property
    def global_iqr_whisker(self) -> float:
        """IQR whisker used by the optional global filter."""
        return self._global_iqr_whisker

    @property
    def dedup_radius(self) -> float:
        """Spatial duplicate-collapse radius used when building local neighborhoods."""
        return self._dedup_radius

    @property
    def dedup_z_tol(self) -> float:
        """Value-closeness gate paired with ``dedup_radius``."""
        return self._dedup_z_tol

    @property
    def n_threads(self) -> int:
        """Explicit OpenMP thread count passed to native routines.

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
    ) -> SpatialFilter:
        """Create a spatial filter and run it on source data."""
        return cls(**kwargs).fit(x, y, z)

    @property
    def is_fitted(self) -> bool:
        """Whether source data and filtering outputs are available."""
        return self._fitted

    @property
    def model_params(self) -> dict[str, float | int | str]:
        """Readable parameters for the variogram used by the default radius."""
        self._require_fitted()
        if self._model_params is None:
            raise ValueError("SpatialFilter was fit with an explicit radius; no variogram model was fitted")
        return dict(self._model_params)

    @property
    def statistics(self) -> FilterStatistics:
        """Local percent difference, z-score, and probability arrays."""
        self._require_fitted()
        assert self._pct_diff is not None
        assert self._local_z is not None
        assert self._local_prob is not None
        return FilterStatistics(self._pct_diff.copy(), self._local_z.copy(), self._local_prob.copy())

    def fit(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> SpatialFilter:
        """Run global and local filtering for source samples.

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
        SpatialFilter
            The fitted filter. Use :meth:`mask`, :meth:`filter`, or
            :attr:`statistics` to read results.

        Examples
        --------
        >>> filt = SpatialFilter(radius=30.0, apply_global_iqr=False)
        >>> filt.fit(x, y, z)
        >>> keep = filt.mask()
        """
        x_array = as_float64_1d(x, "x")
        y_array = as_float64_1d(y, "y")
        z_array = as_float64_1d(z, "z")
        validate_same_size_xyz(x_array, y_array, z_array)

        finite_mask = finite_mask_xyz(x_array, y_array, z_array)
        if not bool(np.any(finite_mask)):
            raise ValueError("SpatialFilter requires at least one finite point")

        input_mask = finite_mask.copy()
        x_valid = np.ascontiguousarray(x_array[finite_mask])
        y_valid = np.ascontiguousarray(y_array[finite_mask])
        z_valid = np.ascontiguousarray(z_array[finite_mask])

        if self.apply_global_iqr:
            global_mask = self._iqr_mask(z_valid, self.global_iqr_whisker)
            finite_indices = np.flatnonzero(finite_mask)
            input_mask[finite_indices[~global_mask]] = False
            x_valid = np.ascontiguousarray(x_valid[global_mask])
            y_valid = np.ascontiguousarray(y_valid[global_mask])
            z_valid = np.ascontiguousarray(z_valid[global_mask])

        if x_valid.size == 0:
            raise ValueError("SpatialFilter requires at least one point after global filtering")

        model_params = None
        if self.radius is None:
            variogram = Variogram(
                n_lags=_VARIOGRAM_N_LAGS,
                max_pairs=_VARIOGRAM_MAX_PAIRS,
                max_distance=_VARIOGRAM_MAX_DISTANCE,
            ).fit(
                x_valid,
                y_valid,
                z_valid,
            )
            model_params = variogram.model_params

        radius_value = self._resolve_radius(model_params)
        pct_diff, local_z, local_prob, status = run_local_zstats(
            x_valid,
            y_valid,
            z_valid,
            radius_value,
            self.cell_factor,
            self.max_neighbors,
            self.candidate_fraction,
            self.dedup_radius,
            self.dedup_z_tol,
            self.n_threads,
        )
        if status != 0:
            raise_status_error(status, FILTER_STATUS_ERRORS, "native spatial filter execution failed")

        local_mask, status = run_filter_mask(
            pct_diff,
            local_prob,
            local_z,
            self._filter_level_code,
        )
        if status != 0:
            raise_status_error(status, FILTER_STATUS_ERRORS, "native spatial filter mask failed")

        keep_mask = np.zeros(x_array.size, dtype=bool)
        keep_mask[input_mask] = local_mask

        full_pct_diff = np.full(x_array.size, np.nan, dtype=np.float64)
        full_local_z = np.full(x_array.size, np.nan, dtype=np.float64)
        full_local_prob = np.full(x_array.size, np.nan, dtype=np.float64)
        full_pct_diff[input_mask] = pct_diff
        full_local_z[input_mask] = local_z
        full_local_prob[input_mask] = local_prob

        self._x = x_array
        self._y = y_array
        self._z = z_array
        self._keep_mask = keep_mask
        self._pct_diff = full_pct_diff
        self._local_z = full_local_z
        self._local_prob = full_local_prob
        self._model_params = model_params
        self._fitted = True
        return self

    def mask(self) -> np.ndarray:
        """Return the boolean keep mask for the fitted source arrays."""
        self._require_fitted()
        assert self._keep_mask is not None
        return self._keep_mask.copy()

    def filter(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return filtered ``x``, ``y``, and ``z`` arrays.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Source coordinates and values where the keep mask is ``True``.

        Examples
        --------
        >>> x_keep, y_keep, z_keep = filt.filter()
        """
        self._require_fitted()
        assert self._x is not None
        assert self._y is not None
        assert self._z is not None
        assert self._keep_mask is not None
        return self._x[self._keep_mask], self._y[self._keep_mask], self._z[self._keep_mask]

    def fit_mask(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        """Fit the filter and return the boolean keep mask."""
        return self.fit(x, y, z).mask()

    def fit_filter(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the filter and return filtered source arrays."""
        return self.fit(x, y, z).filter()

    def __call__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Alias for :meth:`fit_filter`."""
        return self.fit_filter(x, y, z)

    def _resolve_radius(self, model_params: dict[str, float | int | str] | None) -> float:
        # An explicit self.radius was already validated in __init__ and can't change
        # afterwards, so only the model-derived value needs checking here.
        if self.radius is not None:
            return self.radius

        if model_params is None:
            raise ValueError("SpatialFilter requires a fitted variogram or explicit radius")
        radius_value = float(model_params["range"]) / 3.0
        if not np.isfinite(radius_value) or radius_value <= 0.0:
            raise ValueError("radius must be finite and greater than zero")
        return radius_value

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("call fit(x, y, z) before reading filtering results")

    @staticmethod
    def _validate_config(
        radius: float | None,
        filter_level: str,
        max_neighbors: int,
        candidate_fraction: float,
        cell_factor: int,
        global_iqr_whisker: float,
        dedup_radius: float,
        dedup_z_tol: float,
    ) -> int:
        if radius is not None and (not np.isfinite(radius) or radius <= 0.0):
            raise ValueError("radius must be finite and greater than zero")
        if filter_level not in FILTER_LEVELS:
            valid = ", ".join(repr(name) for name in FILTER_LEVELS)
            raise ValueError(f"filter_level must be one of {valid}, got {filter_level!r}")
        if max_neighbors < 0:
            raise ValueError("max_neighbors must be >= 0")
        if not np.isfinite(candidate_fraction) or candidate_fraction <= 0.0 or candidate_fraction > 1.0:
            raise ValueError("candidate_fraction must be in (0, 1]")
        if cell_factor <= 0:
            raise ValueError("cell_factor must be greater than zero")
        if not np.isfinite(global_iqr_whisker) or global_iqr_whisker < 0.0:
            raise ValueError("global_iqr_whisker must be finite and >= 0")
        if not np.isfinite(dedup_radius) or dedup_radius < 0.0:
            raise ValueError("dedup_radius must be finite and >= 0")
        if not np.isfinite(dedup_z_tol) or dedup_z_tol < 0.0:
            raise ValueError("dedup_z_tol must be finite and >= 0")
        return FILTER_LEVELS[filter_level]

    @staticmethod
    def _iqr_mask(z: np.ndarray, whisker: float) -> np.ndarray:
        q1, q3 = np.quantile(z, (0.25, 0.75))
        iqr = q3 - q1
        lower = q1 - whisker * iqr
        upper = q3 + whisker * iqr
        return (z >= lower) & (z <= upper)


__all__ = ["FilterStatistics", "SpatialFilter"]
