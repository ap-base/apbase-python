from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from apbase.common.arrays import (
    as_float64_1d,
    extract_target_xy,
    filter_finite_xyz,
    validate_same_size_xyz,
)
from apbase.common.exceptions import COKRIGING_STATUS_ERRORS, raise_status_error
from apbase.common.validation import require_neighbor_bounds
from apbase.config import (
    resolve_max_lmc_structures_ceiling,
    resolve_max_neighbors_ceiling,
    resolve_max_secondary_variables_ceiling,
    resolve_n_threads,
    resolve_variogram_profile,
)
from apbase.cross_variogram import CrossVariogram
from apbase.variogram import Variogram

from ._native import (
    run_co_kriging_icm,
    run_co_kriging_lmc,
    run_collocated_cokriging,
    run_fit_icm_model,
    run_fit_lmc_model,
)

if TYPE_CHECKING:
    from apbase.grid import Grid

_METHODS = ("icm", "lmc", "collocated")
_MODEL_NAMES = {1: "spherical", 2: "exponential", 3: "gaussian"}
SecondaryInput = (
    Mapping[str, tuple[ArrayLike, ArrayLike, ArrayLike]] | Sequence[tuple[ArrayLike, ArrayLike, ArrayLike]]
)


@dataclass(frozen=True, slots=True)
class _CollocatedFit:
    model_values: np.ndarray
    rhos: np.ndarray


@dataclass(frozen=True, slots=True)
class _ICMFit:
    model_id: float
    model_range: float
    nugget_matrix: np.ndarray
    sill_matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class _LMCFit:
    model_id: float
    structure_ranges: np.ndarray
    coefficient_matrices: np.ndarray


class CoKriging:
    """Fit a coregionalization model and interpolate values with cokriging.

    Estimates a primary variable using one or more spatially correlated
    secondary variables (e.g. NDVI/elevation/soil EC alongside a sparse
    yield sample). Run :func:`~apbase.cross_variogram.
    screen_secondary_variables` first to pick admissible secondaries --
    ``CoKriging`` does not screen its inputs itself.

    Parameters
    ----------
    method:
        ``"collocated"``, ``"icm"`` (Intrinsic Coregionalization Model), or
        ``"lmc"`` (full Linear Model of Coregionalization).
    radius:
        Local search radius for the primary neighborhood. If ``None``, uses
        ``range / 3`` from the primary's variogram. For
        ``method="collocated"``, each secondary must also have a value
        within this radius of every target, or that target comes back
        ``NaN``; ``"icm"``/``"lmc"`` search each variable's neighborhood
        independently, so heterotopic secondaries work directly.
    max_neighbors, min_neighbors:
        Primary neighbor bounds, same meaning as :class:`~apbase.kriging.
        Kriging`.
    n_structures:
        ``method="lmc"`` only: number of nested coregionalization
        structures, including the nugget. Default ``2``.
    model_values:
        Optional pre-fitted coregionalization model vector,
        ``"collocated"`` only. If provided, :meth:`fit` skips fitting the
        primary variogram and per-secondary cross-variograms.
    """

    __slots__ = (
        "_method",
        "_radius",
        "_max_neighbors",
        "_min_neighbors",
        "_initial_model_values",
        "_x0",
        "_y0",
        "_z0",
        "_primary_model_params",
        "_secondary_names",
        "_secondary_x",
        "_secondary_y",
        "_secondary_z",
        "_secondary_offset",
        "_n_structures",
        "_fit",
    )

    def __init__(
        self,
        *,
        method: str = "collocated",
        radius: float | None = None,
        max_neighbors: int = 40,
        min_neighbors: int = 3,
        n_structures: int = 2,
        model_values: ArrayLike | None = None,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if model_values is not None and method != "collocated":
            raise NotImplementedError(f'model_values override is not supported yet for method="{method}"')
        self._method = method
        self._radius = None if radius is None else float(radius)
        self._max_neighbors = int(max_neighbors)
        self._min_neighbors = int(min_neighbors)
        self._n_structures = int(n_structures)
        if self._n_structures < 2:
            raise ValueError("n_structures must be >= 2 (nugget + at least one continuous structure)")
        self._initial_model_values = (
            None if model_values is None else self._normalize_model_values(model_values)
        )

        self._validate_config(self._max_neighbors, self._min_neighbors)
        if self.radius is not None and (not np.isfinite(self.radius) or self.radius <= 0.0):
            raise ValueError("radius must be finite and greater than zero")

        self._x0: np.ndarray | None = None
        self._y0: np.ndarray | None = None
        self._z0: np.ndarray | None = None
        self._primary_model_params: dict[str, float | int | str] | None = None
        self._secondary_names: tuple[str, ...] | None = None
        self._secondary_x: np.ndarray | None = None
        self._secondary_y: np.ndarray | None = None
        self._secondary_z: np.ndarray | None = None
        self._secondary_offset: np.ndarray | None = None
        self._fit: _CollocatedFit | _ICMFit | _LMCFit | None = None

    @property
    def method(self) -> str:
        return self._method

    @property
    def radius(self) -> float | None:
        return self._radius

    @property
    def max_neighbors(self) -> int:
        return self._max_neighbors

    @property
    def min_neighbors(self) -> int:
        return self._min_neighbors

    @property
    def n_structures(self) -> int:
        """Number of nested LMC structures (S). Only meaningful for ``method="lmc"``."""
        return self._n_structures

    @property
    def n_threads(self) -> int:
        """OpenMP thread count. Resolved live from ``apbase.config``."""
        return resolve_n_threads()

    @classmethod
    def from_data(
        cls,
        x0: ArrayLike,
        y0: ArrayLike,
        z0: ArrayLike,
        secondaries: SecondaryInput,
        **kwargs: Any,
    ) -> CoKriging:
        """Create a cokriging interpolator and fit it from source data."""
        return cls(**kwargs).fit(x0, y0, z0, secondaries)

    @property
    def is_fitted(self) -> bool:
        return self._fit is not None

    @property
    def model_values(self) -> np.ndarray:
        """Native coregionalization model vector.

        ``"collocated"``: ``[model_id, nugget, partial_sill, range, sse,
        rho_1..rho_K]``. ``"icm"``: ``[model_id, range,
        nugget_matrix.ravel(), sill_matrix.ravel()]``. ``"lmc"``:
        ``[model_id, structure_ranges, coefficient_matrices.ravel()]``.
        Flattened for a single-vector view in every case -- see
        :attr:`model_params` for the actual matrices.
        """
        self._require_fitted()
        if isinstance(self._fit, _CollocatedFit):
            return self._fit.model_values
        if isinstance(self._fit, _ICMFit):
            return np.concatenate(
                [
                    np.asarray([self._fit.model_id, self._fit.model_range], dtype=np.float64),
                    self._fit.nugget_matrix.ravel(),
                    self._fit.sill_matrix.ravel(),
                ]
            )
        assert isinstance(self._fit, _LMCFit)
        return np.concatenate(
            [
                np.asarray([self._fit.model_id], dtype=np.float64),
                self._fit.structure_ranges,
                self._fit.coefficient_matrices.ravel(),
            ]
        )

    @property
    def model_params(self) -> dict[str, Any]:
        """Readable fitted parameters -- shape differs by :attr:`method`.

        ``"collocated"``: ``{"primary": {...}, "rho": {name: rho, ...}}``.
        ``"icm"``: ``{"primary": {...}, "model_id", "model_name", "range",
        "secondary_names", "nugget_matrix", "sill_matrix"}``. ``"lmc"``:
        ``{"primary": {...}, "model_id", "model_name", "structure_ranges",
        "secondary_names", "coefficient_matrices"}`` (``coefficient_matrices``
        has shape ``(n_structures, K+1, K+1)``, index 0 = nugget structure).
        Variable 0 of every matrix is the primary, 1..K follow
        ``secondary_names`` order.
        """
        self._require_fitted()
        assert self._primary_model_params is not None
        assert self._secondary_names is not None
        if isinstance(self._fit, _CollocatedFit):
            return {
                "primary": dict(self._primary_model_params),
                "rho": dict(
                    zip(self._secondary_names, (float(r) for r in self._fit.rhos), strict=True)
                ),
            }
        if isinstance(self._fit, _ICMFit):
            model_id = int(round(self._fit.model_id))
            return {
                "primary": dict(self._primary_model_params),
                "model_id": model_id,
                "model_name": _MODEL_NAMES.get(model_id, "unknown"),
                "range": self._fit.model_range,
                "secondary_names": self._secondary_names,
                "nugget_matrix": self._fit.nugget_matrix,
                "sill_matrix": self._fit.sill_matrix,
            }
        assert isinstance(self._fit, _LMCFit)
        model_id = int(round(self._fit.model_id))
        return {
            "primary": dict(self._primary_model_params),
            "model_id": model_id,
            "model_name": _MODEL_NAMES.get(model_id, "unknown"),
            "structure_ranges": self._fit.structure_ranges,
            "secondary_names": self._secondary_names,
            "coefficient_matrices": self._fit.coefficient_matrices,
        }

    def fit(
        self,
        x0: ArrayLike,
        y0: ArrayLike,
        z0: ArrayLike,
        secondaries: SecondaryInput,
    ) -> CoKriging:
        """Fit the coregionalization model from source data.

        Parameters
        ----------
        x0, y0, z0:
            Primary variable source coordinates and values.
        secondaries:
            Mapping of name to ``(x, y, z)`` source arrays, or a sequence of
            ``(x, y, z)`` tuples (named ``secondary_1``, ``secondary_2``, ...
            in that case) -- e.g.
            ``screen_secondary_variables(...).selected`` works directly here.

        Returns
        -------
        CoKriging
            The fitted interpolator. Rows where any of ``x0``/``y0``/``z0``
            (or a given secondary's own coordinates/values) are ``NaN`` or
            infinite are ignored on their respective side.
        """
        x0_array = as_float64_1d(x0, "x0")
        y0_array = as_float64_1d(y0, "y0")
        z0_array = as_float64_1d(z0, "z0")
        validate_same_size_xyz(x0_array, y0_array, z0_array)
        x0_valid, y0_valid, z0_valid = filter_finite_xyz(x0_array, y0_array, z0_array)
        if x0_valid.size < self.min_neighbors:
            raise ValueError("cokriging requires at least min_neighbors finite primary points")

        names, secondary_arrays = self._normalize_secondaries(secondaries)
        if not secondary_arrays:
            raise ValueError("cokriging requires at least one secondary variable")
        max_secondaries = resolve_max_secondary_variables_ceiling()
        if len(secondary_arrays) > max_secondaries:
            raise ValueError(f"number of secondaries must be <= {max_secondaries}")

        z0_mean = float(np.mean(z0_valid))
        z0_std = float(np.std(z0_valid))
        if not np.isfinite(z0_std) or z0_std <= 0.0:
            raise ValueError("primary has zero variance; cannot standardize secondaries")

        standardized_x: list[np.ndarray] = []
        standardized_y: list[np.ndarray] = []
        standardized_z: list[np.ndarray] = []
        offsets = [0]
        for name, (xu, yu, zu) in zip(names, secondary_arrays, strict=True):
            zu_mean = float(np.mean(zu))
            zu_std = float(np.std(zu))
            if not np.isfinite(zu_std) or zu_std <= 0.0:
                raise ValueError(f"secondary {name!r} has zero variance; cannot standardize")
            standardized_x.append(xu)
            standardized_y.append(yu)
            standardized_z.append(z0_mean + (zu - zu_mean) * (z0_std / zu_std))
            offsets.append(offsets[-1] + xu.size)

        secondary_x = np.concatenate(standardized_x)
        secondary_y = np.concatenate(standardized_y)
        secondary_z = np.concatenate(standardized_z)
        secondary_offset = np.ascontiguousarray(offsets, dtype=np.int32)

        n_lags, max_pairs, max_distance = resolve_variogram_profile()

        if self._method == "collocated":
            self._fit_collocated(
                x0_valid, y0_valid, z0_valid, names, secondary_arrays, n_lags, max_pairs, max_distance
            )
        elif self._method == "icm":
            self._fit_icm(
                x0_valid, y0_valid, z0_valid, secondary_x, secondary_y, secondary_z, secondary_offset,
                n_lags, max_pairs, max_distance,
            )
        else:
            max_structures = resolve_max_lmc_structures_ceiling()
            if self._n_structures > max_structures:
                raise ValueError(f"n_structures must be <= {max_structures}")
            self._fit_lmc(
                x0_valid, y0_valid, z0_valid, secondary_x, secondary_y, secondary_z, secondary_offset,
                n_lags, max_pairs, max_distance,
            )

        self._x0 = x0_valid
        self._y0 = y0_valid
        self._z0 = z0_valid
        self._secondary_names = tuple(names)
        self._secondary_x = secondary_x
        self._secondary_y = secondary_y
        self._secondary_z = secondary_z
        self._secondary_offset = secondary_offset
        return self

    def _fit_collocated(
        self,
        x0_valid: np.ndarray,
        y0_valid: np.ndarray,
        z0_valid: np.ndarray,
        names: list[str],
        secondary_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        n_lags: int,
        max_pairs: int,
        max_distance: float,
    ) -> None:
        if self._initial_model_values is None:
            primary_variogram = Variogram(n_lags=n_lags, max_pairs=max_pairs, max_distance=max_distance).fit(
                x0_valid, y0_valid, z0_valid
            )
            rhos = []
            for xu, yu, zu in secondary_arrays:
                cross = CrossVariogram(n_lags=n_lags, max_pairs=max_pairs, max_distance=max_distance).fit(
                    x0_valid, y0_valid, z0_valid, xu, yu, zu
                )
                rhos.append(cross.rho)
            model_values = np.concatenate(
                [primary_variogram.model_values, np.asarray(rhos, dtype=np.float64)]
            )
            primary_model_params = primary_variogram.model_params
        else:
            model_values = self._initial_model_values
            k_expected = model_values.size - 5
            if k_expected != len(secondary_arrays):
                raise ValueError(
                    f"model_values encodes {k_expected} secondaries, but {len(secondary_arrays)} were given"
                )
            primary_model_params = Variogram.from_model(model_values[:5]).model_params

        self._fit = _CollocatedFit(model_values=model_values, rhos=model_values[5:])
        self._primary_model_params = primary_model_params

    def _fit_icm(
        self,
        x0_valid: np.ndarray,
        y0_valid: np.ndarray,
        z0_valid: np.ndarray,
        secondary_x: np.ndarray,
        secondary_y: np.ndarray,
        secondary_z: np.ndarray,
        secondary_offset: np.ndarray,
        n_lags: int,
        max_pairs: int,
        max_distance: float,
    ) -> None:
        model_id, model_range, nugget_matrix, sill_matrix, status = run_fit_icm_model(
            x0_valid, y0_valid, z0_valid, secondary_x, secondary_y, secondary_z, secondary_offset,
            n_lags, max_pairs, max_distance, 0.0, self.n_threads,
        )
        if status != 0:
            raise_status_error(status, COKRIGING_STATUS_ERRORS, "native ICM fit failed")

        self._fit = _ICMFit(
            model_id=model_id, model_range=model_range, nugget_matrix=nugget_matrix, sill_matrix=sill_matrix
        )
        self._primary_model_params = {
            "model_id": int(round(model_id)),
            "model_name": _MODEL_NAMES.get(int(round(model_id)), "unknown"),
            "nugget": float(nugget_matrix[0, 0]),
            "partial_sill": float(sill_matrix[0, 0]),
            "sill": float(nugget_matrix[0, 0] + sill_matrix[0, 0]),
            "range": float(model_range),
        }

    def _fit_lmc(
        self,
        x0_valid: np.ndarray,
        y0_valid: np.ndarray,
        z0_valid: np.ndarray,
        secondary_x: np.ndarray,
        secondary_y: np.ndarray,
        secondary_z: np.ndarray,
        secondary_offset: np.ndarray,
        n_lags: int,
        max_pairs: int,
        max_distance: float,
    ) -> None:
        model_id, structure_ranges, coefficient_matrices, status = run_fit_lmc_model(
            x0_valid, y0_valid, z0_valid, secondary_x, secondary_y, secondary_z, secondary_offset,
            self._n_structures, n_lags, max_pairs, max_distance, 0.0, self.n_threads,
        )
        if status != 0:
            raise_status_error(status, COKRIGING_STATUS_ERRORS, "native LMC fit failed")

        self._fit = _LMCFit(
            model_id=model_id, structure_ranges=structure_ranges, coefficient_matrices=coefficient_matrices
        )
        nugget_diag = float(coefficient_matrices[0, 0, 0])
        continuous_diag = float(coefficient_matrices[1:, 0, 0].sum())
        self._primary_model_params = {
            "model_id": int(round(model_id)),
            "model_name": _MODEL_NAMES.get(int(round(model_id)), "unknown"),
            "nugget": nugget_diag,
            "partial_sill": continuous_diag,
            "sill": nugget_diag + continuous_diag,
            "range": float(structure_ranges[-1]),
        }

    def interpolate(self, targets: Grid | ArrayLike) -> np.ndarray:
        """Estimate primary values at target coordinates.

        Parameters
        ----------
        targets:
            Target coordinates, same accepted forms as
            :meth:`~apbase.kriging.Kriging.interpolate`.

        Returns
        -------
        numpy.ndarray
            One estimate per target. A target is ``NaN`` if it has fewer than
            ``min_neighbors`` primary neighbors within ``radius``; for
            ``method="collocated"``, also if any secondary has no source
            point within ``radius`` of it (see the class docstring).
        """
        self._require_fitted()
        assert self._x0 is not None
        assert self._y0 is not None
        assert self._z0 is not None
        assert self._primary_model_params is not None
        assert self._secondary_x is not None
        assert self._secondary_y is not None
        assert self._secondary_z is not None
        assert self._secondary_offset is not None

        target_x, target_y = extract_target_xy(targets)
        radius_value = self._resolve_radius(self._primary_model_params)

        if isinstance(self._fit, _CollocatedFit):
            estimates, status = run_collocated_cokriging(
                self._x0, self._y0, self._z0, target_x, target_y, self._fit.model_values,
                self._secondary_x, self._secondary_y, self._secondary_z, self._secondary_offset,
                radius_value, self.max_neighbors, self.min_neighbors, self.n_threads,
            )
        elif isinstance(self._fit, _ICMFit):
            estimates, status = run_co_kriging_icm(
                self._x0, self._y0, self._z0, target_x, target_y,
                self._fit.model_id, self._fit.model_range, self._fit.nugget_matrix, self._fit.sill_matrix,
                self._secondary_x, self._secondary_y, self._secondary_z, self._secondary_offset,
                radius_value, self.max_neighbors, self.min_neighbors, self.n_threads,
            )
        else:
            assert isinstance(self._fit, _LMCFit)
            estimates, status = run_co_kriging_lmc(
                self._x0, self._y0, self._z0, target_x, target_y,
                self._fit.model_id, self._fit.structure_ranges, self._fit.coefficient_matrices,
                self._secondary_x, self._secondary_y, self._secondary_z, self._secondary_offset,
                radius_value, self.max_neighbors, self.min_neighbors, self.n_threads,
            )
        if status != 0:
            raise_status_error(status, COKRIGING_STATUS_ERRORS, "native cokriging execution failed")
        return estimates

    def predict(self, targets: Grid | ArrayLike) -> np.ndarray:
        """Alias for :meth:`interpolate`."""
        return self.interpolate(targets)

    def fit_interpolate(
        self,
        x0: ArrayLike,
        y0: ArrayLike,
        z0: ArrayLike,
        secondaries: SecondaryInput,
        targets: Grid | ArrayLike,
    ) -> np.ndarray:
        """Fit the interpolator and immediately estimate target values."""
        return self.fit(x0, y0, z0, secondaries).interpolate(targets)

    def __call__(self, targets: Grid | ArrayLike) -> np.ndarray:
        """Alias for :meth:`interpolate`."""
        return self.interpolate(targets)

    def _require_fitted(self) -> None:
        if self._fit is None:
            raise ValueError("call fit(x0, y0, z0, secondaries) before interpolate(targets)")

    def _resolve_radius(self, primary_model_params: dict[str, float | int | str]) -> float:
        if self.radius is not None:
            return self.radius
        radius_value = float(primary_model_params["range"]) / 3.0
        if not np.isfinite(radius_value) or radius_value <= 0.0:
            raise ValueError("radius must be finite and greater than zero")
        return radius_value

    @staticmethod
    def _normalize_secondaries(
        secondaries: SecondaryInput,
    ) -> tuple[list[str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        if isinstance(secondaries, Mapping):
            names = [str(name) for name in secondaries]
            raw = [secondaries[name] for name in secondaries]
        else:
            raw = list(secondaries)
            names = [f"secondary_{i + 1}" for i in range(len(raw))]

        arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for name, value in zip(names, raw, strict=True):
            xu, yu, zu = value
            xu_array = as_float64_1d(xu, f"secondaries[{name!r}][0]")
            yu_array = as_float64_1d(yu, f"secondaries[{name!r}][1]")
            zu_array = as_float64_1d(zu, f"secondaries[{name!r}][2]")
            validate_same_size_xyz(xu_array, yu_array, zu_array)
            xu_valid, yu_valid, zu_valid = filter_finite_xyz(xu_array, yu_array, zu_array)
            if xu_valid.size == 0:
                raise ValueError(f"secondary {name!r} has no finite points")
            arrays.append((xu_valid, yu_valid, zu_valid))
        return names, arrays

    @staticmethod
    def _normalize_model_values(model_values: ArrayLike) -> np.ndarray:
        model_array = as_float64_1d(model_values, "model_values")
        if model_array.size < 6:
            raise ValueError("model_values must contain 5 primary values plus at least 1 rho")
        return np.ascontiguousarray(model_array, dtype=np.float64)

    @staticmethod
    def _validate_config(max_neighbors: int, min_neighbors: int) -> None:
        require_neighbor_bounds(
            max_neighbors,
            min_neighbors,
            max_neighbors_ceiling=resolve_max_neighbors_ceiling(),
        )


__all__ = ["CoKriging"]
