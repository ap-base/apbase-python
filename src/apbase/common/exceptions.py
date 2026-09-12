from __future__ import annotations

from typing import NoReturn


class ApbaseError(Exception):
    """Base class for all errors raised by apbase."""


class ValidationError(ApbaseError, ValueError):
    """Raised when input arrays or parameters fail validation."""


class NativeExtensionError(ApbaseError):
    """Raised when the compiled native extension is missing or invalid."""


class NativeExecutionError(ApbaseError):
    """Raised when a native extension call reports a non-zero status."""


class VariogramError(ApbaseError):
    """Raised when variogram fitting or evaluation fails."""


class CrossVariogramError(ApbaseError):
    """Raised when cross-variogram fitting or covariate screening fails."""


class SpatialFilterError(ApbaseError):
    """Raised when spatial filtering fails."""


class InterpolationError(ApbaseError):
    """Raised when IDW or kriging interpolation fails."""


class CrossValidationError(ApbaseError):
    """Raised when leave-one-out cross-validation fails."""


def raise_status_error(
    status: int,
    status_errors: dict[int, type[NativeExecutionError]],
    fallback_message: str,
) -> NoReturn:
    """Raise the exception mapped to a native status code.

    Parameters
    ----------
    status : int
        Status code returned by a native routine.
    status_errors : dict[int, type[NativeExecutionError]]
        Mapping from known status codes to concrete exception classes.
    fallback_message : str
        Message used when ``status`` is not present in ``status_errors``.

    Returns
    -------
    NoReturn
        This function always raises.

    Raises
    ------
    NativeExecutionError
        Always raised, either as a mapped subclass or as the generic fallback.
    """
    error_cls = status_errors.get(status)
    if error_cls is not None:
        raise error_cls()
    raise NativeExecutionError(f"{fallback_message} (status {status})")


# ---------------------------------------------------------------------------
# IDW
# ---------------------------------------------------------------------------


class IDWInvalidDataError(InterpolationError, NativeExecutionError):
    """Raised when the native IDW kernel receives invalid source or target data."""

    def __init__(self) -> None:
        super().__init__("invalid source or target data")


class IDWInvalidRadiusError(InterpolationError, NativeExecutionError):
    """Raised when the IDW search radius is not finite and greater than zero."""

    def __init__(self) -> None:
        super().__init__("radius must be finite and greater than zero")


class IDWInvalidPowerError(InterpolationError, NativeExecutionError):
    """Raised when the IDW distance power is not finite and greater than zero."""

    def __init__(self) -> None:
        super().__init__("power must be finite and greater than zero")


class IDWInvalidConfigurationError(InterpolationError, NativeExecutionError):
    """Raised when the IDW neighbor, grid, or thread configuration is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid neighbor, grid, or thread configuration")


class IDWAllocationError(InterpolationError, NativeExecutionError):
    """Raised when the native IDW kernel fails to allocate its workspace."""

    def __init__(self) -> None:
        super().__init__("native IDW allocation failed")


IDW_STATUS_ERRORS: dict[int, type[NativeExecutionError]] = {
    1: IDWInvalidDataError,
    2: IDWInvalidRadiusError,
    3: IDWInvalidPowerError,
    4: IDWInvalidConfigurationError,
    5: IDWAllocationError,
}


# ---------------------------------------------------------------------------
# Kriging
# ---------------------------------------------------------------------------


class KrigingInvalidDataError(InterpolationError, NativeExecutionError):
    """Raised when the native kriging kernel receives invalid source or target data."""

    def __init__(self) -> None:
        super().__init__("invalid source or target data")


class KrigingInvalidRadiusError(InterpolationError, NativeExecutionError):
    """Raised when the kriging search radius is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid radius")


class KrigingInvalidConfigurationError(InterpolationError, NativeExecutionError):
    """Raised when the kriging neighbor configuration is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid neighbor configuration")


class KrigingAllocationError(InterpolationError, NativeExecutionError):
    """Raised when the native kriging kernel fails to allocate its workspace."""

    def __init__(self) -> None:
        super().__init__("kriging workspace allocation failed")


class KrigingCapacityExceededError(InterpolationError, NativeExecutionError):
    """Raised when the kriging workspace capacity is exceeded."""

    def __init__(self) -> None:
        super().__init__("kriging workspace capacity exceeded")


class KrigingInvalidModelError(InterpolationError, NativeExecutionError):
    """Raised when kriging receives an invalid variogram model."""

    def __init__(self) -> None:
        super().__init__("invalid variogram model")


KRIGING_STATUS_ERRORS: dict[int, type[NativeExecutionError]] = {
    1: KrigingInvalidDataError,
    2: KrigingInvalidRadiusError,
    3: KrigingInvalidConfigurationError,
    4: KrigingAllocationError,
    5: KrigingCapacityExceededError,
    6: KrigingInvalidModelError,
}


# ---------------------------------------------------------------------------
# Cokriging
# ---------------------------------------------------------------------------


class CoKrigingInvalidDataError(InterpolationError, NativeExecutionError):
    """Raised when the native cokriging kernel receives invalid source or target data."""

    def __init__(self) -> None:
        super().__init__("invalid source or target data")


class CoKrigingInvalidRadiusError(InterpolationError, NativeExecutionError):
    """Raised when the cokriging search radius is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid radius")


class CoKrigingInvalidConfigurationError(InterpolationError, NativeExecutionError):
    """Raised when the cokriging neighbor configuration is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid neighbor configuration")


class CoKrigingAllocationError(InterpolationError, NativeExecutionError):
    """Raised when the native cokriging kernel fails to allocate its workspace."""

    def __init__(self) -> None:
        super().__init__("cokriging workspace allocation failed")


class CoKrigingCapacityExceededError(InterpolationError, NativeExecutionError):
    """Raised when the cokriging workspace capacity is exceeded."""

    def __init__(self) -> None:
        super().__init__("cokriging workspace capacity exceeded")


class CoKrigingInvalidModelError(InterpolationError, NativeExecutionError):
    """Raised when cokriging receives an invalid coregionalization model."""

    def __init__(self) -> None:
        super().__init__("invalid coregionalization model")


class CoKrigingNonPSDCoregionalizationError(InterpolationError, NativeExecutionError):
    """Raised when the fitted coregionalization matrix is not positive semidefinite.

    Reserved from Milestone 1 onward but only ever raised starting at
    Milestone 3 (ICM): the collocated model (rho per secondary) has no
    coregionalization matrix to validate.
    """

    def __init__(self) -> None:
        super().__init__("coregionalization matrix is not positive semidefinite")


COKRIGING_STATUS_ERRORS: dict[int, type[NativeExecutionError]] = {
    1: CoKrigingInvalidDataError,
    2: CoKrigingInvalidRadiusError,
    3: CoKrigingInvalidConfigurationError,
    4: CoKrigingAllocationError,
    5: CoKrigingCapacityExceededError,
    6: CoKrigingInvalidModelError,
    7: CoKrigingNonPSDCoregionalizationError,
}


# ---------------------------------------------------------------------------
# Spatial filtering
# ---------------------------------------------------------------------------


class SpatialFilterInvalidDataError(SpatialFilterError, NativeExecutionError):
    """Raised when the native spatial filter kernel receives invalid source data."""

    def __init__(self) -> None:
        super().__init__("invalid source data")


class SpatialFilterInvalidRadiusError(SpatialFilterError, NativeExecutionError):
    """Raised when the spatial filter search radius is not finite and greater than zero."""

    def __init__(self) -> None:
        super().__init__("radius must be finite and greater than zero")


class SpatialFilterInvalidConfigurationError(SpatialFilterError, NativeExecutionError):
    """Raised when the spatial filter neighbor, cell, filter, or thread configuration is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid neighbor, cell, filter, or thread configuration")


class SpatialFilterAllocationError(SpatialFilterError, NativeExecutionError):
    """Raised when the native spatial filter kernel fails to allocate its workspace."""

    def __init__(self) -> None:
        super().__init__("native spatial filter allocation failed")


FILTER_STATUS_ERRORS: dict[int, type[NativeExecutionError]] = {
    1: SpatialFilterInvalidDataError,
    2: SpatialFilterInvalidRadiusError,
    3: SpatialFilterInvalidConfigurationError,
    4: SpatialFilterAllocationError,
}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


class CrossValidationInvalidDataError(CrossValidationError, NativeExecutionError):
    """Raised when the native cross-validation kernel receives invalid source or sample data."""

    def __init__(self) -> None:
        super().__init__("invalid source or sample data")


class CrossValidationInvalidRadiusError(CrossValidationError, NativeExecutionError):
    """Raised when the cross-validation search radius is not finite and greater than zero."""

    def __init__(self) -> None:
        super().__init__("radius must be finite and greater than zero")


class CrossValidationInvalidPowerError(CrossValidationError, NativeExecutionError):
    """Raised when the cross-validation IDW power is not finite and greater than zero."""

    def __init__(self) -> None:
        super().__init__("power must be finite and greater than zero")


class CrossValidationInvalidConfigurationError(CrossValidationError, NativeExecutionError):
    """Raised when the cross-validation neighbor, grid, or thread configuration is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid neighbor, grid, or thread configuration")


class CrossValidationAllocationError(CrossValidationError, NativeExecutionError):
    """Raised when the native cross-validation kernel fails to allocate its workspace."""

    def __init__(self) -> None:
        super().__init__("native cross-validation allocation failed")


class CrossValidationInvalidModelError(CrossValidationError, NativeExecutionError):
    """Raised when cross-validation receives an invalid variogram model."""

    def __init__(self) -> None:
        super().__init__("invalid variogram model")


class CrossValidationInvalidSampleIndexError(CrossValidationError, NativeExecutionError):
    """Raised when a cross-validation sample index is out of range."""

    def __init__(self) -> None:
        super().__init__("sample_idx contains an out-of-range index")


CROSS_VALIDATE_STATUS_ERRORS: dict[int, type[NativeExecutionError]] = {
    1: CrossValidationInvalidDataError,
    2: CrossValidationInvalidRadiusError,
    3: CrossValidationInvalidPowerError,
    4: CrossValidationInvalidConfigurationError,
    5: CrossValidationAllocationError,
    6: CrossValidationInvalidModelError,
    7: CrossValidationInvalidSampleIndexError,
}


# ---------------------------------------------------------------------------
# Variogram
# ---------------------------------------------------------------------------


class VariogramInvalidModelIdError(VariogramError, NativeExecutionError):
    """Raised when the fitted model_info vector has a non-finite model id."""

    def __init__(self) -> None:
        super().__init__("model_info returned a non-finite model_id")


class VariogramInvalidModelError(VariogramError, NativeExecutionError):
    """Raised when the native extension returns an unrecognized model id."""

    def __init__(self) -> None:
        super().__init__("invalid model returned by the native extension")


class VariogramInvalidRangeError(VariogramError, NativeExecutionError):
    """Raised when the fitted variogram range is invalid."""

    def __init__(self) -> None:
        super().__init__("variogram returned an invalid range")


class VariogramInvalidThreadCountError(VariogramError, NativeExecutionError):
    """Raised when the requested thread count is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid thread count")


SELECT_STATUS_ERRORS: dict[int, type[NativeExecutionError]] = {
    1: VariogramInvalidModelIdError,
    2: VariogramInvalidModelError,
    3: VariogramInvalidRangeError,
    4: VariogramInvalidThreadCountError,
}


# ---------------------------------------------------------------------------
# Cross-variogram / covariate screening
# ---------------------------------------------------------------------------
#
# fit_cross_variogram (the fit layer) has no status output, mirroring
# fit_variogram_models -- see cross_variogram_fit.f90. These map
# screen_covariate's status codes; screen_secondary_variables (the high-level
# API) deliberately does NOT raise on them -- a non-zero status there (e.g.
# insufficient paired sample) is a normal, reportable per-candidate outcome,
# recorded in CovariateReport.status instead of raised. They exist for
# callers using the lower-level native wrapper directly.


class CrossVariogramInsufficientSampleError(CrossVariogramError, NativeExecutionError):
    """Raised when fewer than 4 colocated pairs are available for the Fisher z-test."""

    def __init__(self) -> None:
        super().__init__("fewer than 4 colocated pairs available")


class CrossVariogramInvalidAutoModelError(CrossVariogramError, NativeExecutionError):
    """Raised when a primary or candidate auto-variogram model is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid primary or candidate auto-variogram model")


class CrossVariogramInvalidCrossModelError(CrossVariogramError, NativeExecutionError):
    """Raised when the fitted cross-variogram model is invalid."""

    def __init__(self) -> None:
        super().__init__("invalid cross-variogram model")


CROSS_VARIOGRAM_STATUS_ERRORS: dict[int, type[NativeExecutionError]] = {
    1: CrossVariogramInsufficientSampleError,
    2: CrossVariogramInvalidAutoModelError,
    3: CrossVariogramInvalidCrossModelError,
}


__all__ = [
    "ApbaseError",
    "CrossValidationError",
    "CrossVariogramError",
    "InterpolationError",
    "NativeExecutionError",
    "NativeExtensionError",
    "SpatialFilterError",
    "ValidationError",
    "VariogramError",
]
