from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import floor
from typing import Literal, overload

import numpy as np

from apbase.config import resolve_local_mode_extent_km_ceiling

from .exceptions import ValidationError

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class CoordinateTransform:
    """Metadata needed to invert :func:`prepare_metric_xy`'s rebase/reprojection.

    Obtained via ``prepare_metric_xy(..., return_transform=True)`` and passed to
    :func:`restore_original_xy` to recover coordinates in the original input metric.
    """

    coordinate_system: str
    geographic_mode: str | None
    epsg: int | None
    x0: float
    y0: float
    lon0: float | None
    lat0: float | None


def guess_xy_coordinate_system(x: np.ndarray, y: np.ndarray, assume_finite: bool = False) -> str:
    if assume_finite:
        xmin = float(x.min())
        xmax = float(x.max())
        ymin = float(y.min())
        ymax = float(y.max())
    else:
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return "invalid"

        xv = x[mask]
        yv = y[mask]
        xmin = float(xv.min())
        xmax = float(xv.max())
        ymin = float(yv.min())
        ymax = float(yv.max())

    if (
        -180.0 <= xmin <= 180.0
        and -180.0 <= xmax <= 180.0
        and -90.0 <= ymin <= 90.0
        and -90.0 <= ymax <= 90.0
    ):
        return "geographic"

    if (
        100_000.0 <= xmin <= 900_000.0
        and 100_000.0 <= xmax <= 900_000.0
        and 0.0 <= ymin <= 10_000_000.0
        and 0.0 <= ymax <= 10_000_000.0
    ):
        return "utm"

    return "projected"


@lru_cache(maxsize=64)
def _utm_transformer(epsg: int):
    try:
        from pyproj import Transformer
    except ImportError:
        raise ValidationError("pyproj is required to convert geographic coordinates to UTM") from None

    try:
        return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    except Exception:
        raise ValidationError("failed to prepare UTM conversion") from None


def _utm_epsg_from_lonlat(lon: np.ndarray, lat: np.ndarray) -> int:
    mask = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(mask):
        raise ValidationError("lon/lat must contain at least one finite coordinate")

    lon0 = float(np.mean(lon[mask]))
    lat0 = float(np.mean(lat[mask]))
    zone = int(floor((lon0 + 180.0) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    return (32600 if lat0 >= 0.0 else 32700) + zone


def _lonlat_to_local_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    lon0: float,
    lat0: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat0_rad = np.deg2rad(lat0)
    x_local = EARTH_RADIUS_M * np.cos(lat0_rad) * np.deg2rad(lon - lon0)
    y_local = EARTH_RADIUS_M * np.deg2rad(lat - lat0)
    return np.ascontiguousarray(x_local), np.ascontiguousarray(y_local)


def _local_mode_extent_km(x_local: np.ndarray, y_local: np.ndarray, finite_mask: np.ndarray) -> float:
    """Bounding-box diagonal (km) of already-projected local-tangent-plane meters.

    Shared by :func:`_check_local_mode_extent` (raises past the ceiling) and
    :func:`_auto_geographic_mode` (picks 'utm' instead of raising), so both
    read the same extent instead of two independent hypot passes.
    """
    if not np.any(finite_mask):
        return 0.0
    xv = x_local[finite_mask]
    yv = y_local[finite_mask]
    return float(np.hypot(xv.max() - xv.min(), yv.max() - yv.min())) / 1000.0


def _check_local_mode_extent(x_local: np.ndarray, y_local: np.ndarray, finite_mask: np.ndarray) -> None:
    """Reject ``geographic_mode="local"`` once the input outgrows the flat-plane approximation.

    ``x_local``/``y_local`` are already-projected local-tangent-plane meters
    (see :func:`_lonlat_to_local_xy`); the approximation error grows with
    distance from the anchor lon0/lat0, so a large bounding-box diagonal is
    a direct proxy for approximation error, not just a heuristic on the raw
    lon/lat values.
    """
    if not np.any(finite_mask):
        return
    ceiling_km = resolve_local_mode_extent_km_ceiling()
    extent_km = _local_mode_extent_km(x_local, y_local, finite_mask)
    if extent_km > ceiling_km:
        raise ValidationError(
            f"geographic_mode='local' approximates distance with a flat tangent plane anchored on "
            f"the input's mean lon/lat, which grows inaccurate over large extents; this input spans "
            f"~{extent_km:.0f} km, above the {ceiling_km:.0f} km ceiling "
            "(apbase.config['local_mode_extent_km_ceiling']). Use geographic_mode='utm' for "
            "regional or multi-zone extents."
        )


def _auto_geographic_mode(x_local: np.ndarray, y_local: np.ndarray, finite_mask: np.ndarray) -> str:
    """Pick 'local' vs 'utm' from the input's own extent when geographic_mode is not given.

    Mirrors :func:`_check_local_mode_extent`'s ceiling but chooses instead of
    raising: small extents (<= local_mode_extent_km_ceiling, 50 km by
    default) get the cheaper flat-plane 'local' projection (no UTM zone
    lookup or pyproj dependency); larger ones get 'utm' automatically, since
    'local' would just be rejected for them anyway.
    """
    extent_km = _local_mode_extent_km(x_local, y_local, finite_mask)
    return "local" if extent_km <= resolve_local_mode_extent_km_ceiling() else "utm"


def _lonlat_to_utm_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    epsg: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(lon) & np.isfinite(lat)
    x_utm = np.full(lon.shape, np.nan, dtype=np.float64)
    y_utm = np.full(lat.shape, np.nan, dtype=np.float64)

    transformer = _utm_transformer(epsg)
    try:
        x_valid, y_valid = transformer.transform(lon[mask], lat[mask])
    except Exception:
        raise ValidationError("error converting geographic coordinates to UTM") from None
    x_utm[mask] = x_valid
    y_utm[mask] = y_valid
    return np.ascontiguousarray(x_utm), np.ascontiguousarray(y_utm)


@overload
def prepare_metric_xy(
    x: np.ndarray,
    y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    geographic_mode: str | None = None,
    *,
    return_transform: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int | None]: ...


@overload
def prepare_metric_xy(
    x: np.ndarray,
    y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    geographic_mode: str | None = None,
    *,
    return_transform: Literal[True],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int | None, CoordinateTransform]: ...


def prepare_metric_xy(
    x: np.ndarray,
    y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    geographic_mode: str | None = None,
    return_transform: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int | None]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int | None, CoordinateTransform]
):
    """Convert source/target coordinates to a shared metric system, rebased to zero.

    Parameters
    ----------
    x, y:
        Source coordinates.
    target_x, target_y:
        Target coordinates. Pass the same arrays as ``x``/``y`` to reuse the
        source coordinates as targets; this is detected by identity, not by
        value, and avoids a redundant conversion.
    geographic_mode:
        ``"local"``, ``"utm"``, or ``None`` (default). Only used when
        ``x``/``y`` are detected as geographic; ignored otherwise.
        ``None`` auto-detects: the input's own bounding-box diagonal is
        compared against ``apbase.config["local_mode_extent_km_ceiling"]``
        (50 km by default, see
        :func:`apbase.config.resolve_local_mode_extent_km_ceiling`) and
        ``"local"`` is picked below it, ``"utm"`` above -- the same threshold
        that makes an explicit ``"local"`` raise. Pass ``"local"`` or
        ``"utm"`` explicitly to force that projection regardless of extent;
        forcing ``"local"`` past the ceiling still raises, since the
        flat-plane approximation is not valid there.
    return_transform:
        When ``True``, append a :class:`CoordinateTransform` capturing this
        call's rebase (and reprojection, if any) as a 7th return value. Pass
        it to :func:`restore_original_xy` to recover coordinates in the
        original input metric after downstream processing.

    Returns
    -------
    x_metric, y_metric:
        Rebased source coordinates.
    target_x_metric, target_y_metric:
        Rebased target coordinates.
    coordinate_system:
        One of ``"geographic"``, ``"utm"``, or ``"projected"`` (see
        :func:`guess_xy_coordinate_system`), describing the *detected input*
        system, not the output (output is always metric/rebased).
    epsg:
        The UTM EPSG code used for conversion, or ``None`` when the input
        was not geographic or ``geographic_mode="local"`` was used.
    transform:
        Only present when ``return_transform=True``. See above.
    """
    if geographic_mode is not None and geographic_mode not in {"local", "utm"}:
        raise ValidationError("geographic_mode must be 'local', 'utm', or None (auto-detect)")

    target_is_source = target_x is x and target_y is y
    coordinate_system = guess_xy_coordinate_system(x, y)
    epsg = None
    lon0 = None
    lat0 = None
    mode: str | None = None

    if coordinate_system == "geographic":
        source_mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(source_mask):
            raise ValidationError("x and y must contain finite coordinates")

        mean_lon0 = float(np.mean(x[source_mask]))
        mean_lat0 = float(np.mean(y[source_mask]))

        mode = geographic_mode
        x_local = y_local = None
        if mode is None:
            x_local, y_local = _lonlat_to_local_xy(x, y, mean_lon0, mean_lat0)
            mode = _auto_geographic_mode(x_local, y_local, source_mask)

        if mode == "utm":
            epsg = _utm_epsg_from_lonlat(x, y)
            x_metric, y_metric = _lonlat_to_utm_xy(x, y, epsg)
            if target_is_source:
                target_x_metric = x_metric
                target_y_metric = y_metric
            else:
                target_x_metric, target_y_metric = _lonlat_to_utm_xy(target_x, target_y, epsg)
        else:
            lon0 = mean_lon0
            lat0 = mean_lat0
            if x_local is None or y_local is None:
                x_metric, y_metric = _lonlat_to_local_xy(x, y, lon0, lat0)
            else:
                x_metric, y_metric = x_local, y_local
            _check_local_mode_extent(x_metric, y_metric, source_mask)
            if target_is_source:
                target_x_metric = x_metric
                target_y_metric = y_metric
            else:
                target_x_metric, target_y_metric = _lonlat_to_local_xy(target_x, target_y, lon0, lat0)
    else:
        x_metric = np.array(x, dtype=np.float64, copy=True)
        y_metric = np.array(y, dtype=np.float64, copy=True)
        if target_is_source:
            target_x_metric = x_metric
            target_y_metric = y_metric
        else:
            target_x_metric = np.array(target_x, dtype=np.float64, copy=True)
            target_y_metric = np.array(target_y, dtype=np.float64, copy=True)

    finite_source = np.isfinite(x_metric) & np.isfinite(y_metric)
    if not np.any(finite_source):
        raise ValidationError("x and y must contain finite coordinates")

    x0 = float(x_metric[finite_source].min())
    y0 = float(y_metric[finite_source].min())
    x_metric[finite_source] -= x0
    y_metric[finite_source] -= y0

    if target_is_source:
        target_x_metric = x_metric
        target_y_metric = y_metric
    else:
        finite_target = np.isfinite(target_x_metric) & np.isfinite(target_y_metric)
        target_x_metric[finite_target] -= x0
        target_y_metric[finite_target] -= y0

    if not return_transform:
        return (
            np.ascontiguousarray(x_metric),
            np.ascontiguousarray(y_metric),
            np.ascontiguousarray(target_x_metric),
            np.ascontiguousarray(target_y_metric),
            coordinate_system,
            epsg,
        )

    transform = CoordinateTransform(
        coordinate_system=coordinate_system,
        geographic_mode=mode,
        epsg=epsg,
        x0=x0,
        y0=y0,
        lon0=lon0,
        lat0=lat0,
    )
    return (
        np.ascontiguousarray(x_metric),
        np.ascontiguousarray(y_metric),
        np.ascontiguousarray(target_x_metric),
        np.ascontiguousarray(target_y_metric),
        coordinate_system,
        epsg,
        transform,
    )


def restore_original_xy(
    x: np.ndarray,
    y: np.ndarray,
    transform: CoordinateTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`prepare_metric_xy`: recover coordinates in the original input metric.

    Parameters
    ----------
    x, y:
        Coordinates produced downstream of a ``prepare_metric_xy``-rebased pipeline
        (e.g. a grid or interpolation result), in the same rebased metric system.
    transform:
        The :class:`CoordinateTransform` returned by
        ``prepare_metric_xy(..., return_transform=True)`` for the original conversion.

    Returns
    -------
    x_original, y_original:
        Coordinates in the original input system (lon/lat, real UTM, or the
        original projected unit — matching what was passed into
        ``prepare_metric_xy``).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_out = np.full(x.shape, np.nan, dtype=np.float64)
    y_out = np.full(y.shape, np.nan, dtype=np.float64)
    if not np.any(mask):
        return np.ascontiguousarray(x_out), np.ascontiguousarray(y_out)

    x_shifted = x[mask] + transform.x0
    y_shifted = y[mask] + transform.y0

    if transform.coordinate_system != "geographic":
        x_out[mask] = x_shifted
        y_out[mask] = y_shifted
        return np.ascontiguousarray(x_out), np.ascontiguousarray(y_out)

    if transform.geographic_mode == "utm":
        try:
            from pyproj.enums import TransformDirection
        except ImportError:
            raise ValidationError(
                "pyproj is required to convert UTM coordinates back to geographic"
            ) from None

        transformer = _utm_transformer(transform.epsg)
        try:
            lon, lat = transformer.transform(x_shifted, y_shifted, direction=TransformDirection.INVERSE)
        except Exception:
            raise ValidationError("error converting UTM coordinates back to geographic") from None
        x_out[mask] = lon
        y_out[mask] = lat
        return np.ascontiguousarray(x_out), np.ascontiguousarray(y_out)

    assert transform.lon0 is not None and transform.lat0 is not None  # geographic_mode == "local"
    lat0_rad = np.deg2rad(transform.lat0)
    x_out[mask] = transform.lon0 + np.rad2deg(x_shifted / (EARTH_RADIUS_M * np.cos(lat0_rad)))
    y_out[mask] = transform.lat0 + np.rad2deg(y_shifted / EARTH_RADIUS_M)
    return np.ascontiguousarray(x_out), np.ascontiguousarray(y_out)


def apply_metric_transform(
    x: np.ndarray,
    y: np.ndarray,
    transform: CoordinateTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an existing :class:`CoordinateTransform` to new x/y coordinates.

    Forward counterpart to :func:`restore_original_xy`: converts ``x``/``y`` -- in
    the same original system the transform was derived from -- into the same
    rebased metric system ``prepare_metric_xy`` produced, without recomputing a
    new rebase origin. Useful to bring auxiliary geometry (e.g. a boundary) into
    the same metric space as coordinates already processed by
    ``prepare_metric_xy``.

    Parameters
    ----------
    x, y:
        Coordinates in the same original system the transform was derived from.
    transform:
        The :class:`CoordinateTransform` returned by
        ``prepare_metric_xy(..., return_transform=True)``.

    Returns
    -------
    x_metric, y_metric:
        Coordinates in the rebased metric system.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_out = np.full(x.shape, np.nan, dtype=np.float64)
    y_out = np.full(y.shape, np.nan, dtype=np.float64)
    if not np.any(mask):
        return np.ascontiguousarray(x_out), np.ascontiguousarray(y_out)

    if transform.geographic_mode == "utm":
        assert transform.epsg is not None
        x_metric, y_metric = _lonlat_to_utm_xy(x[mask], y[mask], transform.epsg)
    elif transform.geographic_mode == "local":
        assert transform.lon0 is not None and transform.lat0 is not None
        x_metric, y_metric = _lonlat_to_local_xy(x[mask], y[mask], transform.lon0, transform.lat0)
    else:
        x_metric = x[mask]
        y_metric = y[mask]

    x_out[mask] = x_metric - transform.x0
    y_out[mask] = y_metric - transform.y0
    return np.ascontiguousarray(x_out), np.ascontiguousarray(y_out)


def reproject_geographic_xy(
    x: np.ndarray,
    y: np.ndarray,
    geographic_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic (lon/lat) coordinates to metric, with no rebase.

    Unlike :func:`prepare_metric_xy`, the result is **not** shifted to a
    zero-based origin: it is the raw UTM easting/northing
    (``geographic_mode="utm"``) or raw local-equirectangular meters anchored
    on the input's own mean lon/lat (``geographic_mode="local"``), each
    derived independently from ``x``/``y`` themselves.

    Use this -- instead of :func:`apply_metric_transform` -- for auxiliary
    geometry (e.g. a boundary) when there is no existing
    :class:`CoordinateTransform` to reuse and the geometry must be converted
    on its own terms, without assuming it shares an origin/rebase with any
    other already-processed coordinates.

    Parameters
    ----------
    x, y:
        Geographic (lon/lat) coordinates.
    geographic_mode:
        ``"local"``, ``"utm"``, or ``None`` (default). ``None`` auto-detects
        the same way :func:`prepare_metric_xy` does: ``"local"`` below
        ``apbase.config["local_mode_extent_km_ceiling"]`` (50 km by default),
        ``"utm"`` above it. Pass ``"local"`` or ``"utm"`` explicitly to force
        that projection; forcing ``"local"`` past the ceiling still raises.

    Returns
    -------
    x_metric, y_metric:
        Raw projected/local coordinates, not rebased.
    """
    if geographic_mode is not None and geographic_mode not in {"local", "utm"}:
        raise ValidationError("geographic_mode must be 'local', 'utm', or None (auto-detect)")

    mask = np.isfinite(x) & np.isfinite(y)
    x_out = np.full(x.shape, np.nan, dtype=np.float64)
    y_out = np.full(y.shape, np.nan, dtype=np.float64)
    if not np.any(mask):
        return np.ascontiguousarray(x_out), np.ascontiguousarray(y_out)

    mode = geographic_mode
    x_local = y_local = None
    if mode is None:
        lon0 = float(np.mean(x[mask]))
        lat0 = float(np.mean(y[mask]))
        x_local, y_local = _lonlat_to_local_xy(x[mask], y[mask], lon0, lat0)
        mode = _auto_geographic_mode(x_local, y_local, np.ones(x_local.shape, dtype=bool))

    if mode == "utm":
        epsg = _utm_epsg_from_lonlat(x[mask], y[mask])
        x_metric, y_metric = _lonlat_to_utm_xy(x[mask], y[mask], epsg)
    else:
        if x_local is None or y_local is None:
            lon0 = float(np.mean(x[mask]))
            lat0 = float(np.mean(y[mask]))
            x_metric, y_metric = _lonlat_to_local_xy(x[mask], y[mask], lon0, lat0)
        else:
            x_metric, y_metric = x_local, y_local
        _check_local_mode_extent(x_metric, y_metric, np.ones(x_metric.shape, dtype=bool))

    x_out[mask] = x_metric
    y_out[mask] = y_metric
    return np.ascontiguousarray(x_out), np.ascontiguousarray(y_out)


__all__ = [
    "CoordinateTransform",
    "apply_metric_transform",
    "guess_xy_coordinate_system",
    "prepare_metric_xy",
    "reproject_geographic_xy",
    "restore_original_xy",
]
