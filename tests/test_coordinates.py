from __future__ import annotations

import numpy as np
import pytest

from apbase.common.coordinates import prepare_metric_xy, reproject_geographic_xy
from apbase.common.exceptions import ValidationError
from apbase.config import config


def _reset() -> None:
    config._values.clear()


def test_prepare_metric_xy_auto_picks_local_below_ceiling() -> None:
    # geographic_mode=None (the default) auto-detects instead of raising:
    # a field-scale extent auto-picks 'local' (epsg stays None).
    lon = np.array([-47.0, -47.001, -47.0, -47.001])
    lat = np.array([-15.0, -15.0, -15.001, -15.001])

    x, y, tx, ty, coordinate_system, epsg = prepare_metric_xy(lon, lat, lon, lat)
    assert coordinate_system == "geographic"
    assert epsg is None


def test_prepare_metric_xy_auto_picks_utm_past_ceiling() -> None:
    # A regional extent auto-picks 'utm' instead of raising or silently
    # falling back to an invalid flat-plane approximation.
    lon = np.array([-46.6, -38.5])
    lat = np.array([-23.5, -12.9])

    x, y, tx, ty, coordinate_system, epsg = prepare_metric_xy(lon, lat, lon, lat)
    assert coordinate_system == "geographic"
    assert epsg is not None


def test_reproject_geographic_xy_auto_detects_geographic_mode() -> None:
    lon = np.array([-47.0, -47.001])
    lat = np.array([-15.0, -15.0])

    x, y = reproject_geographic_xy(lon, lat)
    assert x.shape == lon.shape
    assert y.shape == lat.shape


def test_prepare_metric_xy_local_mode_accepts_field_scale_extent() -> None:
    # A single field, a few hundred meters across -- well within the flat
    # tangent-plane approximation's validity, and the intended use case for
    # geographic_mode="local".
    lon = np.array([-47.0, -47.001, -47.0, -47.001])
    lat = np.array([-15.0, -15.0, -15.001, -15.001])

    x, y, tx, ty, coordinate_system, epsg = prepare_metric_xy(
        lon, lat, lon, lat, geographic_mode="local",
    )
    assert coordinate_system == "geographic"
    assert epsg is None
    assert len(x) == len(lon)


def test_prepare_metric_xy_local_mode_rejects_regional_extent() -> None:
    # ~1100 km apart (São Paulo to Salvador-ish longitude span) -- the flat
    # tangent-plane approximation is no longer a reasonable stand-in for a
    # real projection at this scale.
    lon = np.array([-46.6, -38.5])
    lat = np.array([-23.5, -12.9])

    with pytest.raises(ValidationError, match="local_mode_extent_km_ceiling"):
        prepare_metric_xy(lon, lat, lon, lat, geographic_mode="local")


def test_prepare_metric_xy_utm_mode_unaffected_by_extent() -> None:
    # geographic_mode="utm" reprojects onto a real projection, so the local
    # flat-plane extent ceiling does not apply to it.
    lon = np.array([-46.6, -38.5])
    lat = np.array([-23.5, -12.9])

    x, y, tx, ty, coordinate_system, epsg = prepare_metric_xy(
        lon, lat, lon, lat, geographic_mode="utm",
    )
    assert coordinate_system == "geographic"
    assert epsg is not None


def test_local_mode_extent_ceiling_is_configurable() -> None:
    _reset()
    try:
        lon = np.array([-47.0, -47.001, -47.0, -47.001])
        lat = np.array([-15.0, -15.0, -15.001, -15.001])
        config["local_mode_extent_km_ceiling"] = 0.0

        with pytest.raises(ValidationError, match="local_mode_extent_km_ceiling"):
            prepare_metric_xy(lon, lat, lon, lat, geographic_mode="local")
    finally:
        _reset()


def test_reproject_geographic_xy_local_mode_rejects_regional_extent() -> None:
    lon = np.array([-46.6, -38.5])
    lat = np.array([-23.5, -12.9])

    with pytest.raises(ValidationError, match="local_mode_extent_km_ceiling"):
        reproject_geographic_xy(lon, lat, "local")


if __name__ == "__main__":
    test_prepare_metric_xy_auto_picks_local_below_ceiling()
    test_prepare_metric_xy_auto_picks_utm_past_ceiling()
    test_reproject_geographic_xy_auto_detects_geographic_mode()
    test_prepare_metric_xy_local_mode_accepts_field_scale_extent()
    test_prepare_metric_xy_local_mode_rejects_regional_extent()
    test_prepare_metric_xy_utm_mode_unaffected_by_extent()
    test_local_mode_extent_ceiling_is_configurable()
    test_reproject_geographic_xy_local_mode_rejects_regional_extent()
