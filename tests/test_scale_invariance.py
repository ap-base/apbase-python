from __future__ import annotations

import numpy as np
import pytest

from apbase import create_map
from apbase.cokriging import CoKriging
from apbase.cross_validate import cross_validate, cross_validate_cokriging
from apbase.cross_variogram import CrossVariogram
from apbase.kriging import Kriging
from apbase.variogram import Variogram

SCALES = (1e-9, 1e-3, 1.0, 1e4, 1e9)
SOLVER_SCALES = (1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9)


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 1000, 300), rng.uniform(0, 1000, 300)
    z = np.sin(x / 150) * np.cos(y / 150) * 10 + rng.normal(0, 1.0, 300)
    u = 2 * np.sin(x / 150) * np.cos(y / 150) * 10 + rng.normal(0, 3.0, 300)
    return x, y, z, u


def _assert_same_model(values: list[np.ndarray], scales: tuple[float, ...]) -> None:
    reference = values[0] / scales[0] ** 2
    for scaled, c in zip(values[1:], scales[1:], strict=True):
        np.testing.assert_allclose(scaled / c**2, reference, rtol=1e-6, atol=1e-9 * np.abs(reference).max())


@pytest.mark.parametrize("fit", ["variogram", "cross_variogram"])
def test_variogram_fits_do_not_depend_on_the_unit_of_the_values(fit: str) -> None:
    x, y, z, u = _data()
    models = []
    for c in SCALES:
        if fit == "variogram":
            models.append(Variogram().fit(x, y, c * z).model_values)
        else:
            models.append(CrossVariogram().fit(x, y, c * z, x, y, c * u).model_values)
    assert len({int(m[0]) for m in models}) == 1
    np.testing.assert_allclose([m[3] for m in models], models[0][3], rtol=1e-9)
    _assert_same_model([m[1:3] for m in models], SCALES)


@pytest.mark.parametrize("method", ["icm", "lmc"])
def test_coregionalization_fits_do_not_depend_on_the_unit_of_the_values(method: str) -> None:
    x, y, z, u = _data()
    params = [CoKriging(method=method).fit(x, y, c * z, {"u": (x, y, c * u)}).model_params for c in SCALES]
    assert len({p["model_id"] for p in params}) == 1
    if method == "icm":
        np.testing.assert_allclose([p["range"] for p in params], params[0]["range"], rtol=1e-9)
        matrices = [np.concatenate([p["nugget_matrix"].ravel(), p["sill_matrix"].ravel()]) for p in params]
        _assert_same_model(matrices, SCALES)
    else:
        for p in params:
            np.testing.assert_allclose(p["structure_ranges"], params[0]["structure_ranges"], rtol=1e-9)
        _assert_same_model([p["coefficient_matrices"].ravel() for p in params], SCALES)


def _predict(solver: str, c: float) -> dict[str, np.ndarray]:
    x, y, z, u = _data()
    secondaries = {"u": (x, y, c * u)}
    targets = np.column_stack([np.linspace(50, 950, 30), np.linspace(80, 920, 30)])
    if solver == "kriging":
        return {solver: Kriging(max_neighbors=40, min_neighbors=3).fit(x, y, c * z).interpolate(targets)}
    if solver == "cross_validate":
        return {solver: cross_validate(x, y, c * z, max_points=100, seed=0).kriging.predicted}
    if solver == "cross_validate_cokriging":
        result = cross_validate_cokriging(x, y, c * z, secondaries, max_points=100, seed=0)
        return {method: values.predicted for method, values in result.by_method.items()}
    return {solver: CoKriging(method=solver).fit(x, y, c * z, secondaries).interpolate(targets)}


@pytest.mark.parametrize(
    "solver", ["kriging", "cross_validate", "collocated", "icm", "lmc", "cross_validate_cokriging"]
)
def test_solvers_do_not_depend_on_the_unit_of_the_values(solver: str) -> None:
    reference = _predict(solver, 1.0)
    assert all(np.isfinite(values).all() for values in reference.values())
    for c in SOLVER_SCALES:
        scaled = _predict(solver, c)
        assert scaled.keys() == reference.keys(), c
        for method, values in scaled.items():
            assert np.isfinite(values).all(), (method, c)
            np.testing.assert_allclose(values / c, reference[method], rtol=1e-6, atol=1e-9)


def test_create_map_does_not_depend_on_the_unit_of_the_values() -> None:
    x, y, z, _ = _data()
    reference = create_map(x, y, z, resolution=50.0, seed=0)
    for c in SOLVER_SCALES:
        result = create_map(x, y, c * z, resolution=50.0, seed=0)
        assert result.method == reference.method, c
        np.testing.assert_allclose(result.z / c, reference.z, rtol=1e-6, atol=1e-9)
