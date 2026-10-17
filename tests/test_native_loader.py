from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import apbase

PACKAGE_PARENT = Path(apbase.__file__).resolve().parents[1]

KERNEL_LOADERS = (
    ("apbase.cokriging", "load_cokriging_module"),
    ("apbase.cross_validate", "load_cross_validate_module"),
    ("apbase.cross_variogram", "load_cross_variogram_module"),
    ("apbase.filtering", "load_filter_module"),
    ("apbase.idw", "load_idw_module"),
    ("apbase.kriging", "load_kriging_module"),
    ("apbase.variogram", "load_variogram_module"),
)


@pytest.mark.parametrize(("module", "loader"), KERNEL_LOADERS)
def test_kernel_module_imported_after_native_load_still_runs(module: str, loader: str) -> None:
    script = textwrap.dedent(
        f"""
        import importlib
        import apbase._native._lib as facade

        facade.lib.load()
        native = importlib.import_module("{module}._native")
        native.{loader}()
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(PACKAGE_PARENT), env.get("PYTHONPATH"))))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
