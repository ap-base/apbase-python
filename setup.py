from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Mark wheels as platform-specific because native libraries are bundled."""

    def has_ext_modules(self) -> bool:
        return True

    def is_pure(self) -> bool:
        return False


class build_py(_build_py):
    """Copy Python packages from a clean build tree without native compilation."""

    def run(self) -> None:
        package_build_dir = Path(self.build_lib) / "apbase"
        if package_build_dir.exists():
            shutil.rmtree(package_build_dir)
        super().run()


setup(
    cmdclass={"build_py": build_py},
    distclass=BinaryDistribution,
)
