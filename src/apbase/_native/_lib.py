"""Lazy facade for the compiled native extension `apbase._native.lib`.

A single .so (built by tools/build_native_extensions.py) backs every
kernel (filtering, variogram, kriging, idw, cross_validate). This module
only imports and caches it; symbol validation is handled per kernel in
each package's `_native.py`.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from apbase._native._runtime import configure_native_threading
from apbase.common.exceptions import NativeExtensionError


class _LazyLib:
    """Lazy loader for the single compiled apbase._native.lib extension."""

    __slots__ = ("_module",)

    def __init__(self) -> None:
        self._module: ModuleType | None = None

    def load(self) -> ModuleType:
        """Load and cache the compiled apbase._native.lib extension module."""
        if self._module is None:
            configure_native_threading()
            try:
                self._module = import_module("apbase._native.lib")
            except Exception as exc:
                raise NativeExtensionError(f"apbase native extension is unavailable: {exc}") from exc
        return self._module

    def __getattr__(self, name: str):
        return getattr(self.load(), name)


lib = _LazyLib()

__all__ = ["lib"]
