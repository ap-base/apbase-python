"""Private subpackage holding the compiled apbase native extension.

Not part of the public API. Kernels import the lazy facade from here:
`from apbase._native import lib`.
"""

from __future__ import annotations

from apbase._native._lib import lib

__all__ = ["lib"]
