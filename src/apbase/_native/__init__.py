"""Private subpackage holding the compiled apbase native extension.

Not part of the public API. Kernels import the lazy facade from its own
module, `from apbase._native._lib import lib`: once the compiled submodule
`apbase._native.lib` is imported, Python rebinds the package attribute
`apbase._native.lib` to the extension module itself, so importing `lib` from
this package afterwards would return the extension instead of the facade.
"""

from __future__ import annotations

from apbase._native._lib import lib

__all__ = ["lib"]
