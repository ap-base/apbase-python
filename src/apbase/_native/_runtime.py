from __future__ import annotations

import os

_NATIVE_THREAD_DEFAULTS = {
    "OPENBLAS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}


def configure_native_threading() -> None:
    """Use one BLAS thread by default unless the caller configured native libs."""
    for name, value in _NATIVE_THREAD_DEFAULTS.items():
        os.environ.setdefault(name, value)
