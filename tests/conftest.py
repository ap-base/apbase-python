from __future__ import annotations

import ctypes
import gc

import pytest


def _trim_heap() -> None:
    """Return freed glibc arenas to the OS.

    gc.collect() only drops Python references; the C allocator keeps the
    underlying heap for reuse within the process. Across ~50+ tests in one
    pytest process, that heap only grows, so RSS peaks at whichever test
    runs last regardless of its own size. malloc_trim(0) forces glibc to
    give idle arenas back, keeping RSS close to each test's own footprint.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # non-glibc platform (e.g. macOS, musl): no-op


@pytest.fixture(autouse=True)
def _release_memory_after_test():
    yield
    gc.collect()
    _trim_heap()
