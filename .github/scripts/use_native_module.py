"""Put the native module built for this interpreter into the package tree.

    python .github/scripts/use_native_module.py <dist/python dir> <src/apbase/_native dir>

tools/build_python.py writes `<dist>/<tag>/lib.<tag>.<ext>` (tag from this
interpreter's EXT_SUFFIX, e.g. `cpython-312-x86_64-linux-gnu`). Every other
`lib.*` module in the package tree is removed first: package-data ships all
of them, so a wheel would otherwise carry the Linux module committed for
development next to its own.
"""

from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    dist_dir, package_dir = (Path(arg) for arg in sys.argv[1:3])
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    tag = suffix.lstrip(".").rsplit(".", 1)[0]
    module = dist_dir / tag / f"lib{suffix}"
    if not module.is_file():
        raise SystemExit(f"native module not built for this interpreter: {module}")

    for pattern in ("lib.*.so", "lib.*.pyd"):
        for old in package_dir.glob(pattern):
            old.unlink()
            print(f"removed {old}")
    shutil.copy2(module, package_dir / module.name)
    print(f"using {module}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
