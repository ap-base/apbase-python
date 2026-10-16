"""Download the native module built for this interpreter into the package tree.

    KERNELS_TOKEN=... python .github/scripts/fetch_native_module.py <kernels release tag> <src/apbase/_native dir>

The kernels release carries one `lib.<tag>.<ext>` per platform and Python
version (tag from EXT_SUFFIX, e.g. `cpython-312-x86_64-linux-gnu`). Every
other `lib.*` module in the package tree is removed first: package-data ships
all of them, so a wheel would otherwise carry the Linux module committed for
development next to its own.
"""

from __future__ import annotations

import json
import os
import sys
import sysconfig
import urllib.request
from pathlib import Path

REPOSITORY = "ap-base/apbase-fortran-kernels"


def request(url: str, accept: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {os.environ['KERNELS_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )


def main() -> int:
    release_tag, package_dir = sys.argv[1], Path(sys.argv[2])
    name = f"lib{sysconfig.get_config_var('EXT_SUFFIX') or '.so'}"

    url = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{release_tag}"
    with urllib.request.urlopen(request(url, "application/vnd.github+json")) as response:
        assets = {asset["name"]: asset["url"] for asset in json.load(response)["assets"]}
    if name not in assets:
        raise SystemExit(f"kernels release {release_tag} has no {name}; available: {sorted(assets)}")

    for pattern in ("lib.*.so", "lib.*.pyd"):
        for old in package_dir.glob(pattern):
            old.unlink()
            print(f"removed {old}")
    with urllib.request.urlopen(request(assets[name], "application/octet-stream")) as response:
        (package_dir / name).write_bytes(response.read())
    print(f"using {name} from kernels release {release_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
