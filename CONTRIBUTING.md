# Contributing to APbase (Python)

Thanks for your interest in improving APbase. Bug reports, documentation fixes,
tests and code changes are all welcome.

- Questions and ideas: [Discussions](https://github.com/ap-base/apbase-python/discussions).
- Bugs and feature requests: [Issues](https://github.com/ap-base/apbase-python/issues)
  (use the templates).
- Security problems: never in a public issue; see [SECURITY.md](SECURITY.md).

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

The repository carries the compiled native module only for Linux x86_64 with
CPython 3.14 (`src/apbase/_native/lib.cpython-314-x86_64-linux-gnu.so`), so
development from source needs that platform. On other platforms, install the
wheel from PyPI to use the package.

```bash
sudo apt install libgfortran5 libgomp1 libopenblas0-pthread   # Ubuntu/Debian runtime
git clone https://github.com/ap-base/apbase-python.git
cd apbase-python
python3.14 -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
```

The numerical kernels are prebuilt under `src/apbase/_native/`. Pull requests
do not modify these files; if you find a problem in a kernel, open an issue
with a minimal example.

## Checks

Run the tests as CI does before opening a pull request:

```bash
python -c "from apbase._native._lib import lib; lib.load()"   # native module loads
python -m pytest -q
```

Ruff and mypy are configured in `pyproject.toml`; keep new code clean:

```bash
ruff check src tests
mypy
```

## Code conventions

- The top-level namespace stays small (`Map`, `create_map`, `config`); building
  blocks live in their submodules (`apbase.idw`, `apbase.kriging`, ...).
- Optional parameters are keyword-only, with NumPy-style docstrings.
- Errors derive from `apbase.common.ApbaseError`.
- Keep public names, defaults and error messages stable; a change to any of
  them goes in [CHANGELOG.md](CHANGELOG.md).
- Add tests next to the behavior you change (`tests/test_<module>.py`).

## Pull requests

1. Fork the repository (or create a branch, if you have access) from `main`.
2. Keep each pull request focused on one change, with tests and docs.
3. Add a line to the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md).
4. Open the pull request against `main`. It needs a passing CI and an approval
   from a code owner before merging.

## Releases (maintainers)

1. Bump `version` in `pyproject.toml` and the version badge in the README.
2. Update `version` and `date-released` in `CITATION.cff`.
3. Move the `Unreleased` entries of `CHANGELOG.md` under the new version.
4. Run "Build wheels" with the version and the kernels release tag; it creates
   the GitHub release with the wheels.
5. Run "Publish to PyPI" with the same tag.
