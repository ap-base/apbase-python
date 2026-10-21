# Security Policy

## Supported versions

Security fixes go into the latest release on PyPI. Older versions are not
patched; upgrade to the latest `0.1.x`.

| Version | Supported |
|---------|-----------|
| 0.1.x (latest) | Yes |
| Older releases | No |

## Reporting a vulnerability

Please do not open a public issue for security problems. Report them
privately through either channel:

- GitHub: [Report a vulnerability](https://github.com/ap-base/apbase-python/security/advisories/new)
  (Security tab of this repository).
- Email: [leonardofm@alumni.usp.br](mailto:leonardofm@alumni.usp.br).

Include the affected version, your platform (OS, architecture, Python
version), and the steps or a minimal script that reproduces the problem.

You will get an acknowledgement within 5 business days. Once the issue is
confirmed, a fix is prepared in a private advisory and released as a new
version; you are credited in the advisory unless you prefer otherwise.

## Scope

This policy covers the `apbase` package and the native module it ships under
`src/apbase/_native/`. Problems in NumPy, pyproj, Shapely, OpenBLAS or the
Fortran runtime should be reported to those projects; tell us too if `apbase`
needs a version bump because of them.
