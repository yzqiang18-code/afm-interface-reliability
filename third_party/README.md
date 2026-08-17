# Vendored metric implementations

This directory contains the upstream source files used by the
`afm-interface-reliability` project. Each implementation is pinned to a full Git
commit, and its MIT license is kept beside the source.

| Project use | Upstream | Pinned commit | Vendored entry point |
|---|---|---|---|
| iLIS | <https://github.com/flyark/AFM-LIS> | `914a01cb99aa15adcb3e047a783cf7aef2b1ac17` | `afm_lis/lis.py` |
| pDockQ2 | <https://github.com/DunbrackLab/IPSAE> | `6174cf9e71cb1bd660cc805856a18c4871a6dec3` | `ipsae/ipsae.py` |
| DockQ v2.1.3 | <https://github.com/wallnerlab/DockQ> | `75db7ab4f6b824c70d120c5f620582e164ed5479` | `dockq/src/DockQ/DockQ.py` |

The AFM-LIS and IPSAE scripts are copied verbatim. DockQ is retained as the
minimal pure-Python runtime subset used by the project adapter, including
`operations_nocy.py`. Exact file hashes are recorded in `checksums.sha256` and
checked by the repository validator.

Project-specific batching and output normalization live under
`scripts/metrics/`; they are not upstream code.
