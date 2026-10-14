"""Data-location helpers.

Market data are never shipped with the repository.  By default every data
path resolves under ``<project>/data``; set ``STABLEALPHA_DATA_HOME`` to keep
the (large) local data store elsewhere.  Config files therefore use
repository-relative paths such as ``data/baostock_hs300/panel``.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_HOME_ENV = "STABLEALPHA_DATA_HOME"


def data_home(project: Path | str) -> Path:
    """Return the root of the local data store for ``project``."""

    override = os.environ.get(DATA_HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path(project) / "data").resolve()


def resolve_data_path(project: Path | str, value: str | os.PathLike[str]) -> Path:
    """Resolve a config path.

    Absolute paths are kept.  Relative paths beginning with ``data/`` are
    re-rooted at :func:`data_home`; other relative paths resolve against the
    project directory.
    """

    raw = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if raw.is_absolute():
        return raw.resolve()
    parts = raw.parts
    if parts and parts[0] == "data":
        return data_home(project).joinpath(*parts[1:]).resolve()
    return (Path(project) / raw).resolve()
