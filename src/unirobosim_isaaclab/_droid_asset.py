"""Portable DROID USD discovery for acceptance and native smoke tools."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

DROID_ASSET_ENV = "UNIROBOSIM_DROID_ASSET"


def _path_from(value: object, *, source: str) -> Path:
    if isinstance(value, bytes) or not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{source} must be a string or path-like value")
    raw = os.fspath(value)
    if isinstance(raw, bytes):
        raise TypeError(f"{source} must resolve to a text path")
    if not raw.strip():
        raise ValueError(f"{source} must not be empty")
    return Path(raw).expanduser().resolve()


def resolve_droid_asset_path(
    asset_path: str | os.PathLike[str] | None = None,
    *,
    configured_asset_path: object = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a DROID USD without depending on a repository-local default.

    Resolution order is explicit ``asset_path``, configuration
    ``robot.asset_path``, then :data:`DROID_ASSET_ENV`.  The selected path must
    name an existing regular file.  This function deliberately does not
    validate a particular content digest so the native smoke tool continues to
    accept explicitly supplied DROID revisions.
    """

    if asset_path is not None:
        candidate = _path_from(asset_path, source="asset_path")
    elif configured_asset_path is not None:
        candidate = _path_from(configured_asset_path, source="robot.asset_path")
    else:
        environment = os.environ if environ is None else environ
        environment_value = environment.get(DROID_ASSET_ENV)
        if environment_value is None:
            raise FileNotFoundError(
                "DROID USD was not specified; pass asset_path, set robot.asset_path "
                f"in the acceptance config, or set {DROID_ASSET_ENV}"
            )
        candidate = _path_from(environment_value, source=DROID_ASSET_ENV)

    if not candidate.is_file():
        raise FileNotFoundError(f"DROID USD selected by portable asset discovery does not exist: {candidate}")
    return candidate


__all__ = ("DROID_ASSET_ENV", "resolve_droid_asset_path")
