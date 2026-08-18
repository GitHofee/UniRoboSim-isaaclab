"""Side-effect-free compatibility probe based on package metadata and nvidia-smi."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from collections.abc import Callable

from unirobosim import FrozenMap, ProbeReport, ProviderDescriptor

from .config import IsaacLabAdapterConfig

_EXPECTED = {
    "isaaclab": "6.1.17",
    "isaaclab_physx": "1.1.3",
    "isaacsim": "6.0.1.0",
    "torch": "2.10.0",
}


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_environment(
    config: IsaacLabAdapterConfig,
    descriptor: ProviderDescriptor,
    *,
    version_reader: Callable[[str], str | None] = _distribution_version,
) -> ProbeReport:
    """Inspect compatibility without importing or launching any simulator module."""

    versions = {name: version_reader(name) for name in _EXPECTED}
    issues: list[str] = []
    if sys.version_info[:2] != (3, 12):
        issues.append(f"Python 3.12 is required; found {sys.version_info.major}.{sys.version_info.minor}")
    for name, expected in _EXPECTED.items():
        actual = versions[name]
        if actual is None:
            issues.append(f"required distribution {name} is not installed")
        elif actual != expected and not (name == "torch" and actual.split("+", 1)[0] == expected):
            issues.append(f"{name}=={expected} is required; found {actual}")

    gpu = None
    if config.device.startswith("cuda"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            issues.append(f"nvidia-smi failed: {exc}")
        else:
            gpu = result.stdout.strip() or None
            if result.returncode != 0 or gpu is None:
                message = result.stderr.strip() or f"exit status {result.returncode}"
                issues.append(f"CUDA device probe failed: {message}")

    return ProbeReport(
        descriptor=descriptor,
        available=not issues,
        reason=None if not issues else "; ".join(issues),
        details=FrozenMap(
            {
                "device": config.device,
                "gpu": gpu,
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "versions": versions,
            }
        ),
    )
