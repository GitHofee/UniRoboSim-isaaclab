"""Side-effect-free compatibility probe for supported Isaac Lab installations."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from unirobosim import FrozenMap, ProbeReport, ProviderDescriptor

from .config import IsaacLabAdapterConfig

_SOURCE_PROFILE_ID = "source-isaaclab-3.0.0-beta2"
_OFFICIAL_NGC_PROFILE_ID = "ngc-isaaclab-3.0.0"

_EXPECTED = {
    "isaaclab": "6.1.17",
    "isaaclab_physx": "1.1.3",
    "isaacsim": "6.0.1.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "torchaudio": "2.11.0",
}
_OFFICIAL_NGC_EXPECTED = {
    "isaaclab": "6.1.11",
    "isaaclab_physx": "1.1.3",
    "torch": "2.10.0",
    "torchvision": "0.25.0",
}
_PYTORCH_DISTRIBUTIONS = frozenset({"torch", "torchvision", "torchaudio"})
_SOURCE_STARTUP_BUDGETS_S = (120.0, 90.0)
_OFFICIAL_NGC_STARTUP_BUDGETS_S = (300.0, 300.0)


@dataclass(frozen=True)
class _OfficialBundleEvidence:
    issues: tuple[str, ...]
    details: dict[str, object]


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_matches(name: str, actual: str | None, expected: str) -> bool:
    if actual == expected:
        return True
    return name in _PYTORCH_DISTRIBUTIONS and actual is not None and actual.split("+", 1)[0] == expected


def _package_directory(name: str) -> Path | None:
    """Locate a top-level package without importing it or any simulator parent."""

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    locations = tuple(spec.submodule_search_locations or ())
    if locations:
        return Path(locations[0])
    if spec.origin is None:
        return None
    return Path(spec.origin).parent


def _bounded_parent_with_file(start: Path, relative: str, *, depth: int = 6) -> Path | None:
    candidates = (start, *start.parents[:depth])
    return next((candidate for candidate in candidates if (candidate / relative).is_file()), None)


def _read_exact(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _inspect_official_ngc_bundle() -> _OfficialBundleEvidence:
    """Validate the immutable filesystem/API layout shipped by the NGC 3.0 image.

    The official image intentionally does not install ``isaacsim`` as a Python
    distribution.  Looking only at wheel metadata therefore gives a false
    negative.  This inspection uses module specs and bounded filesystem reads;
    it never imports Isaac Lab, Isaac Sim, Torch, Omni, or USD.
    """

    issues: list[str] = []
    details: dict[str, object] = {}
    isaaclab_package = _package_directory("isaaclab")
    isaaclab_physx_package = _package_directory("isaaclab_physx")
    isaacsim_package = _package_directory("isaacsim")
    package_locations = {
        "isaaclab": isaaclab_package,
        "isaaclab_physx": isaaclab_physx_package,
        "isaacsim": isaacsim_package,
    }
    for name, location in package_locations.items():
        details[f"{name}_module_path"] = None if location is None else str(location)
        if location is None:
            issues.append(f"official NGC bundle module {name} is unavailable")

    if isaaclab_package is None:
        isaaclab_root = None
    else:
        isaaclab_root = _bounded_parent_with_file(isaaclab_package, "VERSION")
    if isaacsim_package is None:
        isaacsim_root = None
    else:
        isaacsim_root = _bounded_parent_with_file(isaacsim_package, "docs/py/VERSION")

    isaaclab_release = None if isaaclab_root is None else _read_exact(isaaclab_root / "VERSION")
    isaacsim_release = None if isaacsim_root is None else _read_exact(isaacsim_root / "docs/py/VERSION")
    isaacsim_build = None if isaacsim_root is None else _read_exact(isaacsim_root / "VERSION")
    details.update(
        {
            "isaaclab_root": None if isaaclab_root is None else str(isaaclab_root),
            "isaaclab_release": isaaclab_release,
            "isaacsim_root": None if isaacsim_root is None else str(isaacsim_root),
            "isaacsim_release": isaacsim_release,
            "isaacsim_build": isaacsim_build,
        }
    )
    if isaaclab_release != "3.0.0":
        issues.append(f"official NGC bundle requires Isaac Lab VERSION 3.0.0; found {isaaclab_release!r}")
    if isaacsim_release != "6.0.1":
        issues.append(f"official NGC bundle requires Isaac Sim VERSION 6.0.1; found {isaacsim_release!r}")
    if isaacsim_build is None or not isaacsim_build.startswith("6.0.1-"):
        issues.append(f"official NGC bundle has an unsupported Isaac Sim build marker {isaacsim_build!r}")

    required_isaaclab_modules = (
        "app/__init__.py",
        "sim/__init__.py",
        "actuators/__init__.py",
        "assets/__init__.py",
        "assets/articulation/__init__.py",
        "assets/rigid_object/__init__.py",
        "sensors/camera/__init__.py",
        "sensors/contact_sensor/__init__.py",
        "sim/schemas/__init__.py",
    )
    missing_api_modules = (
        required_isaaclab_modules
        if isaaclab_package is None
        else tuple(relative for relative in required_isaaclab_modules if not (isaaclab_package / relative).is_file())
    )
    if isaaclab_physx_package is None:
        missing_api_modules += ("isaaclab_physx",)
    else:
        for relative in ("physics/__init__.py", "sim/schemas/__init__.py"):
            if not (isaaclab_physx_package / relative).is_file():
                missing_api_modules += (f"isaaclab_physx/{relative}",)
    details["required_api_modules"] = required_isaaclab_modules
    details["missing_api_modules"] = missing_api_modules
    if missing_api_modules:
        issues.append("official NGC bundle is missing required adapter API modules: " + ", ".join(missing_api_modules))

    debug_extensions: tuple[Path, ...] = ()
    if isaacsim_root is not None:
        debug_extensions = tuple(sorted((isaacsim_root / "extscache").glob("isaacsim.util.debug_draw-*")))
    debug_extension = debug_extensions[0] if debug_extensions else None
    details["debug_draw_extension"] = None if debug_extension is None else str(debug_extension)
    if debug_extension is None or not (debug_extension / "bin").is_dir() or not (debug_extension / "isaacsim").is_dir():
        issues.append("official NGC bundle is missing the required isaacsim.util.debug_draw extension")

    return _OfficialBundleEvidence(tuple(issues), details)


def _profile_version_issues(versions: dict[str, str | None], expected: dict[str, str]) -> list[str]:
    issues: list[str] = []
    for name, required in expected.items():
        actual = versions[name]
        if actual is None:
            issues.append(f"required distribution {name} is not installed")
        elif not _version_matches(name, actual, required):
            issues.append(f"{name}=={required} is required; found {actual}")
    return issues


def recommended_startup_budgets(
    *,
    version_reader: Callable[[str], str | None] = _distribution_version,
    official_bundle_inspector: Callable[[], _OfficialBundleEvidence] = _inspect_official_ngc_bundle,
) -> tuple[float, float]:
    """Return bounded worker budgets for the exact detected packaging profile.

    The NGC bundle performs GPU-specific RTX pipeline compilation on its first
    launch.  That work can remain inside ``AppLauncher`` without emitting worker
    protocol progress for more than 90 seconds.  Only the fully verified NGC
    fingerprint receives the larger finite allowance; every other environment
    retains the source-profile defaults and is still rejected later by ``probe``
    if it does not match a supported profile.
    """

    versions = {name: version_reader(name) for name in _EXPECTED}
    if not _profile_version_issues(versions, _OFFICIAL_NGC_EXPECTED):
        evidence = official_bundle_inspector()
        if not evidence.issues:
            return _OFFICIAL_NGC_STARTUP_BUDGETS_S
    return _SOURCE_STARTUP_BUDGETS_S


def probe_environment(
    config: IsaacLabAdapterConfig,
    descriptor: ProviderDescriptor,
    *,
    version_reader: Callable[[str], str | None] = _distribution_version,
    official_bundle_inspector: Callable[[], _OfficialBundleEvidence] = _inspect_official_ngc_bundle,
) -> ProbeReport:
    """Inspect compatibility without importing or launching any simulator module."""

    versions = {name: version_reader(name) for name in _EXPECTED}
    issues: list[str] = []
    profile: str | None = None
    profile_evidence: dict[str, object] = {}
    if sys.version_info[:2] != (3, 12):
        issues.append(f"Python 3.12 is required; found {sys.version_info.major}.{sys.version_info.minor}")
    source_issues = _profile_version_issues(versions, _EXPECTED)
    official_version_issues = _profile_version_issues(versions, _OFFICIAL_NGC_EXPECTED)
    if not source_issues:
        profile = _SOURCE_PROFILE_ID
    elif not official_version_issues:
        official_evidence = official_bundle_inspector()
        profile_evidence = official_evidence.details
        if official_evidence.issues:
            issues.extend(official_evidence.issues)
        else:
            profile = _OFFICIAL_NGC_PROFILE_ID
    else:
        issues.extend(source_issues)

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
                "runtime_profile": profile,
                "runtime_profile_evidence": profile_evidence,
                "versions": versions,
            }
        ),
    )
