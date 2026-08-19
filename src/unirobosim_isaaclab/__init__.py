"""Public lightweight entry point for the Isaac Lab adapter."""

from .config import IsaacLabAdapterConfig
from .descriptor import CAMERA_CAPABILITIES, CAPABILITIES, DESCRIPTOR, descriptor_for_config
from .probe import probe_environment
from .provider import IsaacLabProvider, IsaacLabSession
from .world import IsaacLabWorld

__version__ = "0.6.0a0"


def create_provider(config: IsaacLabAdapterConfig | None = None) -> IsaacLabProvider:
    return IsaacLabProvider(config)


def create_easy_provider() -> IsaacLabProvider:
    """Entry-point profile where an EasyAPI camera works without native config."""

    return IsaacLabProvider(IsaacLabAdapterConfig(enable_cameras=True, render=True))


__all__ = [
    "CAPABILITIES",
    "CAMERA_CAPABILITIES",
    "DESCRIPTOR",
    "IsaacLabAdapterConfig",
    "IsaacLabProvider",
    "IsaacLabSession",
    "IsaacLabWorld",
    "__version__",
    "create_provider",
    "create_easy_provider",
    "descriptor_for_config",
    "probe_environment",
]
