"""Public lightweight entry point for the Isaac Lab adapter."""

from .config import IsaacLabAdapterConfig
from .descriptor import CAPABILITIES, DESCRIPTOR
from .probe import probe_environment
from .provider import IsaacLabProvider, IsaacLabSession
from .world import IsaacLabWorld

__version__ = "0.1.0a0"


def create_provider(config: IsaacLabAdapterConfig | None = None) -> IsaacLabProvider:
    return IsaacLabProvider(config)


__all__ = [
    "CAPABILITIES",
    "DESCRIPTOR",
    "IsaacLabAdapterConfig",
    "IsaacLabProvider",
    "IsaacLabSession",
    "IsaacLabWorld",
    "__version__",
    "create_provider",
    "probe_environment",
]
