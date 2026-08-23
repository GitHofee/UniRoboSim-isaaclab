"""Public lightweight entry point for the Isaac Lab adapter."""

import os

from unirobosim import ValidationError

from ._version import DISTRIBUTION_VERSION
from .config import IsaacLabAdapterConfig
from .descriptor import CAMERA_CAPABILITIES, CAPABILITIES, DESCRIPTOR, descriptor_for_config
from .probe import probe_environment
from .provider import IsaacLabProvider, IsaacLabSession
from .world import IsaacLabWorld

__version__ = DISTRIBUTION_VERSION
ISAACLAB_LAUNCH_PROFILE_ENV = "UNIROBOSIM_ISAACLAB_LAUNCH_PROFILE"


def create_provider(config: IsaacLabAdapterConfig | None = None) -> IsaacLabProvider:
    return IsaacLabProvider(config)


def create_easy_provider() -> IsaacLabProvider:
    """Create the installed EasyAPI profile selected explicitly by the environment."""

    launch_profile = os.getenv(ISAACLAB_LAUNCH_PROFILE_ENV)
    if launch_profile is None or launch_profile == "headless":
        headless = True
    elif launch_profile == "visible":
        headless = False
    else:
        raise ValidationError(
            "Isaac Lab launch profile must be unset, 'headless', or 'visible'",
            operation="isaaclab.launch_profile.resolve",
        )

    return IsaacLabProvider(
        IsaacLabAdapterConfig(
            headless=headless,
            enable_cameras=True,
            render=True,
        )
    )


__all__ = [
    "CAPABILITIES",
    "CAMERA_CAPABILITIES",
    "DESCRIPTOR",
    "DISTRIBUTION_VERSION",
    "IsaacLabAdapterConfig",
    "IsaacLabProvider",
    "IsaacLabSession",
    "IsaacLabWorld",
    "ISAACLAB_LAUNCH_PROFILE_ENV",
    "__version__",
    "create_provider",
    "create_easy_provider",
    "descriptor_for_config",
    "probe_environment",
]
