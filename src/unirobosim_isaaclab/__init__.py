"""Public lightweight entry point for the Isaac Lab adapter."""

import os

from unirobosim import ValidationError

from ._version import DISTRIBUTION_VERSION
from .config import IsaacLabAdapterConfig
from .descriptor import CAMERA_CAPABILITIES, CAPABILITIES, DESCRIPTOR, descriptor_for_config
from .probe import probe_environment, recommended_startup_budgets
from .provider import IsaacLabProvider, IsaacLabSession
from .world import IsaacLabWorld

__version__ = DISTRIBUTION_VERSION
ISAACLAB_LAUNCH_PROFILE_ENV = "UNIROBOSIM_ISAACLAB_LAUNCH_PROFILE"
ISAACLAB_FLUID_RENDER_MODE_ENV = "UNIROBOSIM_ISAACLAB_FLUID_RENDER_MODE"
ISAACLAB_FLUID_SURFACE_COLOR_ENV = "UNIROBOSIM_ISAACLAB_FLUID_SURFACE_COLOR"
ISAACLAB_FLUID_SURFACE_DISTANCE_SCALE_ENV = "UNIROBOSIM_ISAACLAB_FLUID_SURFACE_DISTANCE_SCALE"
ISAACLAB_FLUID_SURFACE_SMOOTHING_SCALE_ENV = "UNIROBOSIM_ISAACLAB_FLUID_SURFACE_SMOOTHING_SCALE"
ISAACLAB_FLUID_DAMPING_ENV = "UNIROBOSIM_ISAACLAB_FLUID_DAMPING"
ISAACLAB_FLUID_COHESION_ENV = "UNIROBOSIM_ISAACLAB_FLUID_COHESION"
ISAACLAB_FLUID_ADHESION_ENV = "UNIROBOSIM_ISAACLAB_FLUID_ADHESION"
ISAACLAB_FLUID_FRICTION_ENV = "UNIROBOSIM_ISAACLAB_FLUID_FRICTION"
ISAACLAB_FLUID_CFL_ENV = "UNIROBOSIM_ISAACLAB_FLUID_CFL"
ISAACLAB_FLUID_MAX_VELOCITY_ENV = "UNIROBOSIM_ISAACLAB_FLUID_MAX_VELOCITY"
ISAACLAB_FLUID_MAX_DEPENETRATION_VELOCITY_ENV = "UNIROBOSIM_ISAACLAB_FLUID_MAX_DEPENETRATION_VELOCITY"


def _positive_environment_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValidationError(
            f"{name} must be finite and positive",
            operation="isaaclab.fluid_surface.resolve",
        ) from None


def _optional_positive_environment_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value <= 0.0:
        raise ValidationError(f"{name} must be finite and positive", operation="isaaclab.fluid.resolve")
    return value


def _fluid_surface_color_from_environment() -> tuple[float, float, float] | None:
    raw = os.environ.get(ISAACLAB_FLUID_SURFACE_COLOR_ENV)
    if raw is None:
        return None
    try:
        values = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError:
        values = ()
    if len(values) != 3:
        raise ValidationError(
            f"{ISAACLAB_FLUID_SURFACE_COLOR_ENV} must be three comma-separated values in [0, 1]",
            operation="isaaclab.fluid_surface_color.resolve",
        )
    return values  # IsaacLabAdapterConfig performs finite/range validation.


def create_provider(config: IsaacLabAdapterConfig | None = None) -> IsaacLabProvider:
    if config is None:
        worker_startup_hard_timeout_s, worker_kit_launch_idle_timeout_s = recommended_startup_budgets()
        config = IsaacLabAdapterConfig(
            worker_startup_hard_timeout_s=worker_startup_hard_timeout_s,
            worker_kit_launch_idle_timeout_s=worker_kit_launch_idle_timeout_s,
        )
    return IsaacLabProvider(config)


def create_easy_provider(*, launch_profile: str | None = None) -> IsaacLabProvider:
    """Create the installed EasyAPI profile selected by an argument or the environment.

    An explicit profile is authoritative and deliberately avoids consulting process
    state. Omitting it retains the original EasyAPI environment-variable contract.
    """

    resolved_profile = os.getenv(ISAACLAB_LAUNCH_PROFILE_ENV) if launch_profile is None else launch_profile
    if resolved_profile is not None and not isinstance(resolved_profile, str):
        raise ValidationError(
            "Isaac Lab launch profile must be unset, 'headless', 'headless-physics', or 'visible'",
            operation="isaaclab.launch_profile.resolve",
        ) from None
    if resolved_profile is None or resolved_profile == "headless":
        headless = True
        enable_cameras = True
        render = True
        render_on_step = False
        max_render_hz = None
    elif resolved_profile == "headless-physics":
        headless = True
        enable_cameras = False
        render = False
        render_on_step = False
        max_render_hz = None
    elif resolved_profile == "visible":
        headless = False
        enable_cameras = True
        render = True
        render_on_step = True
        max_render_hz = 60.0
    else:
        raise ValidationError(
            "Isaac Lab launch profile must be unset, 'headless', 'headless-physics', or 'visible'",
            operation="isaaclab.launch_profile.resolve",
        )

    worker_startup_hard_timeout_s, worker_kit_launch_idle_timeout_s = recommended_startup_budgets()
    fluid_render_mode = os.environ.get(ISAACLAB_FLUID_RENDER_MODE_ENV, "particles")
    return IsaacLabProvider(
        IsaacLabAdapterConfig(
            headless=headless,
            enable_cameras=enable_cameras,
            render=render,
            render_on_step=render_on_step,
            max_render_hz=max_render_hz,
            fluid_render_mode=fluid_render_mode,
            fluid_surface_color_rgb=_fluid_surface_color_from_environment(),
            fluid_surface_distance_scale=_positive_environment_float(
                ISAACLAB_FLUID_SURFACE_DISTANCE_SCALE_ENV, 2.5
            ),
            fluid_surface_smoothing_scale=_positive_environment_float(
                ISAACLAB_FLUID_SURFACE_SMOOTHING_SCALE_ENV, 3.0
            ),
            fluid_damping=_positive_environment_float(ISAACLAB_FLUID_DAMPING_ENV, 0.0),
            fluid_cohesion=_positive_environment_float(ISAACLAB_FLUID_COHESION_ENV, 0.0),
            fluid_adhesion=_positive_environment_float(ISAACLAB_FLUID_ADHESION_ENV, 0.0),
            fluid_friction=_positive_environment_float(ISAACLAB_FLUID_FRICTION_ENV, 0.2),
            fluid_cfl_coefficient=_positive_environment_float(ISAACLAB_FLUID_CFL_ENV, 1.0),
            fluid_max_velocity_m_s=_optional_positive_environment_float(ISAACLAB_FLUID_MAX_VELOCITY_ENV),
            fluid_max_depenetration_velocity_m_s=_optional_positive_environment_float(
                ISAACLAB_FLUID_MAX_DEPENETRATION_VELOCITY_ENV
            ),
            worker_startup_hard_timeout_s=worker_startup_hard_timeout_s,
            worker_kit_launch_idle_timeout_s=worker_kit_launch_idle_timeout_s,
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
    "ISAACLAB_FLUID_RENDER_MODE_ENV",
    "ISAACLAB_FLUID_SURFACE_COLOR_ENV",
    "ISAACLAB_FLUID_SURFACE_DISTANCE_SCALE_ENV",
    "ISAACLAB_FLUID_SURFACE_SMOOTHING_SCALE_ENV",
    "ISAACLAB_FLUID_DAMPING_ENV",
    "ISAACLAB_FLUID_COHESION_ENV",
    "ISAACLAB_FLUID_ADHESION_ENV",
    "ISAACLAB_FLUID_FRICTION_ENV",
    "ISAACLAB_FLUID_CFL_ENV",
    "ISAACLAB_FLUID_MAX_VELOCITY_ENV",
    "ISAACLAB_FLUID_MAX_DEPENETRATION_VELOCITY_ENV",
    "__version__",
    "create_provider",
    "create_easy_provider",
    "descriptor_for_config",
    "probe_environment",
    "recommended_startup_budgets",
]
