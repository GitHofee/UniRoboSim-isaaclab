"""Validated adapter launch configuration with no simulator imports."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from unirobosim import ValidationError

_DEVICE = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)$")
_ANTI_ALIASING_MODES = {"off": 0, "taa": 1, "fxaa": 2, "dlss": 3, "dlaa": 4}
_FLUID_RENDER_MODES = {"particles", "isosurface"}
_MAX_STARTUP_BUDGET_SECONDS = 300.0


@dataclass(frozen=True)
class IsaacLabAdapterConfig:
    """Small launch surface; world physics remains in :class:`WorldSpec`."""

    headless: bool = True
    device: str = "cuda:0"
    environment_spacing_m: float = 4.0
    enable_cameras: bool = False
    render: bool = False
    anti_aliasing: str = "fxaa"
    texture_streaming: bool = False
    render_on_step: bool = True
    max_render_hz: float | None = None
    fluid_render_mode: str = "particles"
    fluid_surface_color_rgb: tuple[float, float, float] | None = None
    fluid_surface_distance_scale: float = 2.5
    fluid_surface_smoothing_scale: float = 3.0
    fluid_damping: float = 0.0
    fluid_cohesion: float = 0.0
    fluid_adhesion: float = 0.0
    fluid_friction: float = 0.2
    fluid_cfl_coefficient: float = 1.0
    fluid_max_velocity_m_s: float | None = None
    fluid_max_depenetration_velocity_m_s: float | None = None
    experience: str | None = None
    # ``None`` delegates position-drive gains to the authored asset.  Numeric
    # values retain the legacy adapter-wide override for callers that need it.
    position_stiffness: float | None = None
    position_damping: float | None = None
    velocity_damping: float = 100.0
    max_cached_scene_commands: int = 4096
    # Cold containers can spend tens of seconds compiling/loading Kit state.
    # These budgets remain finite and affect only a worker that has not become
    # ready yet, so a warm worker never waits for the unused allowance.
    worker_startup_hard_timeout_s: float = 120.0
    worker_kit_launch_idle_timeout_s: float = 90.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.headless, bool)
            or not isinstance(self.enable_cameras, bool)
            or not isinstance(self.render, bool)
            or not isinstance(self.texture_streaming, bool)
            or not isinstance(self.render_on_step, bool)
        ):
            raise ValidationError("launch flags must be boolean", operation="isaaclab.config.validate")
        if not isinstance(self.device, str) or not _DEVICE.fullmatch(self.device):
            raise ValidationError(
                "device must be cpu, cuda, or cuda:<index>",
                operation="isaaclab.config.validate",
                details={"device": self.device},
            )
        if not isinstance(self.anti_aliasing, str) or self.anti_aliasing.lower() not in _ANTI_ALIASING_MODES:
            raise ValidationError(
                "anti_aliasing must be off, taa, fxaa, dlss, or dlaa",
                operation="isaaclab.config.validate",
                details={"anti_aliasing": self.anti_aliasing},
            )
        object.__setattr__(self, "anti_aliasing", self.anti_aliasing.lower())
        if not isinstance(self.fluid_render_mode, str) or self.fluid_render_mode.lower() not in _FLUID_RENDER_MODES:
            raise ValidationError(
                "fluid_render_mode must be particles or isosurface",
                operation="isaaclab.config.validate",
                details={"fluid_render_mode": self.fluid_render_mode},
            )
        object.__setattr__(self, "fluid_render_mode", self.fluid_render_mode.lower())
        if self.fluid_surface_color_rgb is not None:
            try:
                color = tuple(float(value) for value in self.fluid_surface_color_rgb)
            except (TypeError, ValueError):
                raise ValidationError(
                    "fluid_surface_color_rgb must contain three finite values in [0, 1]",
                    operation="isaaclab.config.validate",
                ) from None
            if len(color) != 3 or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in color):
                raise ValidationError(
                    "fluid_surface_color_rgb must contain three finite values in [0, 1]",
                    operation="isaaclab.config.validate",
                )
            object.__setattr__(self, "fluid_surface_color_rgb", color)
        for name in ("fluid_surface_distance_scale", "fluid_surface_smoothing_scale"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValidationError(f"{name} must be finite and positive", operation="isaaclab.config.validate")
            try:
                normalized = float(value)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"{name} must be finite and positive", operation="isaaclab.config.validate"
                ) from None
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValidationError(f"{name} must be finite and positive", operation="isaaclab.config.validate")
            object.__setattr__(self, name, normalized)
        for name in ("fluid_damping", "fluid_cohesion", "fluid_adhesion", "fluid_friction", "fluid_cfl_coefficient"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValidationError(f"{name} must be finite and non-negative", operation="isaaclab.config.validate")
            try:
                normalized = float(value)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"{name} must be finite and non-negative", operation="isaaclab.config.validate"
                ) from None
            if not math.isfinite(normalized) or normalized < 0.0:
                raise ValidationError(f"{name} must be finite and non-negative", operation="isaaclab.config.validate")
            object.__setattr__(self, name, normalized)
        for name in ("fluid_max_velocity_m_s", "fluid_max_depenetration_velocity_m_s"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool):
                raise ValidationError(f"{name} must be finite and positive", operation="isaaclab.config.validate")
            try:
                normalized = float(value)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"{name} must be finite and positive", operation="isaaclab.config.validate"
                ) from None
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValidationError(f"{name} must be finite and positive", operation="isaaclab.config.validate")
            object.__setattr__(self, name, normalized)
        if isinstance(self.environment_spacing_m, bool):
            raise ValidationError("environment spacing must be numeric", operation="isaaclab.config.validate")
        try:
            spacing = float(self.environment_spacing_m)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "environment spacing must be numeric", operation="isaaclab.config.validate", cause=exc
            ) from exc
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValidationError(
                "environment spacing must be positive and finite", operation="isaaclab.config.validate"
            )
        if self.experience is not None and (not isinstance(self.experience, str) or not self.experience.strip()):
            raise ValidationError(
                "experience must be a non-empty path when provided", operation="isaaclab.config.validate"
            )
        if self.render and self.headless and not self.enable_cameras:
            raise ValidationError("headless rendering requires enable_cameras", operation="isaaclab.config.validate")
        optional_gains = {
            "position_stiffness": self.position_stiffness,
            "position_damping": self.position_damping,
        }
        normalized_gains: dict[str, float] = {}
        for name, value in optional_gains.items():
            if value is None:
                continue
            if isinstance(value, bool):
                raise ValidationError(f"{name} must be numeric", operation="isaaclab.config.validate")
            try:
                gain = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"{name} must be numeric", operation="isaaclab.config.validate", cause=exc
                ) from exc
            if not math.isfinite(gain) or gain < 0.0:
                raise ValidationError(f"{name} must be non-negative and finite", operation="isaaclab.config.validate")
            normalized_gains[name] = gain
        if isinstance(self.velocity_damping, bool):
            raise ValidationError("velocity_damping must be numeric", operation="isaaclab.config.validate")
        try:
            velocity_damping = float(self.velocity_damping)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "velocity_damping must be numeric", operation="isaaclab.config.validate", cause=exc
            ) from exc
        if not math.isfinite(velocity_damping) or velocity_damping < 0.0:
            raise ValidationError(
                "velocity_damping must be non-negative and finite", operation="isaaclab.config.validate"
            )
        normalized_gains["velocity_damping"] = velocity_damping
        object.__setattr__(self, "environment_spacing_m", spacing)
        if self.max_render_hz is not None:
            if isinstance(self.max_render_hz, bool):
                raise ValidationError(
                    "max render rate must be numeric",
                    operation="isaaclab.config.validate",
                )
            try:
                max_render_hz = float(self.max_render_hz)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "max render rate must be numeric",
                    operation="isaaclab.config.validate",
                    cause=exc,
                ) from exc
            if not math.isfinite(max_render_hz) or max_render_hz <= 0.0:
                raise ValidationError(
                    "max render rate must be positive and finite",
                    operation="isaaclab.config.validate",
                )
            object.__setattr__(self, "max_render_hz", max_render_hz)
        for name, value in normalized_gains.items():
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.max_cached_scene_commands, int)
            or isinstance(self.max_cached_scene_commands, bool)
            or not 16 <= self.max_cached_scene_commands <= 1_000_000
        ):
            raise ValidationError(
                "max_cached_scene_commands must be an integer in [16, 1000000]",
                operation="isaaclab.config.validate",
            )
        startup_budgets = {
            "worker_startup_hard_timeout_s": self.worker_startup_hard_timeout_s,
            "worker_kit_launch_idle_timeout_s": self.worker_kit_launch_idle_timeout_s,
        }
        normalized_startup_budgets: dict[str, float] = {}
        for name, value in startup_budgets.items():
            if type(value) not in (int, float):
                raise ValidationError(f"{name} must be numeric", operation="isaaclab.config.validate")
            seconds = float(value)
            if not math.isfinite(seconds) or not 0.0 < seconds <= _MAX_STARTUP_BUDGET_SECONDS:
                raise ValidationError(
                    f"{name} must be positive, finite, and at most {_MAX_STARTUP_BUDGET_SECONDS:g}",
                    operation="isaaclab.config.validate",
                )
            normalized_startup_budgets[name] = seconds
        if (
            normalized_startup_budgets["worker_kit_launch_idle_timeout_s"]
            > normalized_startup_budgets["worker_startup_hard_timeout_s"]
        ):
            raise ValidationError(
                "worker_kit_launch_idle_timeout_s must not exceed worker_startup_hard_timeout_s",
                operation="isaaclab.config.validate",
            )
        for name, value in normalized_startup_budgets.items():
            object.__setattr__(self, name, value)
