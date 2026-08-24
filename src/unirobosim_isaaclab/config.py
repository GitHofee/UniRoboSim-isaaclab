"""Validated adapter launch configuration with no simulator imports."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from unirobosim import ValidationError

_DEVICE = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)$")
_ANTI_ALIASING_MODES = {"off": 0, "taa": 1, "fxaa": 2, "dlss": 3, "dlaa": 4}
_FLUID_RENDER_MODES = {"particles", "isosurface"}


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
    experience: str | None = None
    # ``None`` delegates position-drive gains to the authored asset.  Numeric
    # values retain the legacy adapter-wide override for callers that need it.
    position_stiffness: float | None = None
    position_damping: float | None = None
    velocity_damping: float = 100.0
    max_cached_scene_commands: int = 4096

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
        normalized: dict[str, float] = {}
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
            normalized[name] = gain
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
        normalized["velocity_damping"] = velocity_damping
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
        for name, value in normalized.items():
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
