import subprocess
import sys
from pathlib import Path

import pytest
from unirobosim import ValidationError

from unirobosim_isaaclab.config import IsaacLabAdapterConfig


def test_usd_scope_isolated():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(Path(__file__).with_name("primary_partition_usd_cases.py"))],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("value", [1, "true", None])
def test_typed_flag(value):
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(camera_primary_ray_exclusions=value)


def test_require_camera_and_exclusive_mode():
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(camera_primary_ray_exclusions=True)


@pytest.mark.parametrize("environment", [0, 1, 7])
@pytest.mark.parametrize("suffix", ["/camera", "/robot/body/head/camera", "/robot/arm_l/end/flange/camera"])
def test_actual_authored_camera_path_expands_mounted_and_unmounted(environment, suffix):
    from unirobosim_isaaclab.primary_ray_partition import concrete_camera_path

    assert concrete_camera_path("/World/env_.*" + suffix, environment) == "/World/env_" + str(environment) + suffix


def test_partition_setting_is_only_enabled_at_explicit_startup():
    from unirobosim_isaaclab.native import _launcher_kwargs

    default = _launcher_kwargs(IsaacLabAdapterConfig(enable_cameras=True, render=True))
    explicit = _launcher_kwargs(
        IsaacLabAdapterConfig(enable_cameras=True, render=True, camera_primary_ray_exclusions=True)
    )
    assert "/renderer/scenePartitioning/enabled=true" not in default["kit_args"]
    assert "--/renderer/scenePartitioning/enabled=true" in explicit["kit_args"]
