"""Final factory/YAML contract cases; enable after v4 GPU acceptance and cleanup."""

import pytest
from unirobosim import ValidationError

from unirobosim_isaaclab import create_easy_provider


@pytest.mark.parametrize("value", [True, False])
def test_backend_option_strict_boolean(value):
    provider = create_easy_provider(backend_options={"camera_primary_ray_exclusions": value})
    assert provider._config.camera_primary_ray_exclusions is value


@pytest.mark.parametrize("value", [1, 0, "true", None, [], {}])
def test_backend_option_rejects_nonbool(value):
    with pytest.raises(ValidationError):
        create_easy_provider(backend_options={"camera_primary_ray_exclusions": value})


@pytest.mark.parametrize("value", [[], True, "true", {"unknown": True}])
def test_backend_option_rejects_unknown_or_nonmapping(value):
    with pytest.raises(ValidationError):
        create_easy_provider(backend_options=value)


@pytest.mark.parametrize("value", [True, False])
def test_direct_arg_agreement_allowed(value):
    assert (
        create_easy_provider(
            camera_primary_ray_exclusions=value, backend_options={"camera_primary_ray_exclusions": value}
        )._config.camera_primary_ray_exclusions
        is value
    )
    with pytest.raises(ValidationError):
        create_easy_provider(
            camera_primary_ray_exclusions=not value, backend_options={"camera_primary_ray_exclusions": value}
        )


def test_explicit_service_environment_configuration(monkeypatch):
    monkeypatch.delenv("UNIROBOSIM_ISAACLAB_CAMERA_PRIMARY_RAY_EXCLUSIONS", raising=False)
    assert create_easy_provider()._config.camera_primary_ray_exclusions is False
    monkeypatch.setenv("UNIROBOSIM_ISAACLAB_CAMERA_PRIMARY_RAY_EXCLUSIONS", "true")
    assert create_easy_provider()._config.camera_primary_ray_exclusions is True
    assert create_easy_provider(camera_primary_ray_exclusions=True)._config.camera_primary_ray_exclusions is True
    with pytest.raises(ValidationError):
        create_easy_provider(camera_primary_ray_exclusions=False)
    with pytest.raises(ValidationError):
        create_easy_provider(backend_options={"camera_primary_ray_exclusions": False})


@pytest.mark.parametrize("value", ["1", "TRUE", "yes", " true", "", "0"])
def test_service_environment_rejects_noncanonical_bool(monkeypatch, value):
    monkeypatch.setenv("UNIROBOSIM_ISAACLAB_CAMERA_PRIMARY_RAY_EXCLUSIONS", value)
    with pytest.raises(ValidationError):
        create_easy_provider()
