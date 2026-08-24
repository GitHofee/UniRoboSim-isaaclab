from __future__ import annotations

import os
import sys

import pytest
from unirobosim import ValidationError

import unirobosim_isaaclab
from unirobosim_isaaclab import IsaacLabAdapterConfig

_NATIVE_SDK_ROOTS = {"isaaclab", "isaacsim", "omni", "pxr", "torch", "torchvision", "torchaudio"}


def _easy_config() -> IsaacLabAdapterConfig:
    provider = unirobosim_isaaclab.create_easy_provider()
    return provider._config


def _explicit_config(profile: str) -> IsaacLabAdapterConfig:
    provider = unirobosim_isaaclab.create_easy_provider(launch_profile=profile)
    return provider._config


def test_easy_launch_profile_absent_preserves_batch_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(unirobosim_isaaclab.ISAACLAB_LAUNCH_PROFILE_ENV, raising=False)

    config = _easy_config()

    assert config.headless is True
    assert config.enable_cameras is True
    assert config.render is True
    assert config.render_on_step is False
    assert config.max_render_hz is None


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("headless", (True, True, True, False, None)),
        ("headless-physics", (True, False, False, False, None)),
        ("visible", (False, True, True, True, 60.0)),
    ],
)
def test_easy_launch_profile_exact_values(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    expected: tuple[bool, bool, bool, bool, float | None],
) -> None:
    monkeypatch.setenv(unirobosim_isaaclab.ISAACLAB_LAUNCH_PROFILE_ENV, profile)

    config = _easy_config()

    assert (
        config.headless,
        config.enable_cameras,
        config.render,
        config.render_on_step,
        config.max_render_hz,
    ) == expected


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("headless", (True, True, True, False, None)),
        ("headless-physics", (True, False, False, False, None)),
        ("visible", (False, True, True, True, 60.0)),
    ],
)
def test_explicit_launch_profile_is_authoritative_without_environment_read(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    expected: tuple[bool, bool, bool, bool, float | None],
) -> None:
    monkeypatch.setattr(os, "getenv", lambda name: pytest.fail(f"unexpected environment read: {name}"))

    config = _explicit_config(profile)

    assert (
        config.headless,
        config.enable_cameras,
        config.render,
        config.render_on_step,
        config.max_render_hz,
    ) == expected


@pytest.mark.parametrize("profile", ["", "VISIBLE", " visible", "visible ", "auto", "1", True, 1])
def test_explicit_launch_profile_rejects_noncanonical_values_without_environment_or_sdk_import(
    monkeypatch: pytest.MonkeyPatch,
    profile: object,
) -> None:
    monkeypatch.setattr(os, "getenv", lambda name: pytest.fail(f"unexpected environment read: {name}"))

    with pytest.raises(ValidationError) as caught:
        unirobosim_isaaclab.create_easy_provider(launch_profile=profile)  # type: ignore[arg-type]

    assert caught.value.operation == "isaaclab.launch_profile.resolve"
    assert not (_NATIVE_SDK_ROOTS & set(sys.modules))


@pytest.mark.parametrize("profile", ["", "VISIBLE", " visible", "visible ", "auto", "1"])
def test_easy_launch_profile_rejects_every_other_value_before_sdk_import(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    monkeypatch.setenv(unirobosim_isaaclab.ISAACLAB_LAUNCH_PROFILE_ENV, profile)

    with pytest.raises(ValidationError) as caught:
        unirobosim_isaaclab.create_easy_provider()

    assert caught.value.operation == "isaaclab.launch_profile.resolve"
    assert not (_NATIVE_SDK_ROOTS & set(sys.modules))


def test_easy_launch_profile_hostile_value_has_bounded_non_reflective_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = "secret-token:" + "x" * 1_000_000 + "\ncontrol"
    monkeypatch.setattr(os, "getenv", lambda name: hostile)

    with pytest.raises(ValidationError) as caught:
        unirobosim_isaaclab.create_easy_provider()

    rendered = str(caught.value)
    assert len(rendered) < 512
    assert "secret-token" not in rendered
    assert "control" not in rendered
    assert not (_NATIVE_SDK_ROOTS & set(sys.modules))


def test_easy_provider_reads_the_profile_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def getenv(name: str) -> str | None:
        calls.append(name)
        return None

    monkeypatch.setattr(os, "getenv", getenv)

    config = _easy_config()

    assert config.headless is True
    assert calls == [unirobosim_isaaclab.ISAACLAB_LAUNCH_PROFILE_ENV]


def test_explicit_provider_creation_does_not_read_the_easy_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getenv", lambda name: pytest.fail(f"unexpected environment read: {name}"))

    provider = unirobosim_isaaclab.create_provider()

    assert provider._config.headless is True
