from __future__ import annotations

import ast
import inspect
import tomllib
from pathlib import Path

import pytest

import unirobosim_isaaclab.droid_acceptance as droid_acceptance
from unirobosim_isaaclab._droid_asset import DROID_ASSET_ENV, resolve_droid_asset_path
from unirobosim_isaaclab.droid_acceptance import create_backend_run


def _asset(root: Path, name: str) -> Path:
    result = root / name
    result.write_bytes(b"#usda 1.0\n")
    return result.resolve()


def test_explicit_asset_path_has_highest_precedence(tmp_path: Path) -> None:
    explicit = _asset(tmp_path, "explicit.usd")
    configured = _asset(tmp_path, "configured.usd")
    environment = _asset(tmp_path, "environment.usd")

    assert (
        resolve_droid_asset_path(
            explicit,
            configured_asset_path=configured,
            environ={DROID_ASSET_ENV: str(environment)},
        )
        == explicit
    )


def test_config_asset_path_precedes_environment(tmp_path: Path) -> None:
    configured = _asset(tmp_path, "configured.usd")
    environment = _asset(tmp_path, "environment.usd")

    assert (
        resolve_droid_asset_path(
            configured_asset_path=str(configured),
            environ={DROID_ASSET_ENV: str(environment)},
        )
        == configured
    )


def test_environment_supplies_asset_when_no_explicit_or_config_path(tmp_path: Path) -> None:
    environment = _asset(tmp_path, "environment.usd")

    assert resolve_droid_asset_path(environ={DROID_ASSET_ENV: str(environment)}) == environment


def test_missing_asset_has_actionable_portable_error() -> None:
    with pytest.raises(FileNotFoundError) as caught:
        resolve_droid_asset_path(environ={})

    message = str(caught.value)
    assert "asset_path" in message
    assert "robot.asset_path" in message
    assert DROID_ASSET_ENV in message
    assert "/home/" not in message


def test_selected_path_must_be_a_regular_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_droid_asset_path(tmp_path / "missing.usd", environ={})


def test_acceptance_asset_path_is_optional_keyword_only() -> None:
    parameter = inspect.signature(create_backend_run).parameters["asset_path"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_acceptance_forwards_explicit_and_config_asset_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _asset(tmp_path, "selected.usd")
    captured: list[tuple[object, object]] = []

    def resolve(asset_path: object, *, configured_asset_path: object) -> Path:
        captured.append((asset_path, configured_asset_path))
        return selected

    monkeypatch.setattr(droid_acceptance, "resolve_droid_asset_path", resolve)
    spec = {
        "schema_version": "fastsim-droid-three-backend-equivalence/4",
        "simulation": {"physics_hz": 240},
        "camera": {"fps": 30},
        "robot": {"asset_path": "configured.usd"},
    }

    with pytest.raises(RuntimeError, match="missing or changed"):
        create_backend_run(spec, "rulebased_blocking", tmp_path / "output", asset_path="explicit.usd")

    assert captured == [("explicit.usd", "configured.usd")]


def test_native_smoke_cli_has_no_machine_local_default() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "scripts" / "droid_090_native_smoke.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    asset_argument = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(isinstance(value, ast.Constant) and value.value == "--asset" for value in node.args)
    )
    assert not any(keyword.arg == "default" for keyword in asset_argument.keywords)
    assert "/home/" not in source


def test_release_author_identity_is_hofee() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["authors"] == [{"name": "Hofee", "email": "lexhofee@gmail.com"}]
