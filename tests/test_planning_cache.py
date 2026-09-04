from __future__ import annotations

import sqlite3

from unirobosim_isaaclab._planning_cache import PlanningMeshCache, cache_key
from unirobosim_isaaclab.native_planning import _mesh_source_sha256


def _component(seed: int = 1) -> tuple[bytes, int, int]:
    return (bytes([seed]) * 24, 1, 1)


def test_cache_key_is_stable_and_length_delimited() -> None:
    assert cache_key("ab", "c") == cache_key("ab", "c")
    assert cache_key("ab", "c") != cache_key("a", "bc")
    assert cache_key("mesh", "scale:1") != cache_key("mesh", "scale:2")


def test_planning_mesh_cache_round_trips_across_connections(tmp_path) -> None:
    path = tmp_path / "planning.sqlite3"
    first = PlanningMeshCache(path)
    first.put("mesh", (_component(3), _component(7)))
    first.close()

    second = PlanningMeshCache(path)
    assert second.get("mesh") == (_component(3), _component(7))
    second.close()


def test_corrupt_cache_entry_is_a_safe_miss(tmp_path) -> None:
    path = tmp_path / "planning.sqlite3"
    cache = PlanningMeshCache(path)
    cache.put("mesh", (_component(),))
    cache.close()
    connection = sqlite3.connect(path)
    connection.execute("UPDATE mesh SET payload = ? WHERE key = ?", (b"broken", "mesh"))
    connection.commit()
    connection.close()

    reopened = PlanningMeshCache(path)
    assert reopened.get("mesh") is None
    reopened.put("mesh", (_component(9),))
    assert reopened.get("mesh") == (_component(9),)
    reopened.close()


def test_invalid_component_is_not_persisted(tmp_path) -> None:
    cache = PlanningMeshCache(tmp_path / "planning.sqlite3")
    cache.put("invalid", ((b"short", 1, 1),))
    assert cache.get("invalid") is None
    cache.close()


def test_mesh_source_hash_is_stable_and_content_sensitive() -> None:
    arguments = (((0.0, 1.0, 2.0),), (3,), (0, 0, 0), (), "none", "rightHanded")
    expected = _mesh_source_sha256(*arguments)
    assert _mesh_source_sha256(*arguments) == expected
    assert _mesh_source_sha256(((0.0, 1.0, 2.5),), *arguments[1:]) != expected
    assert _mesh_source_sha256(*arguments[:-1], "leftHanded") != expected
