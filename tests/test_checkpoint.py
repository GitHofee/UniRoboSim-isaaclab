from __future__ import annotations

from pathlib import Path

import pytest
from unirobosim import (
    ArrayValue,
    ArticulationCommand,
    CheckpointWorld,
    CommandMode,
    EntityPath,
    ValidationError,
)

from .helpers import FakeNativeRuntime, make_articulation_asset, make_world
from .test_lifecycle_world import open_test_session


def _attachment_record() -> dict[str, object]:
    return {
        "attachment_id": "held-object",
        "environment_index": 0,
        "parent_path": "/robots/arm",
        "parent_link_name": None,
        "child_path": "/props/marker",
        "child_link_name": None,
        "parent_T_child": {
            "position": [0.0, 0.0, 0.1],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }


def test_checkpoint_round_trip_is_opaque_and_does_not_rewind_tick(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    try:
        world = session.build(make_world(make_articulation_asset(tmp_path / "world.usda")))
        assert isinstance(world, CheckpointWorld)
        native = runtime.worlds[-1]
        native.checkpoint_state = {
            "schema": "nvidia.isaaclab.native-state/1",
            "revision": 7,
            "attachments": [_attachment_record()],
        }
        checkpoint = world.create_checkpoint()
        world.step(2)
        current_tick = world.tick
        native.checkpoint_state = {
            "schema": "nvidia.isaaclab.native-state/1",
            "revision": 99,
            "attachments": [],
        }

        result = world.restore_checkpoint(checkpoint)

        assert native.checkpoint_state["revision"] == 7
        assert result.tick == current_tick
        assert world.tick == current_tick
        assert result.restored_entity_count == 4
    finally:
        session.close()


def test_checkpoint_rejects_pending_commands_before_native_mutation(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    try:
        world = session.build(make_world(make_articulation_asset(tmp_path / "world.usda")))
        robot = world.resolve(EntityPath("/robots/arm"))
        world.apply_articulation_command(
            ArticulationCommand(
                robot,
                CommandMode.POSITION,
                ArrayValue.from_rows(((0.0, 0.0), (0.0, 0.0))),
            )
        )

        with pytest.raises(ValidationError, match="pending articulation-command"):
            world.create_checkpoint()

        assert not any(call[0] == "capture_checkpoint" for call in runtime.worlds[-1].calls)
    finally:
        session.close()
