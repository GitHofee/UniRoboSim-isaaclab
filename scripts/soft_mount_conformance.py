"""Bounded native soft-mount smoke; no installed-package modifications."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from unirobosim import (
    ArrayValue,
    BoxGeometrySpec,
    DeformableBodySpec,
    DeformableMaterialSpec,
    DeformableRigidMount,
    DeformableTopology,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    PhysicsSpec,
    Pose,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider


def spec(mode: str, substeps: int) -> WorldSpec:
    parent = EntitySpec(
        EntityPath("/parent"),
        EntityKind.RIGID_BODY,
        pose=Pose((0, 0, 1), (0, 0, math.sqrt(0.5), math.sqrt(0.5))),
        box=BoxGeometrySpec((0.04, 0.04, 0.04), 1.0),
    )
    pad = EntitySpec(
        EntityPath("/pad"),
        EntityKind.VOLUME_DEFORMABLE,
        pose=Pose((0, 0, 0.03)),
        deformable=DeformableBodySpec(
            DeformableTopology.VOLUME,
            ArrayValue.from_rows(((0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.04, 0.0), (0.0, 0.0, 0.04))),
            tetrahedra=ArrayValue.from_rows(((0, 1, 2, 3),), dtype="int64"),
            node_mass_kg=0.01,
            elastic_material=DeformableMaterialSpec(20000.0),
            rigid_mount=DeformableRigidMount(parent.path, node_indices=(0,), coupling=mode),
            initial_node_velocities_m_s=ArrayValue.from_rows(
                ((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            ),
            contact_offset_m=0.003,
        ),
    )
    return WorldSpec(
        "soft-mount-" + mode + "-" + str(substeps),
        (pad, parent),
        environments=EnvironmentSpec(2),
        physics=PhysicsSpec(time_step_seconds=1 / 120, substeps=substeps),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["solver", "kinematic_boundary"], required=True)
    p.add_argument("--substeps", type=int, default=2)
    args = p.parse_args()
    result = {"passed": False, "mode": args.mode, "substeps": args.substeps}
    session = None
    try:
        provider = create_provider(IsaacLabAdapterConfig(headless=True, device="cuda:0", render=False))
        session = provider.open()
        world_spec = spec(args.mode, args.substeps)
        result["spec"] = world_spec.to_dict()
        world = session.build(world_spec)
        handle = world.resolve(EntityPath("/pad"))
        parent = world.resolve(EntityPath("/parent"))
        initial = world.read_deformable(handle)
        result["initial"] = initial.node_positions_m.nested()
        result["initial_velocities"] = initial.node_velocities_m_s.nested()
        assert abs(initial.node_velocities_m_s.nested()[0][1][1] - 0.01) < 1e-5
        samples = []
        for i in range(20):
            world.step(1)
            state = world.read_deformable(handle)
            body = world.read_rigid_body(parent)
            assert all(
                math.isfinite(float(v)) for v in (*state.node_positions_m.values, *state.node_velocities_m_s.values)
            )
            samples.append(
                {"step": i + 1, "nodes": state.node_positions_m.nested(), "parent": body.positions_m.nested()}
            )
        before = world.read_deformable(handle).node_positions_m.nested()
        world.reset((0,))
        after = world.read_deformable(handle)
        assert after.node_positions_m.nested()[1] == before[1], "partial reset altered other environment"
        assert (
            max(
                abs(a - b)
                for x, y in zip(after.node_positions_m.nested()[0], initial.node_positions_m.nested()[0], strict=True)
                for a, b in zip(x, y, strict=True)
            )
            < 1e-5
        )
        assert (
            max(
                abs(a - b)
                for x, y in zip(
                    after.node_velocities_m_s.nested()[0], initial.node_velocities_m_s.nested()[0], strict=True
                )
                for a, b in zip(x, y, strict=True)
            )
            < 1e-5
        )
        world.step(2)
        world.close()
        session.close()
        session = None
        result.update(passed=True, samples=samples, partial_reset=True, rotated_initial_velocity=True, close=True)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        if session:
            try:
                session.close()
            except BaseException:
                result["close_error"] = traceback.format_exc()
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "spec")}, indent=2))


if __name__ == "__main__":
    main()
