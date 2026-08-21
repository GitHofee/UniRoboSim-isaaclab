"""Minimal real-backend smoke for an articulation and PBD particles in one world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unirobosim import (
    ArrayValue,
    ArticulationCommand,
    CommandMode,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    ParticleFluidSpec,
    PhysicsSpec,
    Pose,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, IsaacLabProvider, create_provider


def _particles() -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (0.55 + 0.04 * x, -0.06 + 0.04 * y, 1.1 + 0.04 * z) for z in range(2) for y in range(3) for x in range(3)
    )


def run(asset: Path, *, direct: bool) -> dict[str, object]:
    config = IsaacLabAdapterConfig(headless=True, device="cuda:0")
    if direct:
        from unirobosim_isaaclab.native import IsaacLabNativeRuntime

        provider = IsaacLabProvider(
            config,
            runtime_factory=lambda selected: IsaacLabNativeRuntime(selected, process_isolated=False),
        )
    else:
        provider = create_provider(config)
    session = provider.open()
    try:
        spec = WorldSpec(
            "mixed-articulation-fluid-smoke",
            (
                EntitySpec(
                    EntityPath("/articulation/microwave"),
                    EntityKind.ARTICULATION,
                    pose=Pose(position=(0.0, 0.0, 0.5)),
                    joint_names=("door_hinge",),
                    initial_joint_positions=(0.0,),
                    asset_uri=str(asset),
                ),
                EntitySpec(
                    EntityPath("/fluid/water"),
                    EntityKind.PARTICLE_FLUID,
                    particle_fluid=ParticleFluidSpec(
                        ArrayValue.from_nested(_particles()),
                        particle_radius_m=0.018,
                    ),
                ),
            ),
            environments=EnvironmentSpec(1),
            physics=PhysicsSpec(time_step_seconds=1.0 / 120.0, substeps=2),
        )
        world = session.build(spec)
        articulation = world.resolve(EntityPath("/articulation/microwave"))
        fluid = world.resolve(EntityPath("/fluid/water"))
        before_joint = world.read_articulation(articulation)
        before_fluid = world.read_particle_fluid(fluid)
        world.apply_articulation_command(
            ArticulationCommand(
                articulation,
                CommandMode.POSITION,
                ArrayValue.from_nested(((0.35,),)),
            )
        )
        world.step(120)
        after_joint = world.read_articulation(articulation)
        after_fluid = world.read_particle_fluid(fluid)
        result = {
            "status": "passed",
            "joint_before_rad": before_joint.joint_positions.nested(),
            "joint_after_rad": after_joint.joint_positions.nested(),
            "fluid_shape": after_fluid.particle_positions_m.shape,
            "fluid_max_delta_m": max(
                abs(float(after) - float(before))
                for after, before in zip(
                    after_fluid.particle_positions_m.values,
                    before_fluid.particle_positions_m.values,
                    strict=True,
                )
            ),
            "tick": {
                "step_index": world.tick.step_index,
                "sim_time_seconds": world.tick.sim_time_seconds,
            },
        }
        world.close()
        return result
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.asset, direct=args.direct)
    except Exception as exc:
        causes: list[dict[str, str]] = []
        current: BaseException | None = exc
        while current is not None:
            causes.append({"type": type(current).__name__, "message": str(current)})
            current = current.__cause__
        result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "causes": causes}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
