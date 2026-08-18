"""Real Isaac Lab 3.0 / Isaac Sim 6.0.1 GPU conformance runner."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from unirobosim import (
    ArrayValue,
    ArticulationCommand,
    CapabilityId,
    CapabilityRequirement,
    CommandMode,
    DeformableBodySpec,
    DeformableCommand,
    DeformableTopology,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    PhysicsSpec,
    PointCommandMode,
    Pose,
    RigidBodyCommand,
    WorldBuildError,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider


def _versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("unirobosim", "unirobosim-isaaclab", "isaaclab", "isaaclab_physx", "isaacsim", "torch")
    }


def _create_articulated_appliance_usd(path: Path) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Appliance")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    base = UsdGeom.Cube.Define(stage, "/Appliance/base")
    base.CreateSizeAttr(0.2)
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(1.0)

    door = UsdGeom.Cube.Define(stage, "/Appliance/door")
    door.CreateSizeAttr(0.2)
    door.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.4))
    UsdPhysics.RigidBodyAPI.Apply(door.GetPrim())
    UsdPhysics.CollisionAPI.Apply(door.GetPrim())
    UsdPhysics.MassAPI.Apply(door.GetPrim()).CreateMassAttr(1.0)

    fixed = UsdPhysics.FixedJoint.Define(stage, "/Appliance/fixed_base")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/Appliance/base")])

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Appliance/joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Appliance/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Appliance/door")])
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(90.0)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.2))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -0.2))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateMaxForceAttr(1000.0)
    drive.CreateStiffnessAttr(1000.0)
    drive.CreateDampingAttr(100.0)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()


def _create_rigid_contact_usd(path: Path, *, initial_height_m: float) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    rigid = UsdGeom.Cube.Define(stage, "/Asset/Rigid")
    rigid.CreateSizeAttr(0.2)
    UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
    UsdPhysics.CollisionAPI.Apply(rigid.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(rigid.GetPrim()).CreateMassAttr(1.0)
    support = UsdGeom.Cube.Define(stage, "/Asset/Support")
    support.CreateSizeAttr(1.0)
    support.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -initial_height_m))
    UsdPhysics.CollisionAPI.Apply(support.GetPrim()).CreateCollisionEnabledAttr(True)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()


def _rigid_world(dynamic_asset: Path, world_id: str = "native-rigid") -> WorldSpec:
    return WorldSpec(
        world_id,
        (
            EntitySpec(
                EntityPath("/props/cube"),
                EntityKind.RIGID_BODY,
                pose=Pose(position=(0.0, 0.0, 0.8)),
                asset_uri=str(dynamic_asset),
            ),
        ),
        environments=EnvironmentSpec(2),
        physics=PhysicsSpec(time_step_seconds=1.0 / 120.0, substeps=2),
        requirements=(
            CapabilityRequirement(CapabilityId("control.rigid_body.wrench@1")),
            CapabilityRequirement(CapabilityId("contact.binary@1")),
            CapabilityRequirement(CapabilityId("contact.net_normal_force@1")),
        ),
    )


def _articulated_object_world(asset: Path, joint_name: str, world_id: str) -> WorldSpec:
    return WorldSpec(
        world_id,
        (
            EntitySpec(
                EntityPath("/articulated/microwave"),
                EntityKind.ARTICULATION,
                pose=Pose(position=(0.0, 0.0, 0.5)),
                joint_names=(joint_name,),
                initial_joint_positions=(0.0,),
                asset_uri=str(asset),
            ),
        ),
        environments=EnvironmentSpec(2),
        physics=PhysicsSpec(time_step_seconds=1.0 / 120.0, substeps=2),
    )


def _soft_world() -> WorldSpec:
    cloth = EntitySpec(
        EntityPath("/soft/cloth"),
        EntityKind.SURFACE_DEFORMABLE,
        pose=Pose(position=(0.0, 0.0, 1.5)),
        deformable=DeformableBodySpec(
            DeformableTopology.SURFACE,
            ArrayValue.from_nested(((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.3, 0.3, 0.0), (0.0, 0.3, 0.0))),
            surface_triangles=ArrayValue.from_nested(((0, 1, 2), (0, 2, 3)), dtype="int64"),
            node_mass_kg=0.01,
            linear_damping_per_s=0.1,
        ),
    )
    jelly = EntitySpec(
        EntityPath("/soft/jelly"),
        EntityKind.VOLUME_DEFORMABLE,
        pose=Pose(position=(0.5, 0.0, 1.0)),
        deformable=DeformableBodySpec(
            DeformableTopology.VOLUME,
            ArrayValue.from_nested(((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2))),
            tetrahedra=ArrayValue.from_nested(((0, 1, 2, 3),), dtype="int64"),
            kinematic_node_indices=(0,),
            node_mass_kg=0.02,
            linear_damping_per_s=0.1,
            self_collision=True,
        ),
    )
    return WorldSpec(
        "native-soft",
        (cloth, jelly),
        environments=EnvironmentSpec(2),
        physics=PhysicsSpec(time_step_seconds=1.0 / 120.0, substeps=2),
    )


def _finite(values: tuple[float | int | bool, ...]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def run() -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {"status": "running", "checks": [], "versions": _versions()}
    provider = create_provider(IsaacLabAdapterConfig(headless=True, device="cuda:0"))
    probe = provider.probe()
    result["probe"] = {
        "available": probe.available,
        "reason": probe.reason,
        "details": probe.details.to_dict(),
    }
    if not probe.available:
        raise RuntimeError(f"native profile unavailable: {probe.reason}")

    session = provider.open()
    try:
        with tempfile.TemporaryDirectory(prefix="unirobosim-isaaclab-") as directory:
            articulation_usd = Path(directory) / "minimal_articulated_appliance.usda"
            rigid_usd = Path(directory) / "minimal_rigid.usda"
            _create_articulated_appliance_usd(articulation_usd)
            _create_rigid_contact_usd(rigid_usd, initial_height_m=0.8)

            try:
                session.build(_articulated_object_world(articulation_usd, "wrong_joint", "expected-failure"))
            except WorldBuildError as exc:
                result["checks"].append({"name": "transactional_build_failure", "passed": True, "error_code": exc.code})
            else:
                raise AssertionError("joint mismatch build unexpectedly succeeded")

            print("[CONFORMANCE] building rigid/contact world", flush=True)
            rigid = session.build(_rigid_world(rigid_usd))
            cube = rigid.resolve(EntityPath("/props/cube"))
            assert cube.entity_kind is EntityKind.RIGID_BODY
            initial = rigid.read_rigid_body(cube)
            assert initial.positions_m.shape == (2, 3)
            assert initial.orientations_xyzw.shape == (2, 4)
            assert _finite(initial.positions_m.values)
            initial_positions = initial.positions_m.nested()
            rigid_origin_parity = max(
                abs(float(initial_positions[0][axis]) - float(initial_positions[1][axis])) for axis in range(3)
            )
            assert rigid_origin_parity <= 1.0e-5

            rigid.apply_rigid_body_command(
                RigidBodyCommand(
                    cube,
                    ArrayValue.from_nested(((5.0, 0.0, 0.0),)),
                    ArrayValue.from_nested(((0.0, 0.0, 1.0),)),
                    environment_indices=(0,),
                )
            )
            rigid.step(4)
            first_wrench_state = rigid.read_rigid_body(cube)
            rigid.step(4)
            second_wrench_state = rigid.read_rigid_body(cube)
            first_velocity = first_wrench_state.linear_velocities_m_s.nested()
            second_velocity = second_wrench_state.linear_velocities_m_s.nested()
            angular_velocity = second_wrench_state.angular_velocities_rad_s.nested()
            assert float(first_velocity[0][0]) > float(first_velocity[1][0]) + 0.05
            assert float(second_velocity[0][0]) > float(first_velocity[0][0]) + 0.05
            assert abs(float(angular_velocity[0][2])) > abs(float(angular_velocity[1][2])) + 0.05
            selected_environment_velocity_delta = float(second_velocity[0][0]) - float(second_velocity[1][0])

            rigid.apply_rigid_body_command(
                RigidBodyCommand(
                    cube,
                    ArrayValue.from_nested(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
                    ArrayValue.from_nested(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
                )
            )
            rigid.reset((0, 1))
            rigid.step(80)
            settled = rigid.read_rigid_body(cube)
            contact = rigid.read_contact(cube, force_threshold_n=0.1)
            assert contact.net_normal_forces_n.shape == (2, 3)
            assert contact.in_contact.values == (True, True), (
                f"expected contact in both environments; flags={contact.in_contact.values}, "
                f"forces={contact.net_normal_forces_n.rows()}, positions={settled.positions_m.rows()}"
            )
            contact_norms = tuple(
                math.sqrt(sum(float(component) ** 2 for component in row)) for row in contact.net_normal_forces_n.rows()
            )
            assert min(contact_norms) > 0.1
            settled_positions = settled.positions_m.nested()
            assert all(0.55 <= float(position[2]) <= 0.7 for position in settled_positions)

            rigid.reset((0,))
            partially_reset = rigid.read_rigid_body(cube).positions_m.nested()
            partial_reset_height_delta = float(partially_reset[0][2]) - float(partially_reset[1][2])
            assert partial_reset_height_delta > 0.1
            rigid.close()
            result["rigid_environment_origin_parity_max_abs_m"] = rigid_origin_parity
            result["rigid_selected_environment_velocity_delta_m_s"] = selected_environment_velocity_delta
            result["rigid_contact_force_norms_n"] = contact_norms
            result["rigid_partial_reset_height_delta_m"] = partial_reset_height_delta
            result["checks"].append(
                {
                    "name": "rigid_state_persistent_wrench_contact_and_partial_reset",
                    "passed": True,
                }
            )

            print("[CONFORMANCE] building soft-body world", flush=True)
            soft = session.build(_soft_world())
            cloth = soft.read_deformable(soft.resolve(EntityPath("/soft/cloth")))
            jelly_handle = soft.resolve(EntityPath("/soft/jelly"))
            jelly = soft.read_deformable(jelly_handle)
            assert cloth.node_positions_m.shape == (2, 4, 3)
            assert jelly.node_positions_m.shape == (2, 4, 3)
            assert _finite(cloth.node_positions_m.values)
            assert _finite(jelly.node_positions_m.values)
            jelly_by_environment = jelly.node_positions_m.nested()
            parity_error = max(
                abs(float(jelly_by_environment[0][node][axis]) - float(jelly_by_environment[1][node][axis]))
                for node in range(4)
                for axis in range(3)
            )
            assert parity_error <= 1.0e-5, f"environment-local state mismatch: {parity_error} m"
            result["environment_origin_parity_max_abs_m"] = parity_error
            target = (0.5, 0.0, 1.25)
            soft.apply_deformable_command(
                DeformableCommand(
                    jelly_handle,
                    PointCommandMode.POSITION,
                    ArrayValue.from_nested(((target,), (target,))),
                    environment_indices=(0, 1),
                    node_indices=(0,),
                )
            )
            soft.step(4)
            moved_cloth = soft.read_deformable(soft.resolve(EntityPath("/soft/cloth")))
            moved = soft.read_deformable(jelly_handle)
            for environment in moved.node_positions_m.nested():
                _assert_vector_close(environment[0], target, tolerance=0.03)
            cloth_before = cloth.node_positions_m.nested()
            cloth_after = moved_cloth.node_positions_m.nested()
            jelly_before = jelly.node_positions_m.nested()
            jelly_after = moved.node_positions_m.nested()
            surface_drop = min(
                float(cloth_before[environment][node][2]) - float(cloth_after[environment][node][2])
                for environment in range(2)
                for node in range(4)
            )
            volume_free_node_drop = min(
                float(jelly_before[environment][node][2]) - float(jelly_after[environment][node][2])
                for environment in range(2)
                for node in (1, 2, 3)
            )
            assert surface_drop > 1.0e-5, f"surface deformable did not respond to gravity: {surface_drop} m"
            assert volume_free_node_drop > 1.0e-5, (
                f"volume deformable free nodes did not respond to gravity: {volume_free_node_drop} m"
            )
            result["surface_gravity_drop_min_m"] = surface_drop
            result["volume_free_node_gravity_drop_min_m"] = volume_free_node_drop
            soft.close()
            result["checks"].append({"name": "surface_volume_state_and_kinematic_control", "passed": True})

            print("[CONFORMANCE] building articulated-object world", flush=True)
            articulation = session.build(
                _articulated_object_world(articulation_usd, "joint", "native-articulated-appliance")
            )
            handle = articulation.resolve(EntityPath("/articulated/microwave"))
            before = articulation.read_articulation(handle)
            assert before.joint_positions.shape == (2, 1)
            for mode, value in (
                (CommandMode.POSITION, 0.25),
                (CommandMode.VELOCITY, -0.1),
                (CommandMode.EFFORT, 0.2),
            ):
                articulation.apply_articulation_command(
                    ArticulationCommand(handle, mode, ArrayValue.from_nested(((value,), (value,))))
                )
                articulation.step(2)
                state = articulation.read_articulation(handle)
                assert _finite(state.joint_positions.values)
                assert _finite(state.joint_velocities.values)
            articulation.reset((0, 1))
            articulation.close()
            result["checks"].append({"name": "non_robot_articulation_state_and_three_control_modes", "passed": True})
    finally:
        session.close()

    with tempfile.TemporaryDirectory(prefix="unirobosim-isaaclab-reopen-") as directory:
        rigid_usd = Path(directory) / "minimal_rigid.usda"
        _create_rigid_contact_usd(rigid_usd, initial_height_m=0.8)
        reopened_session = provider.open()
        try:
            print("[CONFORMANCE] verifying provider reopen", flush=True)
            reopened_world = reopened_session.build(_rigid_world(rigid_usd, "native-rigid-reopened"))
            assert reopened_world.step(1).step_index == 1
            reopened_world.close()
        finally:
            reopened_session.close()
    result["checks"].append({"name": "provider_reopen_after_worker_shutdown", "passed": True})

    result["status"] = "passed"
    result["elapsed_seconds"] = time.time() - started
    return result


def _assert_vector_close(actual: object, expected: tuple[float, float, float], *, tolerance: float) -> None:
    assert isinstance(actual, (tuple, list)) and len(actual) == 3
    assert all(abs(float(actual[index]) - expected[index]) <= tolerance for index in range(3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run()
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "versions": _versions(),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
