from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from unirobosim import (
    ArrayValue,
    CameraModality,
    DebugBatch,
    DeformableBodySpec,
    DeformableTopology,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    FrozenMap,
    KinematicTarget,
    PointCommandMode,
    Pose,
    ProbeReport,
    ProviderDescriptor,
    WorldSpec,
)

from unirobosim_isaaclab.native_protocols import (
    Matrix,
    NativeArticulationCommand,
    NativeCameraCalibration,
    NativeEntityPrimState,
    NativeKinematicState,
    NativePhysicsDiagnostics,
    NativeRenderStateFrame,
    NativeSensorBatch,
    NativeSensorSample,
    PointBatch,
)


def available_probe(config: object, descriptor: ProviderDescriptor) -> ProbeReport:
    del config
    return ProbeReport(descriptor, True)


def unavailable_probe(config: object, descriptor: ProviderDescriptor) -> ProbeReport:
    del config
    return ProbeReport(descriptor, False, "disabled for test")


def make_articulation_asset(path: Path) -> Path:
    path.write_text("#usda 1.0\n", encoding="utf-8")
    return path


def make_world(asset: Path, *, world_id: str = "test-world", environments: int = 2) -> WorldSpec:
    surface = EntitySpec(
        EntityPath("/soft/cloth"),
        EntityKind.SURFACE_DEFORMABLE,
        deformable=DeformableBodySpec(
            DeformableTopology.SURFACE,
            ArrayValue.from_nested(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            surface_triangles=ArrayValue.from_nested(((0, 1, 2),), dtype="int64"),
        ),
    )
    volume = EntitySpec(
        EntityPath("/soft/jelly"),
        EntityKind.VOLUME_DEFORMABLE,
        deformable=DeformableBodySpec(
            DeformableTopology.VOLUME,
            ArrayValue.from_nested(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
            tetrahedra=ArrayValue.from_nested(((0, 1, 2, 3),), dtype="int64"),
            kinematic_node_indices=(0,),
            self_collision=True,
        ),
    )
    return WorldSpec(
        world_id,
        (
            EntitySpec(EntityPath("/props/marker"), EntityKind.RIGID_BODY, asset_uri=str(asset)),
            EntitySpec(
                EntityPath("/robots/arm"),
                EntityKind.ARTICULATION,
                joint_names=("joint_b", "joint_a"),
                initial_joint_positions=(0.2, -0.1),
                asset_uri=str(asset),
            ),
            surface,
            volume,
        ),
        environments=EnvironmentSpec(environments),
        metadata=FrozenMap({"seed": 7}),
    )


class FakeNativeWorld:
    def __init__(self, spec: WorldSpec, *, close_error: bool = False) -> None:
        self.spec = spec
        self.calls: list[tuple[str, Any]] = []
        self.closed = False
        self.close_error = close_error
        self.reset_error: BaseException | None = None
        self.step_error: BaseException | None = None
        self.rigid_poses = {
            entity.path: [((0.0, 0.0, 1.0), entity.pose.orientation_xyzw) for _ in range(spec.environments.count)]
            for entity in spec.entities
            if entity.kind is EntityKind.RIGID_BODY
        }
        self.entity_prim_poses = {
            entity.path: [
                (
                    Pose((0.0, 0.0, 1.0), entity.pose.orientation_xyzw)
                    if entity.kind is EntityKind.RIGID_BODY
                    else entity.pose
                )
                for _ in range(spec.environments.count)
            ]
            for entity in spec.entities
        }
        self.attachments: dict[tuple[int, str], EntityPath] = {}
        self.checkpoint_state: dict[str, object] = {
            "schema": "nvidia.isaaclab.native-state/1",
            "attachments": [],
        }

    def physics_diagnostics(self) -> NativePhysicsDiagnostics:
        return NativePhysicsDiagnostics(
            native_step_dt_seconds=self.spec.physics.time_step_seconds / self.spec.physics.substeps,
            substeps=self.spec.physics.substeps,
            world_step_dt_seconds=self.spec.physics.time_step_seconds,
            source="fake native context",
        )

    def reset(self, environment_indices: tuple[int, ...]) -> None:
        if self.reset_error is not None:
            raise self.reset_error
        self.calls.append(("reset", environment_indices))
        selected = frozenset(environment_indices)
        self.attachments = {
            key: child for key, child in self.attachments.items() if key[0] not in selected
        }

    def capture_checkpoint(self) -> dict[str, object]:
        self.calls.append(("capture_checkpoint", None))
        return copy.deepcopy(self.checkpoint_state)

    def restore_checkpoint(self, state: dict[str, object]) -> None:
        self.calls.append(("restore_checkpoint", copy.deepcopy(state)))
        self.checkpoint_state = copy.deepcopy(state)

    def apply_render_state(self, frame: NativeRenderStateFrame) -> None:
        self.calls.append(("render_state", frame))

    def apply_articulation(
        self,
        path: EntityPath,
        mode: object,
        targets: Matrix,
        environment_indices: tuple[int, ...],
        degree_of_freedom_indices: tuple[int, ...],
    ) -> None:
        self.calls.append(("articulation", (path, mode, targets, environment_indices, degree_of_freedom_indices)))

    def read_articulation(self, path: EntityPath) -> tuple[Matrix, Matrix]:
        self.calls.append(("read_articulation", path))
        count = self.spec.environments.count
        return tuple((0.2, -0.1) for _ in range(count)), tuple((0.0, 0.0) for _ in range(count))

    def apply_articulation_commands_and_step(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
    ) -> None:
        self.calls.append(("articulation_batch", commands))
        for command in commands:
            self.apply_articulation(
                command.path,
                command.mode,
                command.targets,
                command.environment_indices,
                command.degree_of_freedom_indices,
            )
        self.step(count)

    def apply_articulation_commands_step_and_read(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
        paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[Matrix, Matrix], ...]:
        self.apply_articulation_commands_and_step(commands, count)
        return tuple(self.read_articulation(path) for path in paths)

    def apply_articulation_commands_step_and_read_sensors(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
        paths: tuple[EntityPath, ...],
        sensor_paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[tuple[Matrix, Matrix], ...], NativeSensorBatch]:
        states = self.apply_articulation_commands_step_and_read(commands, count, paths)
        return states, self.read_sensors(sensor_paths)

    def apply_rigid_body_wrench(
        self,
        path: EntityPath,
        forces_n: Matrix,
        torques_n_m: Matrix,
        environment_indices: tuple[int, ...],
    ) -> None:
        self.calls.append(("rigid_wrench", (path, forces_n, torques_n_m, environment_indices)))

    def read_rigid_body(self, path: EntityPath) -> tuple[Matrix, Matrix, Matrix, Matrix]:
        self.calls.append(("read_rigid_body", path))
        count = self.spec.environments.count
        poses = self.rigid_poses[path]
        return (
            tuple(item[0] for item in poses),
            tuple(item[1] for item in poses),
            tuple((0.0, 0.0, 0.0) for _ in range(count)),
            tuple((0.0, 0.0, 0.0) for _ in range(count)),
        )

    def read_entity_prim_states(
        self,
        paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[NativeEntityPrimState, ...], ...]:
        self.calls.append(("read_entity_prim_states", paths))
        return tuple(
            tuple(NativeEntityPrimState(pose) for pose in self.entity_prim_poses[path])
            for path in paths
        )

    def set_entity_prim_pose(
        self,
        path: EntityPath,
        position_m: tuple[float, float, float],
        orientation_xyzw: tuple[float, float, float, float],
        environment_index: int,
    ) -> None:
        self.calls.append(("set_entity_prim_pose", (path, position_m, orientation_xyzw, environment_index)))
        pose = Pose(position_m, orientation_xyzw)
        self.entity_prim_poses[path][environment_index] = pose
        if path in self.rigid_poses:
            self.rigid_poses[path][environment_index] = (position_m, orientation_xyzw)

    def attach_rigid_body(
        self,
        attachment_id: str,
        parent_path: EntityPath,
        parent_link_name: str | None,
        child_path: EntityPath,
        child_link_name: str | None,
        environment_index: int,
        parent_T_child: Pose | None,
    ) -> Pose:
        relative = parent_T_child or Pose((0.0, 0.0, 1.0))
        self.calls.append(
            (
                "attach_rigid_body",
                (
                    attachment_id,
                    parent_path,
                    parent_link_name,
                    child_path,
                    child_link_name,
                    environment_index,
                    relative,
                ),
            )
        )
        self.attachments[environment_index, attachment_id] = child_path
        return relative

    def detach_rigid_body(
        self,
        attachment_id: str,
        child_path: EntityPath,
        environment_index: int,
    ) -> None:
        self.calls.append(("detach_rigid_body", (attachment_id, child_path, environment_index)))
        del self.attachments[environment_index, attachment_id]

    def read_contact(self, path: EntityPath) -> Matrix:
        self.calls.append(("read_contact", path))
        return tuple((0.0, 0.0, 9.81) for _ in range(self.spec.environments.count))

    def apply_deformable_position(
        self,
        path: EntityPath,
        targets: PointBatch,
        environment_indices: tuple[int, ...],
        point_indices: tuple[int, ...],
    ) -> None:
        self.calls.append(("deformable", (path, targets, environment_indices, point_indices)))

    def read_deformable(self, path: EntityPath) -> tuple[PointBatch, PointBatch]:
        self.calls.append(("read_deformable", path))
        entity = next(item for item in self.spec.entities if item.path == path)
        assert entity.deformable is not None
        points = tuple((float(index), 0.0, 1.0) for index in range(entity.deformable.node_count))
        zeros = tuple((0.0, 0.0, 0.0) for _ in points)
        return (
            tuple(points for _ in range(self.spec.environments.count)),
            tuple(zeros for _ in range(self.spec.environments.count)),
        )

    def apply_particle_fluid(
        self,
        path: EntityPath,
        mode: PointCommandMode,
        targets: PointBatch,
        environment_indices: tuple[int, ...],
        particle_indices: tuple[int, ...],
    ) -> None:
        self.calls.append(("fluid", (path, mode, targets, environment_indices, particle_indices)))

    def read_particle_fluid(self, path: EntityPath) -> tuple[PointBatch, PointBatch]:
        self.calls.append(("read_fluid", path))
        entity = next(item for item in self.spec.entities if item.path == path)
        assert entity.particle_fluid is not None
        points = tuple(
            (float(row[0]), float(row[1]), float(row[2]))
            for row in entity.particle_fluid.initial_particle_positions_m.rows()
        )
        zeros = tuple((0.0, 0.0, 0.0) for _ in points)
        return (
            tuple(points for _ in range(self.spec.environments.count)),
            tuple(zeros for _ in range(self.spec.environments.count)),
        )

    def read_sensor(self, path: EntityPath) -> NativeSensorSample:
        self.calls.append(("read_sensor", path))
        entity = next(item for item in self.spec.entities if item.path == path)
        assert entity.camera is not None
        channels = []
        for modality in entity.camera.modalities:
            if modality is CameraModality.RGB:
                shape = (
                    self.spec.environments.count,
                    entity.camera.height_px,
                    entity.camera.width_px,
                    3,
                )
                values: tuple[int | float, ...] | bytes = bytes((17,)) * math.prod(shape)
            elif modality is CameraModality.DEPTH:
                shape = (self.spec.environments.count, entity.camera.height_px, entity.camera.width_px)
                values = (1.25,) * math.prod(shape)
            else:
                shape = (self.spec.environments.count, entity.camera.height_px, entity.camera.width_px, 3)
                values = (0.0, 0.0, 1.0) * (math.prod(shape) // 3)
            channels.append((modality, shape, values))
        return tuple(channels)

    def read_sensors(self, paths: tuple[EntityPath, ...]) -> NativeSensorBatch:
        self.calls.append(("read_sensors", paths))
        samples = []
        for path in paths:
            entity = next(item for item in self.spec.entities if item.path == path)
            assert entity.camera is not None
            channels = []
            for modality in entity.camera.modalities:
                if modality is CameraModality.RGB:
                    shape = (
                        self.spec.environments.count,
                        entity.camera.height_px,
                        entity.camera.width_px,
                        3,
                    )
                    values: tuple[int | float, ...] | bytes = bytes((17,)) * math.prod(shape)
                elif modality is CameraModality.DEPTH:
                    shape = (self.spec.environments.count, entity.camera.height_px, entity.camera.width_px)
                    values = (1.25,) * math.prod(shape)
                else:
                    shape = (self.spec.environments.count, entity.camera.height_px, entity.camera.width_px, 3)
                    values = (0.0, 0.0, 1.0) * (math.prod(shape) // 3)
                channels.append((modality, shape, values))
            samples.append(tuple(channels))
        return tuple(samples)

    def camera_calibration(self, path: EntityPath) -> NativeCameraCalibration:
        self.calls.append(("camera_calibration", path))
        entity = next(item for item in self.spec.entities if item.path == path)
        assert entity.camera is not None
        return NativeCameraCalibration(
            resolution_px=(entity.camera.width_px, entity.camera.height_px),
            intrinsic_matrix=(3.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0),
            projection="perspective",
            focal_length=20.0,
            horizontal_aperture=10.0,
            clipping_range_m=(entity.camera.near_plane_m, entity.camera.far_plane_m),
            position_m=(0.0, 0.0, 0.0),
            orientation_opengl_xyzw=(0.0, 0.0, 0.0, 1.0),
        )

    def read_selected_kinematics(
        self,
        targets: tuple[KinematicTarget, ...],
        environment_index: int = 0,
    ) -> tuple[NativeKinematicState, ...]:
        self.calls.append(("read_selected_kinematics", (targets, environment_index)))
        return tuple(
            NativeKinematicState(
                target.target_id,
                target.entity_path,
                target.link_name,
                (0.4, -0.2, 0.9),
                (0.0, 0.0, 0.0, 1.0),
                (0.1, 0.2, 0.3),
                (0.0, 1.0, 0.0),
            )
            for target in targets
        )

    def publish_debug(self, batch: DebugBatch) -> tuple[int, int, int]:
        self.calls.append(("publish_debug", batch))
        return len(batch.primitives), 0, len(batch.primitives)

    def clear_debug(self, layer: str | None, group: str | None, primitive_id: str | None) -> int:
        self.calls.append(("clear_debug", (layer, group, primitive_id)))
        return 1

    def step(self, count: int) -> None:
        if self.step_error is not None:
            raise self.step_error
        self.calls.append(("step", count))

    def close(self) -> None:
        self.calls.append(("close", None))
        self.closed = True
        if self.close_error:
            raise RuntimeError("native close failed")


class FakeNativeRuntime:
    def __init__(self, *, build_failures: int = 0, close_error: bool = False) -> None:
        self.build_failures = build_failures
        self.close_error = close_error
        self.worlds: list[FakeNativeWorld] = []
        self.closed = False

    def build_world(self, spec: WorldSpec) -> FakeNativeWorld:
        if self.build_failures:
            self.build_failures -= 1
            raise RuntimeError("injected native build failure")
        world = FakeNativeWorld(spec, close_error=self.close_error)
        self.worlds.append(world)
        return world

    def close(self) -> None:
        self.closed = True
