"""Isaac Lab 3.0 native runtime.

This module is imported only after the lightweight compatibility probe succeeds. AppLauncher is
constructed before importing simulation, torch, Omni, or USD modules.
"""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace
from typing import Any

from unirobosim import CommandMode, EntityKind, EntityPath, EntitySpec, WorldSpec

from .config import IsaacLabAdapterConfig
from .native_protocols import Matrix, PointBatch


def _native_name(path: EntityPath) -> str:
    digest = hashlib.sha256(path.value.encode()).hexdigest()[:10]
    return f"{path.name.replace('-', '_').replace('.', '_')}_{digest}"


def _environment_origins(count: int, spacing: float) -> tuple[tuple[float, float, float], ...]:
    columns = math.ceil(math.sqrt(count))
    return tuple(((index % columns) * spacing, (index // columns) * spacing, 0.0) for index in range(count))


def _rotate_xyzw(
    vector: tuple[float, float, float], quaternion: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _surface_from_tetrahedra(
    tetrahedra: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    faces: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for a, b, c, d in tetrahedra:
        for face in ((a, b, c), (a, d, b), (a, c, d), (b, d, c)):
            ordered = sorted(face)
            key = (ordered[0], ordered[1], ordered[2])
            faces[key] = face if key not in faces else None
    return tuple(face for face in faces.values() if face is not None)


def _launcher_kwargs(config: IsaacLabAdapterConfig, *, process_isolated: bool = False) -> dict[str, object]:
    """Build launcher arguments that are safe for an embedded library runtime."""

    launcher_args: dict[str, object] = {
        "headless": config.headless,
        "device": config.device,
        "enable_cameras": config.enable_cameras,
        # Isaac Sim 6 requires fast shutdown to avoid unsafe native-plugin
        # teardown. It is enabled only inside the adapter-owned worker process.
        "fast_shutdown": process_isolated,
    }
    if config.experience is not None:
        launcher_args["experience"] = config.experience
    return launcher_args


class IsaacLabNativeRuntime:
    """Own exactly one Kit application and at most one native world."""

    def __init__(self, config: IsaacLabAdapterConfig, *, process_isolated: bool = False) -> None:
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        self._launcher = AppLauncher(**_launcher_kwargs(config, process_isolated=process_isolated))
        self._app = self._launcher.app

        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]
        from isaaclab.assets import (  # type: ignore[import-not-found]
            Articulation,
            DeformableObject,
            DeformableObjectCfg,
        )
        from isaaclab.assets.articulation import ArticulationCfg  # type: ignore[import-not-found]
        from isaaclab.sim.schemas import define_deformable_body_properties  # type: ignore[import-not-found]
        from isaaclab_physx.physics import PhysxCfg  # type: ignore[import-not-found]
        from isaaclab_physx.sim.schemas import (  # type: ignore[import-not-found]
            PhysxDeformableBodyPropertiesCfg,
        )
        from pxr import UsdGeom, UsdPhysics, Vt  # type: ignore[import-not-found]

        self._modules = SimpleNamespace(
            Articulation=Articulation,
            ArticulationCfg=ArticulationCfg,
            DeformableObject=DeformableObject,
            DeformableObjectCfg=DeformableObjectCfg,
            ImplicitActuatorCfg=ImplicitActuatorCfg,
            PhysxCfg=PhysxCfg,
            PhysxDeformableBodyPropertiesCfg=PhysxDeformableBodyPropertiesCfg,
            define_deformable_body_properties=define_deformable_body_properties,
            sim_utils=sim_utils,
            torch=torch,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            Vt=Vt,
        )
        self._config = config
        self._active_world: IsaacLabNativeWorld | None = None
        self._closed = False

    def build_world(self, spec: WorldSpec) -> IsaacLabNativeWorld:
        if self._closed:
            raise RuntimeError("native runtime is closed")
        if self._active_world is not None:
            raise RuntimeError("native runtime already owns a world")
        world: IsaacLabNativeWorld | None = None
        try:
            world = IsaacLabNativeWorld(self, spec, self._config, self._modules)
        except Exception:
            if world is not None:
                world.close()
            else:
                self._modules.sim_utils.SimulationContext.clear_instance()
            raise
        self._active_world = world
        return world

    def _world_closed(self, world: IsaacLabNativeWorld) -> None:
        if self._active_world is world:
            self._active_world = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        world = self._active_world
        self._active_world = None
        try:
            if world is not None:
                world._close(notify_runtime=False)
        finally:
            self._app.close()


class IsaacLabNativeWorld:
    def __init__(
        self,
        runtime: IsaacLabNativeRuntime,
        spec: WorldSpec,
        config: IsaacLabAdapterConfig,
        modules: SimpleNamespace,
    ) -> None:
        self._runtime = runtime
        self._spec = spec
        self._config = config
        self._m = modules
        self._closed = False
        self._sim: Any | None = None
        self._articulations: dict[EntityPath, Any] = {}
        self._deformables: dict[EntityPath, Any] = {}
        self._joint_maps: dict[EntityPath, tuple[int, ...]] = {}
        self._initial_articulation: dict[EntityPath, tuple[Any, Any, Any]] = {}
        self._initial_deformable: dict[EntityPath, tuple[Any, Any | None]] = {}
        self._origins_cpu = _environment_origins(spec.environments.count, config.environment_spacing_m)
        self._origins: Any | None = None
        self._native_dt = spec.physics.time_step_seconds / spec.physics.substeps
        try:
            self._build()
        except Exception:
            self._close(notify_runtime=False)
            raise

    def _build(self) -> None:
        sim_utils = self._m.sim_utils
        sim_utils.SimulationContext.clear_instance()
        sim_utils.create_new_stage()
        sim_cfg = sim_utils.SimulationCfg(
            dt=self._native_dt,
            gravity=self._spec.physics.gravity_m_s2,
            device=self._config.device,
            physics=self._m.PhysxCfg(),
            use_fabric=True,
            render_interval=1,
        )
        self._sim = sim_utils.SimulationContext(sim_cfg)
        for index, origin in enumerate(self._origins_cpu):
            sim_utils.create_prim(f"/World/env_{index}", "Xform", translation=origin)
        for entity in self._spec.entities:
            if entity.kind is EntityKind.ARTICULATION:
                self._author_articulation(entity)
            elif entity.kind is EntityKind.RIGID_BODY:
                self._author_rigid(entity)
            elif entity.kind in {EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE}:
                self._author_deformable(entity)
        self._sim.reset()
        self._origins = self._m.torch.tensor(self._origins_cpu, device=self._sim.device, dtype=self._m.torch.float32)
        self._initialize_articulations()
        self._initialize_deformables()
        self.reset(tuple(range(self._spec.environments.count)))

    def _author_articulation(self, entity: EntitySpec) -> None:
        assert entity.asset_uri is not None
        cfg = self._m.ArticulationCfg(
            prim_path=f"/World/env_.*/{_native_name(entity.path)}",
            spawn=self._m.sim_utils.UsdFileCfg(usd_path=str(entity.asset_uri).removeprefix("file://")),
            init_state=self._m.ArticulationCfg.InitialStateCfg(
                pos=entity.pose.position,
                rot=entity.pose.orientation_xyzw,
                joint_pos=dict(zip(entity.joint_names, entity.initial_joint_positions, strict=True)),
                joint_vel={".*": 0.0},
            ),
            actuators={
                "unirobosim": self._m.ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=self._config.position_stiffness,
                    damping=self._config.position_damping,
                )
            },
        )
        self._articulations[entity.path] = self._m.Articulation(cfg)

    def _author_rigid(self, entity: EntitySpec) -> None:
        assert entity.asset_uri is not None
        for index in range(self._spec.environments.count):
            root = f"/World/env_{index}/{_native_name(entity.path)}"
            self._m.sim_utils.create_prim(
                root,
                "Xform",
                translation=entity.pose.position,
                orientation=entity.pose.orientation_xyzw,
                usd_path=str(entity.asset_uri).removeprefix("file://"),
            )
            rigid_prims = self._m.sim_utils.get_all_matching_child_prims(
                root,
                lambda prim: prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI),
            )
            if len(rigid_prims) != 1:
                raise ValueError(
                    f"rigid asset must contain exactly one UsdPhysics.RigidBodyAPI prim; found {len(rigid_prims)}"
                )

    def _author_deformable(self, entity: EntitySpec) -> None:
        assert entity.deformable is not None
        deformable = entity.deformable
        points = deformable.rest_positions_m.rows()
        surface: tuple[tuple[int, int, int], ...] = (
            ()
            if deformable.surface_triangles is None
            else tuple((int(face[0]), int(face[1]), int(face[2])) for face in deformable.surface_triangles.rows())
        )
        tetrahedra: tuple[tuple[int, int, int, int], ...] = (
            ()
            if deformable.tetrahedra is None
            else tuple((int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])) for tet in deformable.tetrahedra.rows())
        )
        if not surface and tetrahedra:
            surface = _surface_from_tetrahedra(tetrahedra)
        name = _native_name(entity.path)
        stage = self._m.sim_utils.get_current_stage()
        for index in range(self._spec.environments.count):
            root = f"/World/env_{index}/{name}"
            self._m.sim_utils.create_prim(
                root,
                "Xform",
                translation=entity.pose.position,
                orientation=entity.pose.orientation_xyzw,
            )
            vis_mesh = self._m.UsdGeom.Mesh.Define(stage, f"{root}/vis_mesh")
            vis_mesh.GetPointsAttr().Set(self._m.Vt.Vec3fArray(points))
            vis_mesh.GetFaceVertexIndicesAttr().Set(
                self._m.Vt.IntArray(tuple(value for face in surface for value in face))
            )
            vis_mesh.GetFaceVertexCountsAttr().Set(self._m.Vt.IntArray((3,) * len(surface)))
            if entity.kind is EntityKind.VOLUME_DEFORMABLE:
                sim_mesh = self._m.UsdGeom.TetMesh.Define(stage, f"{root}/sim_mesh")
                sim_mesh.GetPointsAttr().Set(self._m.Vt.Vec3fArray(points))
                sim_mesh.GetTetVertexIndicesAttr().Set(self._m.Vt.Vec4iArray(tetrahedra))
            properties = self._m.PhysxDeformableBodyPropertiesCfg(
                mass=deformable.node_mass_kg * deformable.node_count,
                linear_damping=deformable.linear_damping_per_s,
                self_collision=deformable.self_collision,
            )
            self._m.define_deformable_body_properties(
                root,
                properties,
                deformable_type="surface" if entity.kind is EntityKind.SURFACE_DEFORMABLE else "volume",
            )
        cfg = self._m.DeformableObjectCfg(prim_path=f"/World/env_.*/{name}", spawn=None)
        self._deformables[entity.path] = self._m.DeformableObject(cfg)

    def _initialize_articulations(self) -> None:
        torch = self._m.torch
        assert self._sim is not None
        for path, asset in self._articulations.items():
            entity = next(item for item in self._spec.entities if item.path == path)
            native_names = tuple(asset.joint_names)
            if set(native_names) != set(entity.joint_names) or len(native_names) != len(entity.joint_names):
                raise ValueError(
                    f"joint names for {path.value} do not exactly match the USD; "
                    f"declared={entity.joint_names}, native={native_names}"
                )
            joint_map = tuple(native_names.index(name) for name in entity.joint_names)
            self._joint_maps[path] = joint_map
            root_pose = asset.data.default_root_pose.torch.clone()
            assert self._origins is not None
            root_pose[:, :3] += self._origins
            positions = torch.zeros(
                (self._spec.environments.count, len(native_names)), device=self._sim.device, dtype=torch.float32
            )
            for public_index, native_index in enumerate(joint_map):
                positions[:, native_index] = entity.initial_joint_positions[public_index]
            velocities = torch.zeros_like(positions)
            self._initial_articulation[path] = (root_pose, positions, velocities)

    def _initialize_deformables(self) -> None:
        torch = self._m.torch
        assert self._sim is not None
        for path, asset in self._deformables.items():
            entity = next(item for item in self._spec.entities if item.path == path)
            assert entity.deformable is not None
            if asset.max_sim_vertices_per_body != entity.deformable.node_count:
                raise ValueError(
                    f"deformable node count changed for {path.value}: "
                    f"expected {entity.deformable.node_count}, native {asset.max_sim_vertices_per_body}"
                )
            state = asset.data.nodal_state_w.torch.clone()
            velocities = tuple(
                _rotate_xyzw((float(row[0]), float(row[1]), float(row[2])), entity.pose.orientation_xyzw)
                for row in entity.deformable.initial_velocities().rows()
            )
            state[..., 3:] = torch.tensor(velocities, device=self._sim.device, dtype=state.dtype).unsqueeze(0)
            target = None
            if entity.kind is EntityKind.VOLUME_DEFORMABLE:
                target = asset.data.nodal_kinematic_target.torch.clone()
                target[..., :3] = state[..., :3]
                target[..., 3] = 1.0
                if entity.deformable.kinematic_node_indices:
                    target[:, list(entity.deformable.kinematic_node_indices), 3] = 0.0
            self._initial_deformable[path] = (state, target)

    def reset(self, environment_indices: tuple[int, ...]) -> None:
        env_ids = list(environment_indices)
        for path, asset in self._articulations.items():
            root_pose, positions, velocities = self._initial_articulation[path]
            asset.write_root_pose_to_sim_index(root_pose=root_pose[env_ids], env_ids=env_ids)
            asset.write_joint_position_to_sim_index(position=positions[env_ids], env_ids=env_ids)
            asset.write_joint_velocity_to_sim_index(velocity=velocities[env_ids], env_ids=env_ids)
            asset.set_joint_position_target_index(target=positions[env_ids], env_ids=env_ids)
            asset.set_joint_velocity_target_index(target=velocities[env_ids], env_ids=env_ids)
            asset.set_joint_effort_target_index(target=self._m.torch.zeros_like(positions[env_ids]), env_ids=env_ids)
            asset.reset(env_ids=env_ids)
        for path, asset in self._deformables.items():
            state, target = self._initial_deformable[path]
            asset.write_nodal_state_to_sim_index(state[env_ids], env_ids=env_ids)
            if target is not None:
                asset.write_nodal_kinematic_target_to_sim_index(target[env_ids], env_ids=env_ids)
            asset.reset(env_ids=env_ids)
        assert self._sim is not None
        self._sim.forward()
        self._update_assets(0.0)

    def apply_articulation(
        self,
        path: EntityPath,
        mode: CommandMode,
        targets: Matrix,
        environment_indices: tuple[int, ...],
        degree_of_freedom_indices: tuple[int, ...],
    ) -> None:
        asset = self._articulations[path]
        assert self._sim is not None
        env_ids = list(environment_indices)
        joint_ids = [self._joint_maps[path][index] for index in degree_of_freedom_indices]
        target = self._m.torch.tensor(targets, device=self._sim.device, dtype=self._m.torch.float32)
        zeros = self._m.torch.zeros_like(target)
        if mode is CommandMode.POSITION:
            stiffness = self._config.position_stiffness
            damping = self._config.position_damping
            asset.set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids)
            asset.set_joint_velocity_target_index(target=zeros, joint_ids=joint_ids, env_ids=env_ids)
            asset.set_joint_effort_target_index(target=zeros, joint_ids=joint_ids, env_ids=env_ids)
        elif mode is CommandMode.VELOCITY:
            stiffness = 0.0
            damping = self._config.velocity_damping
            asset.set_joint_velocity_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids)
            asset.set_joint_effort_target_index(target=zeros, joint_ids=joint_ids, env_ids=env_ids)
        else:
            stiffness = 0.0
            damping = 0.0
            asset.set_joint_effort_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids)
        asset.write_joint_stiffness_to_sim_index(
            stiffness=self._m.torch.full_like(target, stiffness), joint_ids=joint_ids, env_ids=env_ids
        )
        asset.write_joint_damping_to_sim_index(
            damping=self._m.torch.full_like(target, damping), joint_ids=joint_ids, env_ids=env_ids
        )
        asset.write_data_to_sim()

    def read_articulation(self, path: EntityPath) -> tuple[Matrix, Matrix]:
        asset = self._articulations[path]
        joint_map = list(self._joint_maps[path])
        positions = asset.data.joint_pos.torch[:, joint_map].detach().cpu().tolist()
        velocities = asset.data.joint_vel.torch[:, joint_map].detach().cpu().tolist()
        return (
            tuple(tuple(float(value) for value in row) for row in positions),
            tuple(tuple(float(value) for value in row) for row in velocities),
        )

    def apply_deformable_position(
        self,
        path: EntityPath,
        targets: PointBatch,
        environment_indices: tuple[int, ...],
        point_indices: tuple[int, ...],
    ) -> None:
        asset = self._deformables[path]
        assert self._sim is not None
        env_ids = list(environment_indices)
        current = asset.data.nodal_kinematic_target.torch[env_ids].clone()
        assert self._origins is not None
        for row_index, environment in enumerate(environment_indices):
            for column_index, point in enumerate(point_indices):
                value = self._m.torch.tensor(targets[row_index][column_index], device=self._sim.device)
                current[row_index, point, :3] = value + self._origins[environment]
                current[row_index, point, 3] = 0.0
        asset.write_nodal_kinematic_target_to_sim_index(current, env_ids=env_ids)

    def read_deformable(self, path: EntityPath) -> tuple[PointBatch, PointBatch]:
        asset = self._deformables[path]
        assert self._origins is not None
        positions = asset.data.nodal_pos_w.torch - self._origins[:, None, :]
        velocities = asset.data.nodal_vel_w.torch
        return self._point_batch(positions), self._point_batch(velocities)

    @staticmethod
    def _point_batch(tensor: Any) -> PointBatch:
        values = tensor.detach().cpu().tolist()
        return tuple(
            tuple((float(vector[0]), float(vector[1]), float(vector[2])) for vector in environment)
            for environment in values
        )

    def step(self, count: int) -> None:
        assert self._sim is not None
        for _ in range(count):
            for _ in range(self._spec.physics.substeps):
                for asset in self._articulations.values():
                    asset.write_data_to_sim()
                for asset in self._deformables.values():
                    asset.write_data_to_sim()
                self._sim.step(render=self._config.render)
                self._update_assets(self._native_dt)

    def _update_assets(self, dt: float) -> None:
        for asset in self._articulations.values():
            asset.update(dt)
        for asset in self._deformables.values():
            asset.update(dt)

    def _close(self, *, notify_runtime: bool) -> None:
        if self._closed:
            return
        self._closed = True
        sim = self._sim
        self._sim = None
        self._articulations.clear()
        self._deformables.clear()
        self._initial_articulation.clear()
        self._initial_deformable.clear()
        try:
            if sim is not None:
                sim.stop()
        finally:
            self._m.sim_utils.SimulationContext.clear_instance()
            if notify_runtime:
                self._runtime._world_closed(self)

    def close(self) -> None:
        self._close(notify_runtime=True)
