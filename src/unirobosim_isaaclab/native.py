"""Isaac Lab 3.0 native runtime.

This module is imported only after the lightweight compatibility probe succeeds. AppLauncher is
constructed before importing simulation, torch, Omni, or USD modules.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from unirobosim import (
    CameraModality,
    CommandMode,
    DebugBatch,
    DebugLifetimeMode,
    DebugPrimitive,
    DebugPrimitiveKind,
    EntityKind,
    EntityPath,
    EntitySpec,
    PointCommandMode,
    WorldSpec,
)

from .config import _ANTI_ALIASING_MODES, IsaacLabAdapterConfig
from .native_debug import NativeDebugOverlay, NativeDebugPayload
from .native_protocols import Matrix, NativeDebugReport, NativeSensorSample, PointBatch

Vector3 = tuple[float, float, float]
Segment = tuple[Vector3, Vector3]
Color = tuple[float, float, float, float]

_GLYPHS = {
    "0": "01110/10001/10011/10101/11001/10001/01110",
    "1": "00100/01100/00100/00100/00100/00100/01110",
    "2": "01110/10001/00001/00010/00100/01000/11111",
    "3": "11110/00001/00001/01110/00001/00001/11110",
    "4": "00010/00110/01010/10010/11111/00010/00010",
    "5": "11111/10000/10000/11110/00001/00001/11110",
    "6": "01110/10000/10000/11110/10001/10001/01110",
    "7": "11111/00001/00010/00100/01000/01000/01000",
    "8": "01110/10001/10001/01110/10001/10001/01110",
    "9": "01110/10001/10001/01111/00001/00001/01110",
    "A": "01110/10001/10001/11111/10001/10001/10001",
    "B": "11110/10001/10001/11110/10001/10001/11110",
    "C": "01111/10000/10000/10000/10000/10000/01111",
    "D": "11110/10001/10001/10001/10001/10001/11110",
    "E": "11111/10000/10000/11110/10000/10000/11111",
    "F": "11111/10000/10000/11110/10000/10000/10000",
    "G": "01111/10000/10000/10111/10001/10001/01110",
    "H": "10001/10001/10001/11111/10001/10001/10001",
    "I": "01110/00100/00100/00100/00100/00100/01110",
    "J": "00001/00001/00001/00001/10001/10001/01110",
    "K": "10001/10010/10100/11000/10100/10010/10001",
    "L": "10000/10000/10000/10000/10000/10000/11111",
    "M": "10001/11011/10101/10101/10001/10001/10001",
    "N": "10001/11001/10101/10011/10001/10001/10001",
    "O": "01110/10001/10001/10001/10001/10001/01110",
    "P": "11110/10001/10001/11110/10000/10000/10000",
    "Q": "01110/10001/10001/10001/10101/10010/01101",
    "R": "11110/10001/10001/11110/10100/10010/10001",
    "S": "01111/10000/10000/01110/00001/00001/11110",
    "T": "11111/00100/00100/00100/00100/00100/00100",
    "U": "10001/10001/10001/10001/10001/10001/01110",
    "V": "10001/10001/10001/10001/10001/01010/00100",
    "W": "10001/10001/10001/10101/10101/10101/01010",
    "X": "10001/10001/01010/00100/01010/10001/10001",
    "Y": "10001/10001/01010/00100/00100/00100/00100",
    "Z": "11111/00001/00010/00100/01000/10000/11111",
    "-": "00000/00000/00000/11111/00000/00000/00000",
    ".": "00000/00000/00000/00000/00000/00110/00110",
    ":": "00000/00110/00110/00000/00110/00110/00000",
    "?": "01110/10001/00001/00010/00100/00000/00100",
    " ": "00000/00000/00000/00000/00000/00000/00000",
}


def _find_debug_extension(isaacsim_root: Path) -> Path:
    extension = next(iter(sorted((isaacsim_root / "extscache").glob("isaacsim.util.debug_draw-*"))), None)
    if extension is None:
        raise RuntimeError(
            "Isaac native debug sink requires isaacsim.util.debug_draw; install "
            "isaacsim-extscache-kit-sdk matching the Isaac Sim version"
        )
    return extension


def _offset(point: tuple[float, ...], origin: Vector3) -> Vector3:
    return float(point[0]) + origin[0], float(point[1]) + origin[1], float(point[2]) + origin[2]


def _text_segments(anchor: Vector3, value: str, scale: float) -> tuple[Segment, ...]:
    cell = scale / 7.0
    cursor = 0.0
    result: list[Segment] = []
    for character in value.upper():
        rows = _GLYPHS.get(character, _GLYPHS["?"]).split("/")
        for row_index, row in enumerate(rows):
            for column_index, enabled in enumerate(row):
                if enabled == "1":
                    left = (
                        anchor[0],
                        anchor[1] + cursor + column_index * cell,
                        anchor[2] + (6 - row_index) * cell,
                    )
                    right = (left[0], left[1] + cell * 0.8, left[2])
                    result.append((left, right))
        cursor += cell * 6.0
    return tuple(result)


def _debug_points(primitive: DebugPrimitive, origins: tuple[Vector3, ...]) -> tuple[Vector3, ...]:
    if primitive.kind is not DebugPrimitiveKind.POINT_SET:
        return ()
    nested = primitive.geometry_m.nested()
    return tuple(
        _offset(point, origins[environment])
        for row_index, environment in enumerate(primitive.environment_indices)
        for point in nested[row_index]
    )


def _append_segment(groups: dict[Color, list[Segment]], color: Color, segment: Segment) -> None:
    groups.setdefault(color, []).append(segment)


def _debug_line_groups(
    primitive: DebugPrimitive,
    origins: tuple[Vector3, ...],
) -> dict[Color, tuple[Segment, ...]]:
    nested = primitive.geometry_m.nested()
    groups: dict[Color, list[Segment]] = {}
    color = primitive.color_rgba
    if primitive.kind is DebugPrimitiveKind.LINE_LIST:
        for row_index, environment in enumerate(primitive.environment_indices):
            for segment in nested[row_index]:
                _append_segment(
                    groups,
                    color,
                    (_offset(segment[0], origins[environment]), _offset(segment[1], origins[environment])),
                )
    elif primitive.kind is DebugPrimitiveKind.COORDINATE_AXES:
        axis_colors: tuple[Color, ...] = (
            (1.0, 0.15, 0.15, color[3]),
            (0.15, 1.0, 0.25, color[3]),
            (0.15, 0.4, 1.0, color[3]),
        )
        axes: tuple[Vector3, ...] = (
            (primitive.size, 0.0, 0.0),
            (0.0, primitive.size, 0.0),
            (0.0, 0.0, primitive.size),
        )
        for row_index, environment in enumerate(primitive.environment_indices):
            for raw_pose in nested[row_index]:
                origin = _offset(raw_pose[:3], origins[environment])
                quaternion = tuple(float(item) for item in raw_pose[3:7])
                for axis, axis_color in zip(axes, axis_colors, strict=True):
                    endpoint = _rotate_xyzw(axis, quaternion)  # type: ignore[arg-type]
                    _append_segment(groups, axis_color, (origin, _offset(endpoint, origin)))
    elif primitive.kind is DebugPrimitiveKind.TEXT:
        assert primitive.text is not None
        for row_index, environment in enumerate(primitive.environment_indices):
            for text_index, anchor in enumerate(nested[row_index]):
                world_anchor = _offset(anchor, origins[environment])
                for segment in _text_segments(world_anchor, primitive.text[row_index][text_index], primitive.size):
                    _append_segment(groups, color, segment)
    elif primitive.kind is DebugPrimitiveKind.BOUNDING_BOX:
        edges = (
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        )
        for row_index, environment in enumerate(primitive.environment_indices):
            for raw_box in nested[row_index]:
                center = _offset(raw_box[:3], origins[environment])
                half = tuple(float(item) * 0.5 for item in raw_box[3:6])
                quaternion = tuple(float(item) for item in raw_box[6:10])
                corners = tuple(
                    _offset(
                        _rotate_xyzw(
                            (
                                half[0] if index & 1 else -half[0],
                                half[1] if index & 2 else -half[1],
                                half[2] if index & 4 else -half[2],
                            ),
                            quaternion,  # type: ignore[arg-type]
                        ),
                        center,
                    )
                    for index in range(8)
                )
                for edge_left, edge_right in edges:
                    _append_segment(groups, color, (corners[edge_left], corners[edge_right]))
    elif primitive.kind is DebugPrimitiveKind.TRAJECTORY:
        for row_index, environment in enumerate(primitive.environment_indices):
            points = tuple(_offset(point, origins[environment]) for point in nested[row_index])
            for point_left, point_right in zip(points, points[1:], strict=False):
                _append_segment(groups, color, (point_left, point_right))
    return {key: tuple(value) for key, value in groups.items()}


def _debug_draw_payload(
    primitives: Iterable[DebugPrimitive],
    origins: tuple[Vector3, ...],
) -> NativeDebugPayload:
    """Lower portable primitives without importing any Isaac/Kit modules."""

    points: list[Vector3] = []
    point_colors: list[Color] = []
    point_sizes: list[float] = []
    line_starts: list[Vector3] = []
    line_ends: list[Vector3] = []
    line_colors: list[Color] = []
    line_widths: list[float] = []
    for primitive in primitives:
        lowered_points = _debug_points(primitive, origins)
        if lowered_points:
            points.extend(lowered_points)
            point_colors.extend((primitive.color_rgba,) * len(lowered_points))
            # The native API uses viewport pixels, while UniRoboSim sizes are expressed in
            # world-scale display units. This bounded conversion keeps tiny SI values visible.
            point_sizes.extend((max(1.0, min(64.0, primitive.size * 50.0)),) * len(lowered_points))
        line_groups = _debug_line_groups(primitive, origins)
        if primitive.kind in {DebugPrimitiveKind.COORDINATE_AXES, DebugPrimitiveKind.TEXT}:
            line_width = 2.0
        else:
            line_width = max(1.0, min(10.0, primitive.size * 40.0))
        for color, segments in line_groups.items():
            for start, end in segments:
                line_starts.append(start)
                line_ends.append(end)
                line_colors.append(color)
                line_widths.append(line_width)
    return NativeDebugPayload(
        points=tuple(points),
        point_colors=tuple(point_colors),
        point_sizes=tuple(point_sizes),
        line_starts=tuple(line_starts),
        line_ends=tuple(line_ends),
        line_colors=tuple(line_colors),
        line_widths=tuple(line_widths),
    )


@dataclass
class _FluidSet:
    points: Any
    initial_positions: tuple[tuple[float, float, float], ...]
    initial_velocities: tuple[tuple[float, float, float], ...]


@dataclass
class _UsdRigid:
    rigid_prim: Any


@dataclass
class _UsdArticulation:
    root_prim: Any


def _ensure_launcher_setting(setting: str) -> None:
    if setting not in sys.argv:
        sys.argv.append(setting)


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


def _transform_position(
    vector: tuple[float, float, float],
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    rotated = _rotate_xyzw(vector, quaternion)
    return (
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
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
    if config.enable_cameras:
        launcher_args["anti_aliasing"] = _ANTI_ALIASING_MODES[config.anti_aliasing]
    if config.experience is not None:
        launcher_args["experience"] = config.experience
    return launcher_args


def _camera_launcher_settings(config: IsaacLabAdapterConfig) -> tuple[str, ...]:
    """Return pre-launch RTX settings for the requested texture residency profile."""

    texture_streaming = "true" if config.texture_streaming else "false"
    settings = [
        "--/renderer/multiGpu/enabled=false",
        "--/rtx-transient/dlssg/enabled=false",
        f"--/rtx-transient/resourcemanager/enableTextureStreaming={texture_streaming}",
    ]
    if not config.texture_streaming:
        settings.append("--/rtx-transient/resourcemanager/texturestreaming/async=false")
    return tuple(settings)


class IsaacLabNativeRuntime:
    """Own exactly one Kit application and at most one native world."""

    def __init__(self, config: IsaacLabAdapterConfig, *, process_isolated: bool = False) -> None:
        if config.enable_cameras:
            # The installed RTX 5090 profile is stable with one renderer device and no frame generation.
            for setting in _camera_launcher_settings(config):
                _ensure_launcher_setting(setting)
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        self._launcher = AppLauncher(**_launcher_kwargs(config, process_isolated=process_isolated))
        self._app = self._launcher.app

        import carb  # type: ignore[import-not-found]

        if config.enable_cameras:
            expected_anti_aliasing = _ANTI_ALIASING_MODES[config.anti_aliasing]
            # Isaac Lab applies its rendering-mode preset after constructing SimulationApp,
            # which currently overwrites SimulationApp's ``anti_aliasing`` launch value.
            # Re-apply the caller's mode after AppLauncher has completed, before any render
            # products exist, then read it back so a silent preset override cannot pass.
            render_settings = carb.settings.get_settings()
            render_settings.set("/rtx/post/aa/op", expected_anti_aliasing)
            render_settings.set(
                "/rtx-transient/resourcemanager/enableTextureStreaming",
                config.texture_streaming,
            )
            if config.fluid_render_mode == "isosurface":
                render_settings.set("/rtx/rendermode", "RealTimePathTracing")
                render_settings.set("/rtx/translucency/enabled", True)
                render_settings.set("/rtx/translucency/maxRefractionBounces", 12)
                render_settings.set("/rtx/rtpt/maxBounces", 6)
                render_settings.set("/rtx/rtpt/maxSpecularAndTransmissionBounces", 6)
                render_settings.set("/rtx/rtpt/maxVolumeBounces", 6)
            actual_anti_aliasing = render_settings.get("/rtx/post/aa/op")
            actual_texture_streaming = render_settings.get("/rtx-transient/resourcemanager/enableTextureStreaming")
            if actual_anti_aliasing != expected_anti_aliasing:
                raise RuntimeError(
                    "Isaac Sim did not apply the requested camera anti-aliasing mode: "
                    f"requested={config.anti_aliasing!r} expected={expected_anti_aliasing} "
                    f"actual={actual_anti_aliasing!r}"
                )
            if actual_texture_streaming != config.texture_streaming:
                raise RuntimeError(
                    "Isaac Sim did not apply the requested texture-streaming mode: "
                    f"requested={config.texture_streaming!r} actual={actual_texture_streaming!r}"
                )
            if config.fluid_render_mode == "isosurface" and not render_settings.get("/rtx/translucency/enabled"):
                raise RuntimeError("Isaac Sim did not enable RTX translucency for fluid isosurface rendering")
            if (
                config.fluid_render_mode == "isosurface"
                and render_settings.get("/rtx/rendermode") != "RealTimePathTracing"
            ):
                raise RuntimeError("Isaac Sim did not enable real-time path tracing for fluid isosurface rendering")
        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import isaacsim  # type: ignore[import-not-found]
        import omni.physics.tensors as physics_tensors  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]
        from isaaclab.assets import (  # type: ignore[import-not-found]
            Articulation,
            DeformableObject,
            DeformableObjectCfg,
            RigidObject,
        )
        from isaaclab.assets.articulation import ArticulationCfg  # type: ignore[import-not-found]
        from isaaclab.assets.rigid_object import RigidObjectCfg  # type: ignore[import-not-found]
        from isaaclab.sensors.camera import Camera, CameraCfg  # type: ignore[import-not-found]
        from isaaclab.sensors.contact_sensor import (  # type: ignore[import-not-found]
            ContactSensor,
            ContactSensorCfg,
        )
        from isaaclab.sim.schemas import define_deformable_body_properties  # type: ignore[import-not-found]
        from isaaclab_physx.physics import PhysxCfg  # type: ignore[import-not-found]
        from isaaclab_physx.sim.schemas import (  # type: ignore[import-not-found]
            PhysxDeformableBodyPropertiesCfg,
        )

        isaacsim_root = Path(next(iter(isaacsim.__path__)))
        debug_extension = _find_debug_extension(isaacsim_root)
        # Loading this one native plugin directly avoids asking Kit to re-resolve
        # all optional packages in the 16 GB pip SDK cache after AppLauncher has
        # started.  The latter is both slow and fails offline on unrelated
        # telemetry metadata in Isaac Sim 6.0.1.
        debug_python_root = str(debug_extension / "isaacsim")
        if debug_python_root not in isaacsim.__path__:
            isaacsim.__path__.append(debug_python_root)
        from isaacsim.util.debug_draw import _debug_draw  # type: ignore[import-not-found]
        from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt  # type: ignore[import-not-found]

        self._modules = SimpleNamespace(
            Articulation=Articulation,
            ArticulationCfg=ArticulationCfg,
            DeformableObject=DeformableObject,
            DeformableObjectCfg=DeformableObjectCfg,
            RigidObject=RigidObject,
            RigidObjectCfg=RigidObjectCfg,
            ContactSensor=ContactSensor,
            ContactSensorCfg=ContactSensorCfg,
            Camera=Camera,
            CameraCfg=CameraCfg,
            ImplicitActuatorCfg=ImplicitActuatorCfg,
            PhysxCfg=PhysxCfg,
            PhysxDeformableBodyPropertiesCfg=PhysxDeformableBodyPropertiesCfg,
            define_deformable_body_properties=define_deformable_body_properties,
            carb=carb,
            debug_draw=_debug_draw,
            debug_draw_plugin_path=debug_extension / "bin",
            sim_utils=sim_utils,
            physics_tensors=physics_tensors,
            torch=torch,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
            PhysxSchema=PhysxSchema,
            Sdf=Sdf,
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
        self._usd_articulations: dict[EntityPath, tuple[_UsdArticulation, ...]] = {}
        self._usd_articulation_views: dict[EntityPath, Any] = {}
        self._initial_usd_articulation: dict[EntityPath, tuple[Any, Any, Any, Any]] = {}
        self._rigids: dict[EntityPath, Any] = {}
        self._usd_rigids: dict[EntityPath, tuple[_UsdRigid, ...]] = {}
        self._usd_rigid_views: dict[EntityPath, Any] = {}
        self._initial_usd_rigid: dict[EntityPath, tuple[Any, Any]] = {}
        self._usd_rigid_wrenches: dict[EntityPath, tuple[Any, Any]] = {}
        self._usd_tensor_view: Any | None = None
        self._contacts: dict[EntityPath, Any] = {}
        self._deformables: dict[EntityPath, Any] = {}
        self._fluids: dict[EntityPath, tuple[_FluidSet, ...]] = {}
        self._cameras: dict[EntityPath, Any] = {}
        self._debug_draw_interface: Any | None = None
        self._debug_overlay: NativeDebugOverlay | None = None
        self._debug_expirations: dict[tuple[str, str, str], int | None] = {}
        self._debug_lifetimes: dict[tuple[str, str, str], DebugLifetimeMode] = {}
        self._step_index = 0
        self._joint_maps: dict[EntityPath, tuple[int, ...]] = {}
        self._initial_articulation: dict[EntityPath, tuple[Any, Any, Any]] = {}
        self._initial_rigid: dict[EntityPath, tuple[Any, Any]] = {}
        self._initial_deformable: dict[EntityPath, tuple[Any, Any | None]] = {}
        self._origins_cpu = _environment_origins(spec.environments.count, config.environment_spacing_m)
        self._origins: Any | None = None
        self._native_dt = spec.physics.time_step_seconds / spec.physics.substeps
        self._has_fluid = any(entity.kind is EntityKind.PARTICLE_FLUID for entity in spec.entities)
        try:
            self._build()
        except Exception:
            self._close(notify_runtime=False)
            raise

    def _native_debug_overlay(self) -> NativeDebugOverlay:
        if self._debug_overlay is None:
            # Keep the optional render-side plugin out of control-only worlds.
            # Loading it during SimulationContext construction causes a busy loop;
            # loading it in minimal experiences can also fail on absent graph
            # services even though simulation itself is fully usable.
            self._m.carb.get_framework().load_plugins(
                loaded_file_wildcards=["isaacsim.util.debug_draw.plugin"],
                search_paths=[str(self._m.debug_draw_plugin_path)],
            )
            self._debug_draw_interface = self._m.debug_draw.acquire_debug_draw_interface()
            self._debug_overlay = NativeDebugOverlay(
                self._debug_draw_interface,
                lambda primitives: _debug_draw_payload(primitives, self._origins_cpu),
            )
        overlay = self._debug_overlay
        assert overlay is not None
        return overlay

    def _build(self) -> None:
        sim_utils = self._m.sim_utils
        sim_utils.SimulationContext.clear_instance()
        sim_utils.create_new_stage()
        has_fluid = self._has_fluid
        unsupported_mixed = tuple(
            entity.path.value
            for entity in self._spec.entities
            if has_fluid and entity.kind in {EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE}
        )
        if unsupported_mixed:
            raise RuntimeError(
                "this Isaac Lab 3.0 profile cannot combine USD-readback particles with "
                f"tensor-backed deformables in one native world: {unsupported_mixed}"
            )
        sim_cfg = sim_utils.SimulationCfg(
            dt=self._native_dt,
            gravity=self._spec.physics.gravity_m_s2,
            device=self._config.device,
            physics=self._m.PhysxCfg(),
            # PhysX 6 exposes particle state through USD sync, not the removed particle tensor view.
            use_fabric=not has_fluid,
            render_interval=1,
        )
        self._sim = sim_utils.SimulationContext(sim_cfg)
        if any(entity.kind is EntityKind.CAMERA_SENSOR for entity in self._spec.entities):
            # Headless RTX scenes have no viewport headlight. A renderer-neutral
            # camera contract must therefore provide deterministic environment
            # illumination or valid geometry can produce an all-black frame.
            water_surface = self._config.fluid_render_mode == "isosurface"
            light = sim_utils.DomeLightCfg(
                intensity=650.0 if water_surface else 2500.0,
                color=(0.55, 0.68, 0.90) if water_surface else (1.0, 1.0, 1.0),
            )
            light.func("/World/unirobosimDefaultDomeLight", light)
            if water_surface:
                key_light = sim_utils.SphereLightCfg(
                    intensity=5000.0,
                    color=(1.0, 0.82, 0.62),
                    radius=0.45,
                    normalize=True,
                )
                key_light.func(
                    "/World/unirobosimFluidKeyLight",
                    key_light,
                    translation=(-1.4, -2.0, 2.4),
                )
                rim_light = sim_utils.SphereLightCfg(
                    intensity=2600.0,
                    color=(0.42, 0.68, 1.0),
                    radius=0.3,
                    normalize=True,
                )
                rim_light.func(
                    "/World/unirobosimFluidRimLight",
                    rim_light,
                    translation=(1.8, 0.8, 1.8),
                )
        if has_fluid:
            # Isaac Sim 6.0 has no public particle tensor view. Particle state therefore uses
            # PhysX-to-USD readback. Articulations and rigid bodies in these worlds use the raw
            # Omni Physics bridge below because Isaac Lab's high-level assets assume a CUDA tensor
            # frontend while readback deliberately selects a CPU frontend.
            self._sim.set_setting("/physics/suppressReadback", False)
            self._sim.set_setting("/physics/updateToUsd", True)
            self._sim.set_setting("/physics/updateParticlesToUsd", True)
            self._sim.set_setting("/physics/updateVelocitiesToUsd", True)
        for index, origin in enumerate(self._origins_cpu):
            sim_utils.create_prim(f"/World/env_{index}", "Xform", translation=origin)
        ground = sim_utils.GroundPlaneCfg(color=(0.2, 0.23, 0.28))
        ground.func("/World/unirobosimGround", ground)
        for entity in self._spec.entities:
            if entity.kind is EntityKind.ARTICULATION:
                self._author_articulation(entity)
            elif entity.kind is EntityKind.RIGID_BODY:
                self._author_rigid(entity)
            elif entity.kind in {EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE}:
                self._author_deformable(entity)
            elif entity.kind is EntityKind.PARTICLE_FLUID:
                self._author_particle_fluid(entity)
            elif entity.kind is EntityKind.CAMERA_SENSOR:
                self._author_camera(entity)
        self._sim.reset()
        self._initialize_usd_articulations()
        self._initialize_usd_rigids()
        self._origins = self._m.torch.tensor(self._origins_cpu, device=self._sim.device, dtype=self._m.torch.float32)
        self._initialize_articulations()
        self._initialize_rigids()
        self._initialize_deformables()
        self.reset(tuple(range(self._spec.environments.count)))
        if self._cameras:
            self._sim.render()
            for camera in self._cameras.values():
                camera.update(0.0, force_recompute=True)

    def _author_articulation(self, entity: EntitySpec) -> None:
        assert entity.asset_uri is not None
        if self._has_fluid:
            self._author_usd_articulation(entity)
            return
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
                    effort_limit_sim=(
                        dict(zip(entity.joint_names, entity.joint_effort_limits, strict=True))
                        if entity.joint_effort_limits
                        else None
                    ),
                    stiffness=self._config.position_stiffness,
                    damping=self._config.position_damping,
                )
            },
        )
        self._articulations[entity.path] = self._m.Articulation(cfg)

    def _author_usd_articulation(self, entity: EntitySpec) -> None:
        """Author an articulation through USD for particle-readback worlds."""

        assert entity.asset_uri is not None
        name = _native_name(entity.path)
        articulations: list[_UsdArticulation] = []
        for index in range(self._spec.environments.count):
            root = f"/World/env_{index}/{name}"
            cfg = self._m.sim_utils.UsdFileCfg(usd_path=str(entity.asset_uri).removeprefix("file://"))
            cfg.func(
                root,
                cfg,
                translation=entity.pose.position,
                orientation=entity.pose.orientation_xyzw,
            )
            root_prims = self._m.sim_utils.get_all_matching_child_prims(
                root,
                lambda prim: prim.HasAPI(self._m.UsdPhysics.ArticulationRootAPI),
                traverse_instance_prims=False,
            )
            if len(root_prims) != 1:
                raise ValueError(
                    f"articulation asset must contain exactly one ArticulationRootAPI prim; found {len(root_prims)}"
                )
            articulations.append(_UsdArticulation(root_prim=root_prims[0]))
        self._usd_articulations[entity.path] = tuple(articulations)

    def _author_rigid(self, entity: EntitySpec) -> None:
        if entity.box is not None:
            color = entity.box.color_rgba
            cfg = self._m.RigidObjectCfg(
                prim_path=f"/World/env_.*/{_native_name(entity.path)}",
                spawn=self._m.sim_utils.CuboidCfg(
                    size=entity.box.dimensions_m,
                    rigid_props=self._m.sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=self._m.sim_utils.MassPropertiesCfg(mass=entity.box.mass_kg),
                    collision_props=self._m.sim_utils.CollisionPropertiesCfg(),
                    visual_material=self._m.sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color[:3],
                        opacity=color[3],
                    ),
                    physics_material=self._m.sim_utils.RigidBodyMaterialCfg(
                        static_friction=entity.box.static_friction,
                        dynamic_friction=entity.box.dynamic_friction,
                        restitution=entity.box.restitution,
                    ),
                    activate_contact_sensors=True,
                ),
                init_state=self._m.RigidObjectCfg.InitialStateCfg(
                    pos=entity.pose.position,
                    rot=entity.pose.orientation_xyzw,
                ),
            )
            self._rigids[entity.path] = self._m.RigidObject(cfg)
            contact_cfg = self._m.ContactSensorCfg(
                prim_path=f"/World/env_.*/{_native_name(entity.path)}",
                update_period=0.0,
                track_pose=False,
                track_air_time=False,
                track_contact_points=False,
                track_friction_forces=False,
                history_length=0,
                debug_vis=False,
            )
            self._contacts[entity.path] = self._m.ContactSensor(contact_cfg)
            return
        assert entity.asset_uri is not None
        if self._has_fluid:
            self._author_usd_rigid(entity)
            return
        name = _native_name(entity.path)
        cfg = self._m.RigidObjectCfg(
            prim_path=f"/World/env_.*/{name}",
            spawn=self._m.sim_utils.UsdFileCfg(
                usd_path=str(entity.asset_uri).removeprefix("file://"),
                activate_contact_sensors=True,
            ),
            init_state=self._m.RigidObjectCfg.InitialStateCfg(
                pos=entity.pose.position,
                rot=entity.pose.orientation_xyzw,
            ),
        )
        self._rigids[entity.path] = self._m.RigidObject(cfg)
        body_suffix: str | None = None
        for index in range(self._spec.environments.count):
            root = f"/World/env_{index}/{name}"
            rigid_prims = self._m.sim_utils.get_all_matching_child_prims(
                root,
                lambda prim: prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI),
            )
            if len(rigid_prims) != 1:
                raise ValueError(
                    f"rigid asset must contain exactly one UsdPhysics.RigidBodyAPI prim; found {len(rigid_prims)}"
                )
            rigid_prim = rigid_prims[0]
            if "PhysxContactReportAPI" not in rigid_prim.GetAppliedSchemas():
                rigid_prim.AddAppliedSchema("PhysxContactReportAPI")
            suffix = rigid_prim.GetPath().pathString.removeprefix(root)
            if body_suffix is None:
                body_suffix = suffix
            elif suffix != body_suffix:
                raise ValueError("rigid body prim must have the same relative path in every environment")
        assert body_suffix is not None
        contact_cfg = self._m.ContactSensorCfg(
            prim_path=f"/World/env_.*/{name}{body_suffix}",
            update_period=0.0,
            track_pose=False,
            track_air_time=False,
            track_contact_points=False,
            track_friction_forces=False,
            history_length=0,
            debug_vis=False,
        )
        self._contacts[entity.path] = self._m.ContactSensor(contact_cfg)

    def _author_usd_rigid(self, entity: EntitySpec) -> None:
        """Author a rigid through USD for worlds that require particle readback."""

        assert entity.asset_uri is not None
        name = _native_name(entity.path)
        bodies: list[_UsdRigid] = []
        for index in range(self._spec.environments.count):
            root = f"/World/env_{index}/{name}"
            cfg = self._m.sim_utils.UsdFileCfg(
                usd_path=str(entity.asset_uri).removeprefix("file://"),
                activate_contact_sensors=True,
            )
            cfg.func(
                root,
                cfg,
                translation=entity.pose.position,
                orientation=entity.pose.orientation_xyzw,
            )
            rigid_prims = self._m.sim_utils.get_all_matching_child_prims(
                root,
                lambda prim: prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI),
            )
            if len(rigid_prims) != 1:
                raise ValueError(
                    f"rigid asset must contain exactly one UsdPhysics.RigidBodyAPI prim; found {len(rigid_prims)}"
                )
            rigid_prim = rigid_prims[0]
            if "PhysxContactReportAPI" not in rigid_prim.GetAppliedSchemas():
                rigid_prim.AddAppliedSchema("PhysxContactReportAPI")
            bodies.append(_UsdRigid(rigid_prim=rigid_prim))
        self._usd_rigids[entity.path] = tuple(bodies)

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

    def _author_particle_fluid(self, entity: EntitySpec) -> None:
        assert entity.particle_fluid is not None
        fluid = entity.particle_fluid
        local_positions = tuple(
            _transform_position(
                (float(row[0]), float(row[1]), float(row[2])),
                entity.pose.position,
                entity.pose.orientation_xyzw,
            )
            for row in fluid.initial_particle_positions_m.rows()
        )
        local_velocities = tuple(
            _rotate_xyzw((float(row[0]), float(row[1]), float(row[2])), entity.pose.orientation_xyzw)
            for row in fluid.initial_velocities().rows()
        )
        name = _native_name(entity.path)
        stage = self._m.sim_utils.get_current_stage()
        sets: list[_FluidSet] = []
        for environment in range(self._spec.environments.count):
            root = f"/World/env_{environment}/{name}"
            self._m.sim_utils.create_prim(root, "Xform")
            system_path = f"{root}/particle_system"
            system = self._m.PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
            system.CreateSimulationOwnerRel().SetTargets((self._m.Sdf.Path("/physicsScene"),))
            system.CreateParticleSystemEnabledAttr().Set(True)
            system.CreateContactOffsetAttr().Set(fluid.particle_radius_m)
            system.CreateRestOffsetAttr().Set(fluid.particle_radius_m * 0.99)
            system.CreateParticleContactOffsetAttr().Set(fluid.particle_radius_m)
            system.CreateSolidRestOffsetAttr().Set(fluid.particle_radius_m * 0.99)
            system.CreateFluidRestOffsetAttr().Set(fluid.particle_radius_m * 0.6)
            system.CreateSolverPositionIterationCountAttr().Set(8)
            system.CreateGlobalSelfCollisionEnabledAttr().Set(True)
            system.CreateNonParticleCollisionEnabledAttr().Set(True)

            if self._config.fluid_render_mode == "isosurface":
                material = self._author_water_material(stage, root, fluid)
            else:
                material = self._m.UsdShade.Material.Define(stage, f"{root}/pbd_material")
                self._apply_pbd_material(material, fluid)
            binding = self._m.UsdShade.MaterialBindingAPI.Apply(system.GetPrim())
            binding.Bind(
                material,
                bindingStrength=self._m.UsdShade.Tokens.weakerThanDescendants,
                materialPurpose="physics",
            )

            points = self._m.UsdGeom.Points.Define(stage, f"{root}/particles")
            points.CreatePointsAttr().Set(self._m.Vt.Vec3fArray(local_positions))
            points.CreateVelocitiesAttr().Set(self._m.Vt.Vec3fArray(local_velocities))
            points.CreateWidthsAttr().Set(
                self._m.Vt.FloatArray((fluid.particle_radius_m * 2.0,) * fluid.particle_count)
            )
            if self._config.fluid_render_mode == "particles":
                points.CreateDisplayColorPrimvar(self._m.UsdGeom.Tokens.constant).Set(
                    self._m.Vt.Vec3fArray(((0.1, 0.45, 1.0),))
                )
            set_api = self._m.PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
            # PhysxParticleSetAPI derives from PhysxParticleAPI; constructing the base view from
            # the applied set schema matches NVIDIA's particleUtils.configure_particle_set path.
            particle_api = self._m.PhysxSchema.PhysxParticleAPI(set_api)
            particle_api.CreateParticleSystemRel().SetTargets((self._m.Sdf.Path(system_path),))
            particle_api.CreateSelfCollisionAttr().Set(True)
            particle_api.CreateParticleGroupAttr().Set(environment)
            set_api.CreateFluidAttr().Set(True)
            mass_api = self._m.UsdPhysics.MassAPI.Apply(points.GetPrim())
            mass_api.CreateMassAttr().Set(fluid.resolved_particle_mass_kg * fluid.particle_count)
            mass_api.CreateDensityAttr().Set(fluid.rest_density_kg_m3)
            if self._config.fluid_render_mode == "isosurface":
                self._author_fluid_isosurface(system, material, fluid)
            sets.append(_FluidSet(points, local_positions, local_velocities))
        self._fluids[entity.path] = tuple(sets)

    def _apply_pbd_material(self, material: Any, fluid: Any) -> None:
        material_api = self._m.PhysxSchema.PhysxPBDMaterialAPI.Apply(material.GetPrim())
        material_api.CreateDensityAttr().Set(fluid.rest_density_kg_m3)
        material_api.CreateViscosityAttr().Set(fluid.dynamic_viscosity_pa_s)
        material_api.CreateSurfaceTensionAttr().Set(fluid.surface_tension_n_m)

    def _author_water_material(self, stage: Any, root: str, fluid: Any) -> Any:
        """Create one material that carries both water rendering and PBD properties."""

        from omni.usd.commands import CreateMdlMaterialPrimCommand  # type: ignore[import-not-found]

        material_path = f"{root}/water_surface_material"
        CreateMdlMaterialPrimCommand(
            mtl_url="OmniSurfacePresets.mdl",
            mtl_name="OmniSurface_DeepWater",
            mtl_path=material_path,
            stage=stage,
            select_new_prim=False,
        ).do()
        water_material = self._m.UsdShade.Material.Get(stage, material_path)
        if not water_material.GetPrim().IsValid():
            raise RuntimeError(f"Isaac Sim did not create the clear-water MDL material at {material_path}")
        self._apply_pbd_material(water_material, fluid)
        return water_material

    def _author_fluid_isosurface(self, system: Any, water_material: Any, fluid: Any) -> None:
        """Add opt-in render-only surface reconstruction with inherited water rendering."""

        system_prim = system.GetPrim()
        smoothing = self._m.PhysxSchema.PhysxParticleSmoothingAPI.Apply(system_prim)
        smoothing.CreateParticleSmoothingEnabledAttr().Set(True)
        smoothing.CreateStrengthAttr().Set(0.5)

        anisotropy = self._m.PhysxSchema.PhysxParticleAnisotropyAPI.Apply(system_prim)
        anisotropy.CreateParticleAnisotropyEnabledAttr().Set(True)
        anisotropy.CreateScaleAttr().Set(5.0)
        anisotropy.CreateMinAttr().Set(1.0)
        anisotropy.CreateMaxAttr().Set(2.0)

        isosurface = self._m.PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(system_prim)
        isosurface.CreateIsosurfaceEnabledAttr().Set(True)
        isosurface.CreateMaxVerticesAttr().Set(1_048_576)
        isosurface.CreateMaxTrianglesAttr().Set(2_097_152)
        isosurface.CreateMaxSubgridsAttr().Set(4_096)
        fluid_rest_offset = fluid.particle_radius_m * 0.6
        grid_spacing = fluid_rest_offset * 1.5
        isosurface.CreateGridSpacingAttr().Set(grid_spacing)
        isosurface.CreateSurfaceDistanceAttr().Set(fluid_rest_offset * 1.6)
        isosurface.CreateGridFilteringPassesAttr().Set("")
        isosurface.CreateGridSmoothingRadiusAttr().Set(fluid_rest_offset * 2.0)
        isosurface.CreateNumMeshSmoothingPassesAttr().Set(2)
        isosurface.CreateNumMeshNormalSmoothingPassesAttr().Set(4)
        self._m.UsdGeom.PrimvarsAPI(system).CreatePrimvar(
            "doNotCastShadows",
            self._m.Sdf.ValueTypeNames.Bool,
        ).Set(True)

        binding = self._m.UsdShade.MaterialBindingAPI.Apply(system_prim)
        binding.Bind(
            water_material,
            bindingStrength=self._m.UsdShade.Tokens.strongerThanDescendants,
        )

    def _author_camera(self, entity: EntitySpec) -> None:
        assert entity.camera is not None
        if not self._config.enable_cameras or not self._config.render:
            raise RuntimeError("camera entities require enable_cameras=True and render=True")
        camera = entity.camera
        aperture = 20.955
        focal_length = aperture / (2.0 * math.tan(math.radians(camera.horizontal_fov_degrees) / 2.0))
        data_types = [
            "rgb" if modality is CameraModality.RGB else "distance_to_camera" for modality in camera.modalities
        ]
        cfg = self._m.CameraCfg(
            prim_path=f"/World/env_.*/{_native_name(entity.path)}",
            update_period=0.0,
            height=camera.height_px,
            width=camera.width_px,
            data_types=data_types,
            depth_clipping_behavior="zero",
            offset=self._m.CameraCfg.OffsetCfg(
                pos=entity.pose.position,
                rot=entity.pose.orientation_xyzw,
                # UniRoboSim camera poses use the OpenGL optical frame: -Z forward,
                # +Y up. Isaac Lab performs the conversion to its native camera
                # frame when the convention is declared explicitly.
                convention="opengl",
            ),
            spawn=self._m.sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=aperture,
                clipping_range=(camera.near_plane_m, camera.far_plane_m),
            ),
        )
        self._cameras[entity.path] = self._m.Camera(cfg)

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

    def _usd_simulation_view(self) -> Any:
        if self._usd_tensor_view is None:
            self._usd_tensor_view = self._m.physics_tensors.create_simulation_view("torch")
            self._usd_tensor_view.set_subspace_roots("/")
        return self._usd_tensor_view

    def _initialize_usd_articulations(self) -> None:
        if not self._usd_articulations:
            return
        tensor_view = self._usd_simulation_view()
        for path, articulations in self._usd_articulations.items():
            entity = next(item for item in self._spec.entities if item.path == path)
            expected_paths = tuple(item.root_prim.GetPath().pathString for item in articulations)
            view = tensor_view.create_articulation_view(list(expected_paths))
            if view.count != self._spec.environments.count or tuple(view.prim_paths) != expected_paths:
                raise RuntimeError(
                    f"USD articulation view for {path.value} did not preserve environment order; "
                    f"expected={expected_paths}, actual={tuple(view.prim_paths)}"
                )
            native_names = tuple(view.shared_metatype.dof_names)
            if set(native_names) != set(entity.joint_names) or len(native_names) != len(entity.joint_names):
                raise ValueError(
                    f"joint names for {path.value} do not exactly match the USD; "
                    f"declared={entity.joint_names}, native={native_names}"
                )
            joint_map = tuple(native_names.index(name) for name in entity.joint_names)
            self._joint_maps[path] = joint_map
            root_pose = view.get_root_transforms().clone()
            root_velocity = view.get_root_velocities().clone()
            positions = view.get_dof_positions().clone()
            for public_index, native_index in enumerate(joint_map):
                positions[:, native_index] = entity.initial_joint_positions[public_index]
            velocities = self._m.torch.zeros_like(positions)
            self._initial_usd_articulation[path] = (root_pose, root_velocity, positions, velocities)
            self._usd_articulation_views[path] = view

    def _initialize_usd_rigids(self) -> None:
        if not self._usd_rigids:
            return
        assert self._sim is not None
        tensor_view = self._usd_simulation_view()
        for path, bodies in self._usd_rigids.items():
            expected_paths = tuple(body.rigid_prim.GetPath().pathString for body in bodies)
            view = tensor_view.create_rigid_body_view(list(expected_paths))
            if view.count != self._spec.environments.count or tuple(view.prim_paths) != expected_paths:
                raise RuntimeError(
                    f"USD rigid view for {path.value} did not preserve environment order; "
                    f"expected={expected_paths}, actual={tuple(view.prim_paths)}"
                )
            transforms = view.get_transforms().clone()
            velocities = view.get_velocities().clone()
            self._usd_rigid_views[path] = view
            self._initial_usd_rigid[path] = (transforms, velocities)
            self._usd_rigid_wrenches[path] = (
                self._m.torch.zeros((view.count, 3), device=transforms.device, dtype=self._m.torch.float32),
                self._m.torch.zeros((view.count, 3), device=transforms.device, dtype=self._m.torch.float32),
            )

    def _initialize_rigids(self) -> None:
        assert self._sim is not None
        for path, asset in self._rigids.items():
            root_pose = asset.data.default_root_pose.torch.clone()
            assert self._origins is not None
            root_pose[:, :3] += self._origins
            root_velocity = asset.data.default_root_vel.torch.clone()
            self._initial_rigid[path] = (root_pose, root_velocity)

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
        for path, view in self._usd_articulation_views.items():
            root_pose, root_velocity, positions, velocities = self._initial_usd_articulation[path]
            indices = self._m.torch.tensor(
                environment_indices,
                device=positions.device,
                dtype=self._m.torch.int64,
            )
            zeros = self._m.torch.zeros_like(positions[indices])
            view.set_root_transforms(root_pose[indices], indices)
            view.set_root_velocities(root_velocity[indices], indices)
            view.set_dof_positions(positions[indices], indices)
            view.set_dof_velocities(velocities[indices], indices)
            view.set_dof_position_targets(positions[indices], indices)
            view.set_dof_velocity_targets(zeros, indices)
            view.set_dof_actuation_forces(zeros, indices)
            stiffness = view.get_dof_stiffnesses().clone()
            damping = view.get_dof_dampings().clone()
            joint_ids = self._joint_maps[path]
            for environment in environment_indices:
                stiffness[environment, list(joint_ids)] = self._config.position_stiffness
                damping[environment, list(joint_ids)] = self._config.position_damping
            view.set_dof_stiffnesses(stiffness[indices], indices)
            view.set_dof_dampings(damping[indices], indices)
        for path, asset in self._rigids.items():
            root_pose, root_velocity = self._initial_rigid[path]
            asset.reset(env_ids=env_ids)
            asset.write_root_pose_to_sim_index(root_pose=root_pose[env_ids], env_ids=env_ids)
            asset.write_root_link_velocity_to_sim_index(root_velocity=root_velocity[env_ids], env_ids=env_ids)
            self._contacts[path].reset(env_ids=env_ids)
        for path, view in self._usd_rigid_views.items():
            transforms, velocities = self._initial_usd_rigid[path]
            indices = self._m.torch.tensor(
                environment_indices,
                device=transforms.device,
                dtype=self._m.torch.int64,
            )
            view.set_transforms(transforms[indices], indices)
            view.set_velocities(velocities[indices], indices)
            forces, torques = self._usd_rigid_wrenches[path]
            forces[indices] = 0.0
            torques[indices] = 0.0
        for path, asset in self._deformables.items():
            state, target = self._initial_deformable[path]
            asset.write_nodal_state_to_sim_index(state[env_ids], env_ids=env_ids)
            if target is not None:
                asset.write_nodal_kinematic_target_to_sim_index(target[env_ids], env_ids=env_ids)
            asset.reset(env_ids=env_ids)
        for sets in self._fluids.values():
            for environment in environment_indices:
                fluid_set = sets[environment]
                fluid_set.points.GetPointsAttr().Set(self._m.Vt.Vec3fArray(fluid_set.initial_positions))
                fluid_set.points.GetVelocitiesAttr().Set(self._m.Vt.Vec3fArray(fluid_set.initial_velocities))
        for camera in self._cameras.values():
            camera.reset(env_ids=env_ids)
        reset_debug_keys = tuple(
            key for key, mode in self._debug_lifetimes.items() if mode is not DebugLifetimeMode.MANUAL
        )
        if reset_debug_keys:
            self._remove_debug_keys(reset_debug_keys)
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
        if path in self._usd_articulation_views:
            self._apply_usd_articulation(path, mode, targets, environment_indices, degree_of_freedom_indices)
            return
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

    def _apply_usd_articulation(
        self,
        path: EntityPath,
        mode: CommandMode,
        targets: Matrix,
        environment_indices: tuple[int, ...],
        degree_of_freedom_indices: tuple[int, ...],
    ) -> None:
        view = self._usd_articulation_views[path]
        joint_ids = tuple(self._joint_maps[path][index] for index in degree_of_freedom_indices)
        positions = view.get_dof_position_targets().clone()
        velocities = view.get_dof_velocity_targets().clone()
        efforts = view.get_dof_actuation_forces().clone()
        stiffness = view.get_dof_stiffnesses().clone()
        damping = view.get_dof_dampings().clone()
        for row_index, environment in enumerate(environment_indices):
            for column_index, joint in enumerate(joint_ids):
                target = float(targets[row_index][column_index])
                if mode is CommandMode.POSITION:
                    positions[environment, joint] = target
                    velocities[environment, joint] = 0.0
                    efforts[environment, joint] = 0.0
                    stiffness[environment, joint] = self._config.position_stiffness
                    damping[environment, joint] = self._config.position_damping
                elif mode is CommandMode.VELOCITY:
                    velocities[environment, joint] = target
                    efforts[environment, joint] = 0.0
                    stiffness[environment, joint] = 0.0
                    damping[environment, joint] = self._config.velocity_damping
                else:
                    efforts[environment, joint] = target
                    stiffness[environment, joint] = 0.0
                    damping[environment, joint] = 0.0
        indices = self._m.torch.tensor(
            environment_indices,
            device=positions.device,
            dtype=self._m.torch.int64,
        )
        view.set_dof_position_targets(positions[indices], indices)
        view.set_dof_velocity_targets(velocities[indices], indices)
        view.set_dof_actuation_forces(efforts[indices], indices)
        view.set_dof_stiffnesses(stiffness[indices], indices)
        view.set_dof_dampings(damping[indices], indices)

    def read_articulation(self, path: EntityPath) -> tuple[Matrix, Matrix]:
        if path in self._usd_articulation_views:
            view = self._usd_articulation_views[path]
            joint_map = list(self._joint_maps[path])
            positions = view.get_dof_positions()[:, joint_map].detach().cpu().tolist()
            velocities = view.get_dof_velocities()[:, joint_map].detach().cpu().tolist()
            return (
                tuple(tuple(float(value) for value in row) for row in positions),
                tuple(tuple(float(value) for value in row) for row in velocities),
            )
        asset = self._articulations[path]
        joint_map = list(self._joint_maps[path])
        positions = asset.data.joint_pos.torch[:, joint_map].detach().cpu().tolist()
        velocities = asset.data.joint_vel.torch[:, joint_map].detach().cpu().tolist()
        return (
            tuple(tuple(float(value) for value in row) for row in positions),
            tuple(tuple(float(value) for value in row) for row in velocities),
        )

    def apply_rigid_body_wrench(
        self,
        path: EntityPath,
        forces_n: Matrix,
        torques_n_m: Matrix,
        environment_indices: tuple[int, ...],
    ) -> None:
        if path in self._usd_rigid_views:
            forces, torques = self._usd_rigid_wrenches[path]
            for row_index, environment in enumerate(environment_indices):
                forces[environment] = self._m.torch.tensor(
                    forces_n[row_index], device=forces.device, dtype=forces.dtype
                )
                torques[environment] = self._m.torch.tensor(
                    torques_n_m[row_index], device=torques.device, dtype=torques.dtype
                )
            return
        asset = self._rigids[path]
        assert self._sim is not None
        env_ids = self._m.torch.tensor(environment_indices, device=self._sim.device, dtype=self._m.torch.int64)
        forces = self._m.torch.tensor(forces_n, device=self._sim.device, dtype=self._m.torch.float32).unsqueeze(1)
        torques = self._m.torch.tensor(torques_n_m, device=self._sim.device, dtype=self._m.torch.float32).unsqueeze(1)
        asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            env_ids=env_ids,
            is_global=True,
        )

    def read_rigid_body(self, path: EntityPath) -> tuple[Matrix, Matrix, Matrix, Matrix]:
        if path in self._usd_rigid_views:
            view = self._usd_rigid_views[path]
            pose = view.get_transforms().clone()
            velocity = view.get_velocities()
            origins = self._m.torch.tensor(self._origins_cpu, device=pose.device, dtype=pose.dtype)
            pose[:, :3] -= origins
            positions = pose[:, :3].detach().cpu().tolist()
            orientations = pose[:, 3:].detach().cpu().tolist()
            linear_velocities = velocity[:, :3].detach().cpu().tolist()
            angular_velocities = velocity[:, 3:].detach().cpu().tolist()
            return (
                tuple(tuple(float(value) for value in row) for row in positions),
                tuple(tuple(float(value) for value in row) for row in orientations),
                tuple(tuple(float(value) for value in row) for row in linear_velocities),
                tuple(tuple(float(value) for value in row) for row in angular_velocities),
            )
        asset = self._rigids[path]
        assert self._origins is not None
        pose = asset.data.root_link_pose_w.torch.clone()
        pose[:, :3] -= self._origins
        velocity = asset.data.root_link_vel_w.torch
        positions = pose[:, :3].detach().cpu().tolist()
        orientations = pose[:, 3:].detach().cpu().tolist()
        linear_velocities = velocity[:, :3].detach().cpu().tolist()
        angular_velocities = velocity[:, 3:].detach().cpu().tolist()
        return tuple(
            tuple(tuple(float(value) for value in row) for row in values)
            for values in (positions, orientations, linear_velocities, angular_velocities)
        )  # type: ignore[return-value]

    def set_rigid_body_pose(
        self,
        path: EntityPath,
        position_m: Vector3,
        orientation_xyzw: tuple[float, float, float, float],
        environment_index: int,
    ) -> None:
        if path in self._usd_rigid_views:
            view = self._usd_rigid_views[path]
            transforms = view.get_transforms().clone()
            origin = self._m.torch.tensor(
                self._origins_cpu[environment_index], device=transforms.device, dtype=transforms.dtype
            )
            transforms[environment_index, :3] = (
                self._m.torch.tensor(position_m, device=transforms.device, dtype=transforms.dtype) + origin
            )
            transforms[environment_index, 3:] = self._m.torch.tensor(
                orientation_xyzw, device=transforms.device, dtype=transforms.dtype
            )
            indices = self._m.torch.tensor((environment_index,), device=transforms.device, dtype=self._m.torch.int64)
            view.set_transforms(transforms[indices], indices)
            view.set_velocities(self._m.torch.zeros((1, 6), device=transforms.device, dtype=transforms.dtype), indices)
        else:
            asset = self._rigids[path]
            assert self._sim is not None and self._origins is not None
            pose = self._m.torch.tensor(
                ((*position_m, *orientation_xyzw),), device=self._sim.device, dtype=self._m.torch.float32
            )
            pose[:, :3] += self._origins[environment_index : environment_index + 1]
            env_ids = self._m.torch.tensor((environment_index,), device=self._sim.device, dtype=self._m.torch.int64)
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            asset.write_root_link_velocity_to_sim_index(
                root_velocity=self._m.torch.zeros((1, 6), device=self._sim.device, dtype=self._m.torch.float32),
                env_ids=env_ids,
            )
        assert self._sim is not None
        self._sim.forward()
        self._update_assets(0.0)

    def read_contact(self, path: EntityPath) -> Matrix:
        if path in self._usd_rigids:
            raise RuntimeError("contact-force readback is unavailable for rigid bodies in particle-fluid worlds")
        net_forces = self._contacts[path].data.net_forces_w
        if net_forces is None:
            raise RuntimeError(f"contact sensor for {path.value} did not expose net force data")
        values = net_forces.torch[:, 0, :].detach().cpu().tolist()
        return tuple(tuple(float(value) for value in row) for row in values)

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

    def apply_particle_fluid(
        self,
        path: EntityPath,
        mode: PointCommandMode,
        targets: PointBatch,
        environment_indices: tuple[int, ...],
        particle_indices: tuple[int, ...],
    ) -> None:
        sets = self._fluids[path]
        for row_index, environment in enumerate(environment_indices):
            fluid_set = sets[environment]
            if mode is PointCommandMode.POSITION:
                values = [
                    tuple(float(component) for component in point) for point in fluid_set.points.GetPointsAttr().Get()
                ]
                for column_index, particle in enumerate(particle_indices):
                    values[particle] = targets[row_index][column_index]
                fluid_set.points.GetPointsAttr().Set(self._m.Vt.Vec3fArray(values))
            elif mode is PointCommandMode.VELOCITY:
                velocities = [
                    tuple(float(component) for component in point)
                    for point in fluid_set.points.GetVelocitiesAttr().Get()
                ]
                for column_index, particle in enumerate(particle_indices):
                    velocities[particle] = targets[row_index][column_index]
                fluid_set.points.GetVelocitiesAttr().Set(self._m.Vt.Vec3fArray(velocities))
            else:
                raise RuntimeError("native particle force commands are unsupported")
        assert self._sim is not None
        self._sim.forward()

    def read_particle_fluid(self, path: EntityPath) -> tuple[PointBatch, PointBatch]:
        positions: list[tuple[tuple[float, float, float], ...]] = []
        velocities: list[tuple[tuple[float, float, float], ...]] = []
        for fluid_set in self._fluids[path]:
            simulation_api = self._m.PhysxSchema.PhysxParticleSetAPI(fluid_set.points.GetPrim())
            simulation_points = simulation_api.GetSimulationPointsAttr().Get()
            point_values = simulation_points if simulation_points else fluid_set.points.GetPointsAttr().Get()
            positions.append(tuple((float(point[0]), float(point[1]), float(point[2])) for point in point_values))
            velocities.append(
                tuple(
                    (float(point[0]), float(point[1]), float(point[2]))
                    for point in fluid_set.points.GetVelocitiesAttr().Get()
                )
            )
        return tuple(positions), tuple(velocities)

    def read_sensor(self, path: EntityPath) -> NativeSensorSample:
        camera = self._cameras[path]
        entity = next(item for item in self._spec.entities if item.path == path)
        assert entity.camera is not None
        assert self._sim is not None
        self._sim.render()
        camera.update(0.0, force_recompute=True)
        channels = []
        for modality in entity.camera.modalities:
            native_name = "rgb" if modality is CameraModality.RGB else "distance_to_camera"
            value = camera.data.output[native_name]
            tensor = getattr(value, "torch", value)
            channel_values: tuple[float | int, ...]
            if modality is CameraModality.RGB:
                tensor = tensor.to(dtype=self._m.torch.uint8)
                shape = tuple(int(size) for size in tensor.shape)
                channel_values = tuple(int(item) for item in tensor.detach().cpu().reshape(-1).tolist())
            else:
                tensor = tensor[..., 0]
                valid = self._m.torch.isfinite(tensor)
                valid &= tensor >= entity.camera.near_plane_m
                valid &= tensor <= entity.camera.far_plane_m
                tensor = self._m.torch.where(valid, tensor, self._m.torch.zeros_like(tensor))
                tensor = tensor.to(dtype=self._m.torch.float32)
                shape = tuple(int(size) for size in tensor.shape)
                channel_values = tuple(float(item) for item in tensor.detach().cpu().reshape(-1).tolist())
            channels.append((modality, shape, channel_values))
        return tuple(channels)

    def publish_debug(self, batch: DebugBatch) -> NativeDebugReport:
        for primitive in batch.primitives:
            key = primitive.key
            if primitive.lifetime.mode is DebugLifetimeMode.FRAME:
                expiration = self._step_index + 1
            elif primitive.lifetime.mode is DebugLifetimeMode.STEPS:
                assert primitive.lifetime.step_count is not None
                expiration = self._step_index + primitive.lifetime.step_count
            else:
                expiration = None
            self._debug_expirations[key] = expiration
            self._debug_lifetimes[key] = primitive.lifetime.mode
        overlay = self._native_debug_overlay()
        overlay.upsert(batch.primitives)
        self._flush_debug_render()
        return len(batch.primitives), 0, overlay.active_count

    def _flush_debug_render(self) -> None:
        if self._config.render:
            assert self._sim is not None
            self._sim.render()

    def _remove_debug_keys(self, keys: tuple[tuple[str, str, str], ...]) -> None:
        for key in keys:
            self._debug_expirations.pop(key)
            self._debug_lifetimes.pop(key)
        if self._native_debug_overlay().remove(keys):
            self._flush_debug_render()

    def clear_debug(self, layer: str | None, group: str | None, primitive_id: str | None) -> int:
        keys = tuple(
            key
            for key in self._native_debug_overlay().keys
            if (layer is None or key[0] == layer)
            and (group is None or key[1] == group)
            and (primitive_id is None or key[2] == primitive_id)
        )
        if keys:
            self._remove_debug_keys(keys)
        return len(keys)

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
                for path, view in self._usd_rigid_views.items():
                    forces, torques = self._usd_rigid_wrenches[path]
                    indices = self._m.torch.arange(view.count, device=forces.device, dtype=self._m.torch.int64)
                    view.apply_forces_and_torques_at_position(forces, torques, None, indices, True)
                for asset in self._articulations.values():
                    asset.write_data_to_sim()
                for asset in self._rigids.values():
                    asset.write_data_to_sim()
                for asset in self._deformables.values():
                    asset.write_data_to_sim()
                self._sim.step(render=self._config.render and self._config.render_on_step)
                self._update_assets(self._native_dt)
            self._step_index += 1
            expired = tuple(
                key
                for key, expiration in self._debug_expirations.items()
                if expiration is not None and expiration <= self._step_index
            )
            if expired:
                self._remove_debug_keys(expired)

    def _update_assets(self, dt: float) -> None:
        for asset in self._articulations.values():
            asset.update(dt)
        for asset in self._rigids.values():
            asset.update(dt)
        for sensor in self._contacts.values():
            sensor.update(dt)
        for asset in self._deformables.values():
            asset.update(dt)

    def _close(self, *, notify_runtime: bool) -> None:
        if self._closed:
            return
        self._closed = True
        sim = self._sim
        self._sim = None
        self._articulations.clear()
        self._usd_articulations.clear()
        self._usd_articulation_views.clear()
        self._initial_usd_articulation.clear()
        self._rigids.clear()
        self._usd_rigids.clear()
        self._usd_rigid_views.clear()
        self._initial_usd_rigid.clear()
        self._usd_rigid_wrenches.clear()
        self._usd_tensor_view = None
        self._contacts.clear()
        self._deformables.clear()
        self._fluids.clear()
        self._cameras.clear()
        self._debug_expirations.clear()
        self._debug_lifetimes.clear()
        if self._debug_overlay is not None:
            self._debug_overlay.close()
            self._debug_overlay = None
        if self._debug_draw_interface is not None:
            self._m.debug_draw.release_debug_draw_interface(self._debug_draw_interface)
            self._debug_draw_interface = None
        self._initial_articulation.clear()
        self._initial_rigid.clear()
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
