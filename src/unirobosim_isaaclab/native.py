"""Isaac Lab 3.0 native runtime.

This module is imported only after the lightweight compatibility probe succeeds. AppLauncher is
constructed before importing simulation, torch, Omni, or USD modules.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import unquote, urlparse

from unirobosim import (
    CameraModality,
    CommandMode,
    DebugBatch,
    DebugLifetimeMode,
    DebugMeshResource,
    DebugMeshStyle,
    DebugPrimitive,
    DebugPrimitiveKind,
    EntityKind,
    EntityPath,
    EntitySpec,
    KinematicTarget,
    PackedFloat32Array,
    PointCommandMode,
    Pose,
    WorldSpec,
)

from .config import _ANTI_ALIASING_MODES, IsaacLabAdapterConfig
from .native_debug import NativeDebugOverlay, NativeDebugPayload
from .native_protocols import (
    Matrix,
    NativeArticulationCommand,
    NativeCameraCalibration,
    NativeDebugReport,
    NativeEntityPrimState,
    NativeKinematicState,
    NativePhysicsDiagnostics,
    NativeRenderArticulationState,
    NativeRenderParticleFluidState,
    NativeRenderRigidBodyState,
    NativeRenderStateFrame,
    NativeSensorBatch,
    NativeSensorSample,
    PointBatch,
)
from .physics_activation import (
    DynamicRigidBodyCandidate,
    PhysicsActivationController,
    build_physics_activation_controller,
)

Vector3 = tuple[float, float, float]
Segment = tuple[Vector3, Vector3]
Color = tuple[float, float, float, float]

_COMPOSITE_UNBOUND_RIGID_MODE_KEY = "composite_unbound_rigid_mode"
_COMPOSITE_UNBOUND_RIGID_MODES = frozenset({"authored", "kinematic", "static"})
_POSITION_STIFFNESS_FALLBACK = 1000.0
_POSITION_DAMPING_FALLBACK = 100.0
_DEFAULT_CAMERA_RENDERER = "RaytracedLighting"
_ISOSURFACE_CAMERA_RENDERER = "RealTimePathTracing"
_DEFAULT_GPU_MAX_NUM_PARTITIONS = 8
_SMALL_WORLD_GPU_MAX_NUM_PARTITIONS = 1
_SMALL_WORLD_MAX_ARTICULATION_DOFS = 16
_SMALL_WORLD_MAX_DYNAMIC_ENTITIES = 16


def _resolved_gpu_max_num_partitions(config: IsaacLabAdapterConfig, spec: WorldSpec) -> int:
    """Select GPU pipeline partitions without changing solver/contact fidelity.

    A single small rigid/articulation world pays more kernel-partition overhead
    than it gains from parallel partitioning.  Multi-environment worlds,
    particle/deformable worlds, and larger articulations retain Isaac Lab's
    default because they provide enough parallel work to benefit from it.
    """

    explicit = config.gpu_max_num_partitions
    if explicit is not None:
        return explicit
    if spec.environments.count != 1:
        return _DEFAULT_GPU_MAX_NUM_PARTITIONS
    if any(
        entity.kind
        in {
            EntityKind.PARTICLE_FLUID,
            EntityKind.SURFACE_DEFORMABLE,
            EntityKind.VOLUME_DEFORMABLE,
        }
        for entity in spec.entities
    ):
        return _DEFAULT_GPU_MAX_NUM_PARTITIONS
    articulation_dofs = sum(
        len(entity.joint_names)
        for entity in spec.entities
        if entity.kind is EntityKind.ARTICULATION
    )
    dynamic_entities = sum(
        entity.kind in {EntityKind.ARTICULATION, EntityKind.RIGID_BODY}
        for entity in spec.entities
    )
    if (
        articulation_dofs <= _SMALL_WORLD_MAX_ARTICULATION_DOFS
        and dynamic_entities <= _SMALL_WORLD_MAX_DYNAMIC_ENTITIES
    ):
        return _SMALL_WORLD_GPU_MAX_NUM_PARTITIONS
    return _DEFAULT_GPU_MAX_NUM_PARTITIONS


def _normalized_quaternion_xyzw(rotation: Any) -> tuple[float, float, float, float]:
    imaginary = rotation.GetImaginary()
    components = (
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(rotation.GetReal()),
    )
    norm = math.sqrt(sum(component * component for component in components))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("USD transform contains an invalid rotation quaternion")
    normalized = tuple(component / norm for component in components)
    return (normalized[0], normalized[1], normalized[2], normalized[3])


def _pose_from_world_matrix(matrix: Any, origin: tuple[float, float, float]) -> Pose:
    """Extract a pose after removing authored scale and shear from a USD matrix."""

    translation = matrix.ExtractTranslation()
    pose_matrix = matrix.RemoveScaleShear()
    orientation = _normalized_quaternion_xyzw(pose_matrix.ExtractRotationQuat())
    return Pose(
        (
            float(translation[0]) - float(origin[0]),
            float(translation[1]) - float(origin[1]),
            float(translation[2]) - float(origin[2]),
        ),
        orientation,
    )


def _position_command_gains(
    stiffness: Any,
    damping: Any,
    *,
    fallback_authored_zero: bool,
    fallback_authored_damping: bool,
) -> tuple[Any, Any]:
    """Resolve cached per-joint gains for an explicit position command.

    Authored zero stiffness usually denotes a passive or velocity-driven joint.
    It is preserved during initialization/reset, but a caller explicitly issuing
    a position command needs a usable drive.  Legacy numeric configuration is an
    intentional global override and therefore never receives this fallback.
    """

    if not fallback_authored_zero:
        return stiffness, damping
    zero_stiffness = stiffness <= 0.0
    stiffness = stiffness.masked_fill(zero_stiffness, _POSITION_STIFFNESS_FALLBACK)
    if fallback_authored_damping:
        damping = damping.masked_fill(zero_stiffness, _POSITION_DAMPING_FALLBACK)
    return stiffness, damping


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
    # The pip SDK places ``extscache`` below the ``isaacsim`` package root,
    # whereas the official NGC image places the package at
    # ``<root>/python_packages/isaacsim`` and the cache at ``<root>/extscache``.
    # Search only this bounded ancestor chain and require the exact extension;
    # never scan arbitrary SDK or filesystem roots.
    search_roots = (isaacsim_root, *isaacsim_root.parents[:3])
    extension = next(
        (
            candidate
            for root in search_roots
            for candidate in sorted((root / "extscache").glob("isaacsim.util.debug_draw-*"))
            if (candidate / "bin").is_dir() and (candidate / "isaacsim").is_dir()
        ),
        None,
    )
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


def _triangle_edges(resource: DebugMeshResource) -> tuple[tuple[int, int], ...]:
    """Return deterministic unique undirected edges for one immutable mesh."""

    edges: set[tuple[int, int]] = set()
    values = resource.triangle_indices.values
    for offset in range(0, len(values), 3):
        triangle = (int(values[offset]), int(values[offset + 1]), int(values[offset + 2]))
        for left, right in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edges.add((left, right) if left < right else (right, left))
    return tuple(sorted(edges))


def _mesh_instance_rows(
    primitive: DebugPrimitive,
    origins: tuple[Vector3, ...],
) -> tuple[tuple[Vector3, tuple[float, float, float, float], Vector3], ...]:
    """Lower environment-local TRS rows into global positions and portable XYZW rotations."""

    if primitive.kind is not DebugPrimitiveKind.MESH_INSTANCE:
        return ()
    nested = primitive.geometry_m.nested()
    return tuple(
        (
            _offset(row[:3], origins[environment]),
            cast(tuple[float, float, float, float], tuple(float(value) for value in row[3:7])),
            cast(Vector3, tuple(float(value) for value in row[7:10])),
        )
        for row_index, environment in enumerate(primitive.environment_indices)
        for row in nested[row_index]
    )


@dataclass
class _FluidSet:
    system: Any
    points: Any
    initial_positions: tuple[tuple[float, float, float], ...]
    initial_velocities: tuple[tuple[float, float, float], ...]
    render_state_visualization_enabled: bool = False


@dataclass
class _UsdRigid:
    rigid_prim: Any


@dataclass
class _UsdArticulation:
    root_prim: Any


@dataclass(frozen=True, slots=True)
class _RuntimeAttachment:
    attachment_id: str
    environment_index: int
    parent_path: EntityPath
    parent_link_name: str | None
    child_path: EntityPath
    child_link_name: str | None
    parent_T_child: Pose
    joint_prim_path: str


@dataclass
class _CompositeRigidState:
    view: Any
    initial_transforms: Any
    initial_velocities: Any
    environment_by_index: tuple[int, ...]
    kinematic: bool


@dataclass
class _CompositeArticulationState:
    view: Any
    initial_root_transforms: Any
    initial_root_velocities: Any
    initial_dof_positions: Any
    initial_dof_velocities: Any
    initial_position_targets: Any
    initial_velocity_targets: Any
    initial_actuation_forces: Any
    initial_stiffnesses: Any
    initial_dampings: Any


@dataclass
class _MountedCamera:
    parent_path: EntityPath
    body_name: str
    body_suffix: str
    local_pose: Pose
    body_index: int | None = None
    raw_body_view: Any | None = None


def _is_kinematic_rigid(prim: Any, usd_physics: Any) -> bool:
    """Read the authored rigid-body motion mode without importing USD at module load time."""

    return bool(usd_physics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())


def _composite_unbound_rigid_mode(entity: EntitySpec) -> str:
    """Return the exact opt-in policy for otherwise unbound composite rigid bodies."""

    mode = entity.metadata.get(_COMPOSITE_UNBOUND_RIGID_MODE_KEY, "authored")
    if type(mode) is not str or mode not in _COMPOSITE_UNBOUND_RIGID_MODES:
        raise ValueError(
            f"composite scene {entity.path.value} metadata {_COMPOSITE_UNBOUND_RIGID_MODE_KEY!r} "
            "must be exactly 'authored', 'kinematic', or 'static'"
        )
    return mode


def _write_usd_rigid_state(
    view: Any,
    transforms: Any,
    velocities: Any,
    indices: Any,
    *,
    kinematic: bool,
) -> None:
    """Write a raw USD rigid state while respecting PhysX kinematic semantics."""

    view.set_transforms(transforms, indices)
    if not kinematic:
        view.set_velocities(velocities, indices)


def _write_high_level_rigid_state(
    asset: Any,
    root_pose: Any,
    root_velocity: Any,
    env_ids: Any,
    *,
    kinematic: bool,
) -> None:
    """Write an Isaac Lab rigid state while respecting PhysX kinematic semantics."""

    asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    if not kinematic:
        asset.write_root_link_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)


def _ensure_launcher_setting(setting: str) -> None:
    if setting not in sys.argv:
        sys.argv.append(setting)


def _native_name(path: EntityPath) -> str:
    digest = hashlib.sha256(path.value.encode()).hexdigest()[:10]
    return f"{path.name.replace('-', '_').replace('.', '_')}_{digest}"


def _native_asset_path(uri: str) -> str:
    """Return the exact local path admitted by the public preflight."""

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return str(Path(unquote(parsed.path)))
    if not parsed.scheme:
        return str(Path(uri))
    raise ValueError("native USD assets must use a local path or file URI")


def _usd_file_cfg(
    modules: SimpleNamespace,
    entity: EntitySpec,
    *,
    activate_contact_sensors: bool = False,
) -> Any:
    """Create one scale-aware USD spawn config for every physical asset kind."""

    if entity.asset_uri is None:
        raise ValueError("USD-backed entities require asset_uri")
    return modules.sim_utils.UsdFileCfg(
        usd_path=_native_asset_path(entity.asset_uri),
        scale=entity.scale_xyz,
        activate_contact_sensors=activate_contact_sensors,
    )


def _scaled_dimensions(
    dimensions: tuple[float, float, float],
    scale_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(dimensions[index] * scale_xyz[index] for index in range(3))  # type: ignore[return-value]


def _camera_native_data_type(modality: CameraModality) -> str:
    if modality is CameraModality.RGB:
        return "rgb"
    if modality is CameraModality.DEPTH:
        return "distance_to_camera"
    if modality is CameraModality.NORMALS:
        return "normals"
    raise ValueError(f"unsupported camera modality: {modality!r}")


def _pack_compatible_rgb_tensors(
    torch_module: Any,
    tensors: tuple[Any, ...],
    staging_cache: dict[tuple[int, tuple[int, ...], str, str], Any],
) -> tuple[tuple[int, ...], tuple[bytes, ...]] | None:
    """Copy equal-shaped CUDA RGB tensors into reusable pinned host storage."""

    if not tensors:
        return (), ()
    devices = tuple(getattr(tensor, "device", None) for tensor in tensors)
    if any(getattr(device, "type", None) != "cuda" for device in devices):
        return None
    device = devices[0]
    device_name = str(device)
    if any(str(candidate) != device_name for candidate in devices[1:]):
        return None
    try:
        normalized = tuple(tensor.to(dtype=torch_module.uint8).contiguous() for tensor in tensors)
    except (AttributeError, TypeError):
        return None
    shape = tuple(int(size) for size in normalized[0].shape)
    if any(tuple(int(size) for size in tensor.shape) != shape for tensor in normalized[1:]):
        return None
    dtype = getattr(normalized[0], "dtype", None)
    if dtype is None or any(getattr(tensor, "dtype", None) != dtype for tensor in normalized[1:]):
        return None
    try:
        stream = torch_module.cuda.current_stream(device=device)
        synchronize = stream.synchronize
    except (AttributeError, RuntimeError, TypeError):
        return None
    key = (len(normalized), shape, device_name, str(dtype))
    host_batch = staging_cache.get(key)
    if host_batch is None:
        try:
            host_batch = torch_module.empty(
                (len(normalized), *shape),
                device="cpu",
                dtype=dtype,
                pin_memory=True,
            )
        except (AttributeError, NotImplementedError, RuntimeError, TypeError):
            return None
        staging_cache[key] = host_batch
    enqueued = False
    try:
        for index, tensor in enumerate(normalized):
            host_batch[index].copy_(tensor, non_blocking=True)
            enqueued = True
    except (AttributeError, NotImplementedError, TypeError):
        if enqueued:
            synchronize()
        return None
    synchronize()
    sample_size = math.prod(shape)
    payloads = tuple(host_batch[index].numpy().tobytes(order="C") for index in range(len(normalized)))
    if any(len(payload) != sample_size for payload in payloads):
        raise RuntimeError("native RGB staging buffer returned an invalid per-camera byte size")
    return shape, payloads


def _relationship_target_path(relationship: Any) -> str | None:
    targets = tuple(relationship.GetTargets())
    return str(targets[0]) if len(targets) == 1 else None


def _nearest_body_path(target_path: str | None, body_paths: set[str], root_path: str) -> str | None:
    candidate = target_path
    while candidate is not None and (candidate == root_path or candidate.startswith(f"{root_path}/")):
        if candidate in body_paths:
            return candidate
        if candidate == root_path:
            break
        candidate = candidate.rsplit("/", 1)[0]
    return None


def _articulation_mount_body_suffix(modules: SimpleNamespace, root_path: str, link_name: str | None) -> str:
    """Resolve a unique physical root/link body below one authored articulation."""

    body_prims = tuple(
        modules.sim_utils.get_all_matching_child_prims(
            root_path,
            lambda prim: prim.HasAPI(modules.UsdPhysics.RigidBodyAPI),
        )
    )
    body_by_path = {str(prim.GetPath()): prim for prim in body_prims}
    if link_name is not None:
        matches = tuple(path for path, prim in body_by_path.items() if prim.GetName() == link_name)
        if len(matches) != 1:
            raise ValueError(
                f"camera mount link {link_name!r} must identify exactly one articulation body; found {len(matches)}"
            )
        selected = matches[0]
    else:
        children: set[str] = set()
        joint_prims = modules.sim_utils.get_all_matching_child_prims(
            root_path,
            lambda prim: prim.IsA(modules.UsdPhysics.Joint),
        )
        body_paths = set(body_by_path)
        for prim in joint_prims:
            joint = modules.UsdPhysics.Joint(prim)
            parent = _nearest_body_path(_relationship_target_path(joint.GetBody0Rel()), body_paths, root_path)
            child = _nearest_body_path(_relationship_target_path(joint.GetBody1Rel()), body_paths, root_path)
            if parent is not None and child is not None and parent != child:
                children.add(child)
        roots = tuple(sorted(body_paths - children))
        if len(roots) != 1:
            raise ValueError(f"camera root mount requires exactly one articulation root body; found {len(roots)}")
        selected = roots[0]
    suffix = selected.removeprefix(root_path)
    if not suffix.startswith("/"):
        raise ValueError("camera mount body must be below the articulation entity root")
    return suffix


def _declared_joint_map(
    path: EntityPath,
    native_names: tuple[str, ...],
    declared_names: tuple[str, ...],
) -> tuple[int, ...]:
    """Map any declared articulation subset without assuming a robot topology."""

    if not set(declared_names).issubset(native_names):
        raise ValueError(
            f"declared joint names for {path.value} are not a subset of the USD; "
            f"declared={declared_names}, native={native_names}"
        )
    return tuple(native_names.index(name) for name in declared_names)


def _declared_joint_path_map(
    path: EntityPath,
    native_paths: object,
    expected_paths: tuple[tuple[str, ...], ...],
) -> tuple[int, ...]:
    """Map logical joints by exact composed Prim path, including duplicate short names."""

    try:
        rows = tuple(tuple(str(item) for item in row) for row in cast(Iterable[Iterable[object]], native_paths))
    except TypeError as exc:
        raise ValueError(f"native articulation for {path.value} did not expose per-environment DOF paths") from exc
    if len(rows) != len(expected_paths) or not rows:
        raise ValueError(
            f"native articulation DOF path environments changed for {path.value}; "
            f"expected={len(expected_paths)}, actual={len(rows)}"
        )
    mapping: tuple[int, ...] | None = None
    for environment, (native_row, expected_row) in enumerate(zip(rows, expected_paths, strict=True)):
        if len(native_row) != len(set(native_row)):
            raise ValueError(
                f"native articulation DOF paths are ambiguous for {path.value} in environment {environment}"
            )
        if any(expected not in native_row for expected in expected_row):
            missing = tuple(expected for expected in expected_row if expected not in native_row)
            raise ValueError(
                f"embedded joint Prim paths for {path.value} are not DOFs of the bound articulation; missing={missing}"
            )
        current = tuple(native_row.index(expected) for expected in expected_row)
        if mapping is None:
            mapping = current
        elif current != mapping:
            raise ValueError(
                f"embedded articulation DOF order changed across environments for {path.value}; "
                f"first={mapping}, environment_{environment}={current}"
            )
    assert mapping is not None
    return mapping


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


def _multiply_xyzw(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    values = (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("attachment quaternion is invalid")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _relative_pose(parent: Pose, child: Pose) -> Pose:
    inverse = (
        -parent.orientation_xyzw[0],
        -parent.orientation_xyzw[1],
        -parent.orientation_xyzw[2],
        parent.orientation_xyzw[3],
    )
    displacement = tuple(child.position[index] - parent.position[index] for index in range(3))
    return Pose(
        _rotate_xyzw(displacement, inverse),  # type: ignore[arg-type]
        _multiply_xyzw(inverse, child.orientation_xyzw),
    )


def _compose_pose(parent: Pose, child: Pose) -> Pose:
    offset = _rotate_xyzw(child.position, parent.orientation_xyzw)
    return Pose(
        tuple(parent.position[index] + offset[index] for index in range(3)),  # type: ignore[arg-type]
        _multiply_xyzw(parent.orientation_xyzw, child.orientation_xyzw),
    )


def _attachment_joint_frames(
    parent_body_pose: Pose,
    child_body_pose: Pose,
    parent_T_child: Pose | None,
) -> tuple[Pose, Pose, Pose]:
    """Resolve the public relation and USD rigid-body-local joint frames.

    ``UsdPhysics.Joint`` local poses are expressed in the coordinate systems of
    the rigid-body Prims targeted by ``body0`` and ``body1``.  PhysX performs its
    own rigid-body-Prim-to-center-of-mass conversion; pre-converting these values
    to COM space applies the offset twice and makes the body jump when the joint
    is first solved.
    """

    relative = parent_T_child or _relative_pose(parent_body_pose, child_body_pose)
    joint_world_pose = (
        child_body_pose
        if parent_T_child is None
        else _compose_pose(parent_body_pose, parent_T_child)
    )
    return (
        relative,
        _relative_pose(parent_body_pose, joint_world_pose),
        _relative_pose(child_body_pose, joint_world_pose),
    )


def _retarget_physical_root_pose(target_entity: Pose, source_entity: Pose, source_root: Pose) -> Pose:
    """Move a physical root by the exact entity-frame transform it currently has."""

    return _compose_pose(target_entity, _relative_pose(source_entity, source_root))


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
    if not config.headless:
        # Isaac Lab 3 defaults to headless when no visualizer is selected, even
        # when SimulationApp receives headless=False. Select the native Kit
        # visualizer explicitly so the caller's non-headless request is real.
        launcher_args["visualizer"] = ["kit"]
        launcher_args["visualizer_explicit"] = True
    if config.enable_cameras:
        launcher_args["anti_aliasing"] = _ANTI_ALIASING_MODES[config.anti_aliasing]
        # Isaac Sim 6 defaults SimulationApp to RealTimePathTracing.  That path
        # depends on the NGX/DLSS Ray Reconstruction denoiser and degrades to a
        # visibly noisy single-sample image when NGX cannot initialize (notably
        # in otherwise valid headless containers).  Ordinary RGB cameras do not
        # need RTPT, so select the stable real-time ray-traced renderer before
        # Kit starts.  Fluid isosurfaces retain their explicit RTPT requirement.
        launcher_args["renderer"] = _camera_render_mode(config)
    if config.experience is not None:
        launcher_args["experience"] = config.experience
    return launcher_args


def _camera_render_mode(config: IsaacLabAdapterConfig) -> str:
    """Return the renderer required by the adapter's camera/fluid contract."""

    if config.fluid_render_mode == "isosurface":
        return _ISOSURFACE_CAMERA_RENDERER
    return _DEFAULT_CAMERA_RENDERER


def _camera_launcher_settings(config: IsaacLabAdapterConfig) -> tuple[str, ...]:
    """Return pre-launch RTX settings for the requested texture residency profile."""

    texture_streaming = "true" if config.texture_streaming else "false"
    settings = [
        "--/renderer/multiGpu/enabled=false",
        "--/rtx-transient/dlssg/enabled=false",
        f"--/rtx-transient/resourcemanager/enableTextureStreaming={texture_streaming}",
    ]
    if _camera_render_mode(config) == _DEFAULT_CAMERA_RENDERER:
        # Isaac Sim 6 disables the legacy RTX Real-Time implementation at Kit
        # startup.  Selecting RaytracedLighting through SimulationApp happens
        # after that startup boundary and is therefore silently mapped back to
        # RTPT unless the implementation is admitted before AppLauncher runs.
        settings.append("--/persistent/rtx/modes/rt/enabled=true")
    if not config.texture_streaming:
        settings.append("--/rtx-transient/resourcemanager/texturestreaming/async=false")
    return tuple(settings)


def _render_interval_steps(native_dt: float, max_render_hz: float | None) -> int:
    """Return a deterministic physics-step interval that never exceeds the render cap."""

    if max_render_hz is None:
        return 1
    native_hz = 1.0 / native_dt
    return max(1, math.ceil(native_hz / max_render_hz - 1.0e-12))


def _render_step_enabled(
    config: IsaacLabAdapterConfig,
    native_step_index: int,
    interval_steps: int,
) -> bool:
    return config.render and config.render_on_step and native_step_index % interval_steps == 0


class IsaacLabNativeRuntime:
    """Own exactly one Kit application and at most one native world."""

    def __init__(
        self,
        config: IsaacLabAdapterConfig,
        *,
        process_isolated: bool = False,
        startup_progress: Callable[[str], None] | None = None,
    ) -> None:
        if config.enable_cameras:
            # The installed RTX 5090 profile is stable with one renderer device and no frame generation.
            for setting in _camera_launcher_settings(config):
                _ensure_launcher_setting(setting)
        if startup_progress is not None:
            startup_progress("sdk_importing")
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        if startup_progress is not None:
            startup_progress("kit_launching")
        self._launcher = AppLauncher(**_launcher_kwargs(config, process_isolated=process_isolated))
        self._app = self._launcher.app
        if startup_progress is not None:
            startup_progress("kit_ready")
            startup_progress("runtime_importing")

        import carb  # type: ignore[import-not-found]
        import omni.physx as omni_physx  # type: ignore[import-not-found]

        if config.enable_cameras:
            expected_anti_aliasing = _ANTI_ALIASING_MODES[config.anti_aliasing]
            expected_render_mode = _camera_render_mode(config)
            # Isaac Lab applies its rendering-mode preset after constructing SimulationApp,
            # which currently overwrites SimulationApp's ``anti_aliasing`` launch value.
            # Re-apply the caller's mode after AppLauncher has completed, before any render
            # products exist, then read it back so a silent preset override cannot pass.
            render_settings = carb.settings.get_settings()
            render_settings.set("/rtx/post/aa/op", expected_anti_aliasing)
            # Re-apply the requested renderer after Isaac Lab's rendering-mode
            # preset so a preset cannot silently restore SimulationApp's RTPT
            # default after the launch configuration selected RaytracedLighting.
            render_settings.set("/rtx/rendermode", expected_render_mode)
            render_settings.set(
                "/rtx-transient/resourcemanager/enableTextureStreaming",
                config.texture_streaming,
            )
            if config.fluid_render_mode == "isosurface":
                render_settings.set("/rtx/translucency/enabled", True)
                render_settings.set("/rtx/translucency/maxRefractionBounces", 12)
                render_settings.set("/rtx/rtpt/maxBounces", 6)
                render_settings.set("/rtx/rtpt/maxSpecularAndTransmissionBounces", 6)
                render_settings.set("/rtx/rtpt/maxVolumeBounces", 6)
            actual_anti_aliasing = render_settings.get("/rtx/post/aa/op")
            actual_render_mode = render_settings.get("/rtx/rendermode")
            actual_texture_streaming = render_settings.get("/rtx-transient/resourcemanager/enableTextureStreaming")
            if actual_anti_aliasing != expected_anti_aliasing:
                raise RuntimeError(
                    "Isaac Sim did not apply the requested camera anti-aliasing mode: "
                    f"requested={config.anti_aliasing!r} expected={expected_anti_aliasing} "
                    f"actual={actual_anti_aliasing!r}"
                )
            if actual_render_mode != expected_render_mode:
                raise RuntimeError(
                    "Isaac Sim did not apply the requested camera renderer: "
                    f"expected={expected_render_mode!r} actual={actual_render_mode!r}"
                )
            if actual_texture_streaming != config.texture_streaming:
                raise RuntimeError(
                    "Isaac Sim did not apply the requested texture-streaming mode: "
                    f"requested={config.texture_streaming!r} actual={actual_texture_streaming!r}"
                )
            if config.fluid_render_mode == "isosurface" and not render_settings.get("/rtx/translucency/enabled"):
                raise RuntimeError("Isaac Sim did not enable RTX translucency for fluid isosurface rendering")
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
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # type: ignore[import-not-found]

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
            omni_physx=omni_physx,
            torch=torch,
            Gf=Gf,
            Usd=Usd,
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
        if startup_progress is not None:
            startup_progress("runtime_ready")

    def build_world(self, spec: WorldSpec) -> IsaacLabNativeWorld:
        if self._closed:
            raise RuntimeError("native runtime is closed")
        if self._active_world is not None:
            raise RuntimeError("native runtime already owns a world")
        world: IsaacLabNativeWorld | None = None
        try:
            planning_demanded = any(
                requirement.capability.value == "planning.scene@2" for requirement in spec.requirements
            )
            world_type: type[IsaacLabNativeWorld]
            if planning_demanded:
                from .native_planning import IsaacLabNativePlanningWorld

                world_type = IsaacLabNativePlanningWorld
            else:
                world_type = IsaacLabNativeWorld
            world = world_type(self, spec, self._config, self._modules)
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
        self._selected_link_views: dict[tuple[EntityPath, str], Any] = {}
        self._initial_usd_articulation: dict[EntityPath, tuple[Any, Any, Any, Any]] = {}
        self._initial_usd_articulation_gains: dict[EntityPath, tuple[Any, Any]] = {}
        self._rigids: dict[EntityPath, Any] = {}
        self._usd_rigids: dict[EntityPath, tuple[_UsdRigid, ...]] = {}
        self._usd_rigid_views: dict[EntityPath, Any] = {}
        self._initial_usd_rigid: dict[EntityPath, tuple[Any, Any]] = {}
        self._usd_rigid_wrenches: dict[EntityPath, tuple[Any, Any]] = {}
        self._kinematic_rigids: dict[EntityPath, bool] = {}
        self._entity_specs = {entity.path: entity for entity in spec.entities}
        self._entity_prim_path_cache: dict[tuple[EntityPath, int], str] = {}
        self._entity_prim_physical_root_cache: dict[tuple[EntityPath, int], bool] = {}
        self._runtime_attachments: dict[tuple[int, str], _RuntimeAttachment] = {}
        self._static_scene_roots: dict[EntityPath, tuple[str, ...]] = {}
        self._composite_scene_roots: dict[EntityPath, tuple[str, ...]] = {}
        self._composite_scene_modes: dict[EntityPath, str] = {}
        self._embedded_joint_paths: dict[EntityPath, tuple[tuple[str, ...], ...]] = {}
        self._composite_rigid_states: list[_CompositeRigidState] = []
        self._composite_articulation_states: list[_CompositeArticulationState] = []
        self._physics_activation: PhysicsActivationController | None = None
        self._physics_activation_live_state = False
        self._mounted_cameras: dict[EntityPath, _MountedCamera] = {}
        self._usd_tensor_view: Any | None = None
        self._contacts: dict[EntityPath, Any] = {}
        self._deformables: dict[EntityPath, Any] = {}
        self._fluids: dict[EntityPath, tuple[_FluidSet, ...]] = {}
        self._cameras: dict[EntityPath, Any] = {}
        self._rgb_host_staging: dict[tuple[int, tuple[int, ...], str, str], Any] = {}
        self._rgb_shared_host: Any | None = None
        self._rgb_shared_host_size = 0
        self._debug_draw_interface: Any | None = None
        self._debug_overlay: NativeDebugOverlay | None = None
        self._debug_expirations: dict[tuple[str, str, str], int | None] = {}
        self._debug_lifetimes: dict[tuple[str, str, str], DebugLifetimeMode] = {}
        self._debug_mesh_resources: dict[str, DebugMeshResource] = {}
        self._debug_mesh_paths: dict[tuple[str, str, str], str] = {}
        self._debug_mesh_resource_ids: dict[tuple[str, str, str], str] = {}
        self._debug_mesh_signatures: dict[tuple[str, str, str], tuple[object, ...]] = {}
        self._step_index = 0
        self._render_revision = 0
        self._rendered_revision = -1
        self._joint_maps: dict[EntityPath, tuple[int, ...]] = {}
        self._initial_articulation: dict[EntityPath, tuple[Any, Any, Any]] = {}
        self._initial_articulation_gains: dict[EntityPath, tuple[Any, Any]] = {}
        self._articulation_control_modes: dict[EntityPath, list[list[CommandMode | None]]] = {}
        self._initial_rigid: dict[EntityPath, tuple[Any, Any]] = {}
        self._rigid_wrenches: dict[EntityPath, tuple[Any, Any]] = {}
        self._initial_entity_prim_poses: dict[EntityPath, tuple[Pose, ...]] = {}
        self._initial_deformable: dict[EntityPath, tuple[Any, Any | None]] = {}
        self._origins_cpu = _environment_origins(spec.environments.count, config.environment_spacing_m)
        self._origins: Any | None = None
        self._native_dt = spec.physics.time_step_seconds / spec.physics.substeps
        self._render_interval_steps = _render_interval_steps(
            self._native_dt,
            config.max_render_hz,
        )
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

    @staticmethod
    def _debug_mesh_path(key: tuple[str, str, str]) -> str:
        identity = "\0".join(key).encode("utf-8")
        return f"/World/UniRoboSimDebug/Instances/debug_{hashlib.sha256(identity).hexdigest()[:24]}"

    def _author_debug_mesh_resource(
        self,
        root_path: str,
        resource: DebugMeshResource,
        primitive: DebugPrimitive,
    ) -> None:
        stage = self._m.sim_utils.get_current_stage()
        prototype_path = f"{root_path}/Prototypes/mesh"
        self._m.UsdGeom.Xform.Define(stage, prototype_path)
        mesh = self._m.UsdGeom.Mesh.Define(stage, f"{prototype_path}/surface")
        vertices = tuple(
            (float(values[0]), float(values[1]), float(values[2]))
            for values in resource.vertices_m.nested()
        )
        triangles = tuple(
            (int(values[0]), int(values[1]), int(values[2]))
            for values in resource.triangle_indices.nested()
        )
        mesh.CreatePointsAttr().Set(self._m.Vt.Vec3fArray(vertices))
        mesh.CreateFaceVertexCountsAttr().Set(self._m.Vt.IntArray([3] * len(triangles)))
        mesh.CreateFaceVertexIndicesAttr().Set(self._m.Vt.IntArray([index for face in triangles for index in face]))
        mesh.CreateSubdivisionSchemeAttr().Set(self._m.UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr().Set(True)
        mesh.CreateDisplayColorPrimvar(self._m.UsdGeom.Tokens.constant).Set(
            self._m.Vt.Vec3fArray([primitive.color_rgba[:3]])
        )
        mesh.CreateDisplayOpacityPrimvar(self._m.UsdGeom.Tokens.constant).Set(
            self._m.Vt.FloatArray([primitive.color_rgba[3]])
        )
        mesh.CreateVisibilityAttr().Set(
            self._m.UsdGeom.Tokens.invisible
            if primitive.mesh_style is DebugMeshStyle.WIREFRAME
            else self._m.UsdGeom.Tokens.inherited
        )

        edges = _triangle_edges(resource)
        curves = self._m.UsdGeom.BasisCurves.Define(stage, f"{prototype_path}/edges")
        curves.CreateTypeAttr().Set(self._m.UsdGeom.Tokens.linear)
        curves.CreateWrapAttr().Set(self._m.UsdGeom.Tokens.nonperiodic)
        curves.CreatePointsAttr().Set(
            self._m.Vt.Vec3fArray([vertices[index] for edge in edges for index in edge])
        )
        curves.CreateCurveVertexCountsAttr().Set(self._m.Vt.IntArray([2] * len(edges)))
        curves.CreateWidthsAttr().Set(self._m.Vt.FloatArray([max(0.0005, primitive.size)]))
        curves.SetWidthsInterpolation(self._m.UsdGeom.Tokens.constant)
        curves.CreateDisplayColorPrimvar(self._m.UsdGeom.Tokens.constant).Set(
            self._m.Vt.Vec3fArray([primitive.color_rgba[:3]])
        )
        curves.CreateDisplayOpacityPrimvar(self._m.UsdGeom.Tokens.constant).Set(
            self._m.Vt.FloatArray([primitive.color_rgba[3]])
        )
        curves.CreateVisibilityAttr().Set(
            self._m.UsdGeom.Tokens.inherited
            if primitive.mesh_style in {DebugMeshStyle.WIREFRAME, DebugMeshStyle.SOLID_WITH_EDGES}
            else self._m.UsdGeom.Tokens.invisible
        )

    def _upsert_debug_mesh(self, primitive: DebugPrimitive) -> None:
        assert primitive.kind is DebugPrimitiveKind.MESH_INSTANCE
        assert primitive.mesh_resource_id is not None
        resource = self._debug_mesh_resources.get(primitive.mesh_resource_id)
        if resource is None:
            raise RuntimeError(f"debug mesh resource is unavailable: {primitive.mesh_resource_id}")
        key = primitive.key
        root_path = self._debug_mesh_path(key)
        stage = self._m.sim_utils.get_current_stage()
        signature = (primitive.mesh_resource_id, primitive.color_rgba, primitive.mesh_style, primitive.size)
        if self._debug_mesh_signatures.get(key) != signature or not stage.GetPrimAtPath(root_path).IsValid():
            stage.RemovePrim(root_path)
            instancer = self._m.UsdGeom.PointInstancer.Define(stage, root_path)
            self._author_debug_mesh_resource(root_path, resource, primitive)
            instancer.CreatePrototypesRel().SetTargets([self._m.Sdf.Path(f"{root_path}/Prototypes/mesh")])
            self._debug_mesh_paths[key] = root_path
            self._debug_mesh_resource_ids[key] = primitive.mesh_resource_id
            self._debug_mesh_signatures[key] = signature
        else:
            instancer = self._m.UsdGeom.PointInstancer(stage.GetPrimAtPath(root_path))

        rows = _mesh_instance_rows(primitive, self._origins_cpu)
        instancer.CreatePositionsAttr().Set(self._m.Vt.Vec3fArray([row[0] for row in rows]))
        instancer.CreateOrientationsAttr().Set(
            self._m.Vt.QuathArray(
                [
                    self._m.Gf.Quath(
                        row[1][3],
                        self._m.Gf.Vec3h(row[1][0], row[1][1], row[1][2]),
                    )
                    for row in rows
                ]
            )
        )
        instancer.CreateScalesAttr().Set(self._m.Vt.Vec3fArray([row[2] for row in rows]))
        instancer.CreateProtoIndicesAttr().Set(self._m.Vt.IntArray([0] * len(rows)))

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
        physx_cfg = self._m.PhysxCfg(
            gpu_max_num_partitions=_resolved_gpu_max_num_partitions(self._config, self._spec)
        )
        sim_cfg = sim_utils.SimulationCfg(
            dt=self._native_dt,
            gravity=self._spec.physics.gravity_m_s2,
            device=self._config.device,
            physics=physx_cfg,
            # PhysX 6 exposes particle state through USD sync, not the removed particle tensor view.
            use_fabric=not has_fluid,
            render_interval=self._render_interval_steps,
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
        if not any(
            entity.kind in {EntityKind.STATIC_SCENE, EntityKind.COMPOSITE_SCENE} for entity in self._spec.entities
        ):
            self._author_procedural_ground()
        # Containers must be composed before any embedded entity resolves exact
        # source Prim paths. This pass is deliberately independent of WorldSpec
        # ordering and authors each source asset exactly once per environment.
        for entity in self._spec.entities:
            if entity.kind is EntityKind.COMPOSITE_SCENE:
                self._author_composite_scene(entity)
        self._bind_embedded_entities()
        for entity in self._spec.entities:
            if entity.embedded_binding is not None or entity.kind is EntityKind.COMPOSITE_SCENE:
                continue
            if entity.kind is EntityKind.ARTICULATION:
                self._author_articulation(entity)
            elif entity.kind is EntityKind.RIGID_BODY:
                self._author_rigid(entity)
            elif entity.kind is EntityKind.STATIC_SCENE:
                self._author_static_scene(entity)
            elif entity.kind in {EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE}:
                self._author_deformable(entity)
            elif entity.kind is EntityKind.PARTICLE_FLUID:
                self._author_particle_fluid(entity)
        # Camera parents must already exist before a mounted sensor resolves its
        # articulation root/link. WorldSpec ordering is deliberately irrelevant.
        for entity in self._spec.entities:
            if entity.kind is EntityKind.CAMERA_SENSOR:
                self._author_camera(entity)
        self._initialize_physics_activation()
        self._sim.reset()
        self._initialize_usd_articulations()
        self._initialize_usd_rigids()
        self._initialize_composite_physics()
        self._origins = self._m.torch.tensor(self._origins_cpu, device=self._sim.device, dtype=self._m.torch.float32)
        self._initialize_articulations()
        self._initialize_rigids()
        self._initialize_deformables()
        self._initialize_dynamic_physics_activation()
        self._physics_activation_live_state = True
        self.reset(tuple(range(self._spec.environments.count)))
        self._initial_entity_prim_poses = {
            path: tuple(state.pose for state in row)
            for path, row in zip(
                tuple(entity.path for entity in self._spec.entities),
                self.read_entity_prim_states(tuple(entity.path for entity in self._spec.entities)),
                strict=True,
            )
        }
        if self._cameras:
            self._ensure_camera_render()
            for camera in self._cameras.values():
                camera.update(0.0, force_recompute=True)

    def _initialize_physics_activation(self) -> None:
        raw = self._spec.metadata.get("fastsim_physics_activation")
        if raw is None:
            return
        anchor_paths = tuple(EntityPath(value) for value in raw["anchor_paths"])
        protected_roots: list[str] = []
        for entity in self._spec.entities:
            binding = entity.embedded_binding
            if binding is None:
                continue
            container_roots = self._composite_scene_roots.get(binding.container_path)
            if container_roots is None:
                raise RuntimeError("embedded physics-activation entity has no composite container")
            for container_root in container_roots:
                protected_roots.extend(
                    f"{container_root}/{item.relative_prim_path}" for item in binding.link_prims
                )
        scene_roots = tuple(
            root
            for collection in (self._static_scene_roots, self._composite_scene_roots)
            for roots in collection.values()
            for root in roots
        )
        self._physics_activation = build_physics_activation_controller(
            self._m,
            self._spec,
            scene_roots=scene_roots,
            protected_roots=tuple(protected_roots),
            anchor_points=lambda: self._physics_activation_anchor_points(anchor_paths),
        )
        if self._physics_activation is not None:
            diagnostics = self._physics_activation.diagnostics
            self._m.carb.log_info(
                "UniRoboSim proximity physics activation: "
                f"enabled={diagnostics.enabled_count} disabled={diagnostics.disabled_count} "
                f"protected={diagnostics.protected_count}"
            )

    def _initialize_dynamic_physics_activation(self) -> None:
        controller = self._physics_activation
        if controller is None:
            return
        if self._config.render:
            # PhysX 6.0's per-rigid-body eDISABLE_SIMULATION path can corrupt
            # RTX/Cubric device state when a render product is active. Static
            # collider activation remains safe, but dynamic-body suspension is
            # therefore restricted to the true physics-only launch profile.
            controller.configure_dynamic_candidates(())
            return
        raw = self._spec.metadata["fastsim_physics_activation"]
        stage = self._m.sim_utils.get_current_stage()
        candidates: list[DynamicRigidBodyCandidate] = []
        for value in raw["managed_paths"]:
            path = EntityPath(value)
            entity = self._entity_specs.get(path)
            if entity is None or entity.kind is not EntityKind.RIGID_BODY:
                raise RuntimeError(f"physics activation managed entity is invalid: {value}")
            body_paths: list[str] = []
            binding = entity.embedded_binding
            if binding is not None:
                container_roots = self._composite_scene_roots.get(binding.container_path)
                if container_roots is None:
                    raise RuntimeError(f"physics activation container is unavailable: {value}")
                for root in container_roots:
                    for item in binding.link_prims:
                        prim = stage.GetPrimAtPath(f"{root}/{item.relative_prim_path}")
                        if not prim or not prim.IsValid() or not prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI):
                            raise RuntimeError(
                                f"physics activation link is not a rigid body: {root}/{item.relative_prim_path}"
                            )
                        body_paths.append(self._prim_path_string(prim))
            else:
                native_name = _native_name(path)
                for environment in range(self._spec.environments.count):
                    root_path = f"/World/env_{environment}/{native_name}"
                    root = stage.GetPrimAtPath(root_path)
                    if not root or not root.IsValid():
                        raise RuntimeError(f"physics activation entity root is unavailable: {root_path}")
                    body_paths.extend(
                        self._prim_path_string(prim)
                        for prim in self._m.Usd.PrimRange(root)
                        if prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                    )
            requested = tuple(dict.fromkeys(body_paths))
            if not requested:
                raise RuntimeError(f"physics activation entity has no rigid bodies: {value}")
            view = self._usd_simulation_view().create_rigid_body_view(list(requested))
            actual = tuple(view.prim_paths)
            if view.count != len(requested) or set(actual) != set(requested):
                raise RuntimeError(
                    "physics activation rigid-body view did not preserve the requested Prim set; "
                    f"entity={value}, requested={len(requested)}, actual={view.count}"
                )
            candidates.append(DynamicRigidBodyCandidate(path=value, view=view))
        controller.configure_dynamic_candidates(tuple(candidates))
        diagnostics = controller.diagnostics
        self._m.carb.log_info(
            "UniRoboSim proximity dynamic activation: "
            f"managed={diagnostics.dynamic_candidate_count} "
            f"enabled={diagnostics.dynamic_enabled_count} disabled={diagnostics.dynamic_disabled_count}"
        )

    def _physics_activation_anchor_points(
        self,
        paths: tuple[EntityPath, ...],
    ) -> Any:
        points: list[Any] = []
        for path in paths:
            positions: Any | None = None
            if not self._physics_activation_live_state:
                entity = self._entity_specs.get(path)
                if entity is None:
                    raise RuntimeError(f"physics activation support entity is unavailable: {path.value}")
                positions = self._m.torch.tensor(
                    [
                        tuple(entity.pose.position[axis] + origin[axis] for axis in range(3))
                        for origin in self._origins_cpu
                    ],
                    device=self._sim.device,
                    dtype=self._m.torch.float32,
                )
            elif path in self._articulations:
                positions = self._articulations[path].data.body_pos_w.torch
            elif path in self._usd_articulation_views:
                positions = self._usd_articulation_views[path].get_link_transforms()[..., :3]
            elif path in self._rigids:
                positions = self._rigids[path].data.root_link_pose_w.torch[..., :3]
            elif path in self._usd_rigid_views:
                positions = self._usd_rigid_views[path].get_transforms()[..., :3]
            if positions is None:
                raise RuntimeError(f"physics activation support entity is not initialized: {path.value}")
            points.append(positions.reshape(-1, 3))
        if not points:
            raise RuntimeError("physics activation has no initialized support entities")
        return self._m.torch.cat(points, dim=0)

    def _pin_physics_activation(self, path: EntityPath) -> None:
        controller = getattr(self, "_physics_activation", None)
        if controller is not None:
            controller.pin(path.value)

    def _author_procedural_ground(self) -> None:
        """Author the implicit ground without loading Isaac's Nucleus USD asset."""

        stage = self._m.sim_utils.get_current_stage()
        root_path = "/World/unirobosimGround"
        self._m.UsdGeom.Xform.Define(stage, root_path)
        plane = self._m.UsdGeom.Plane.Define(stage, f"{root_path}/plane")
        plane.CreateAxisAttr().Set("Z")
        plane.CreateWidthAttr().Set(100.0)
        plane.CreateLengthAttr().Set(100.0)
        plane.CreateDoubleSidedAttr().Set(True)
        plane.CreateDisplayColorPrimvar(self._m.UsdGeom.Tokens.constant).Set(
            self._m.Vt.Vec3fArray([self._m.Gf.Vec3f(0.2, 0.23, 0.28)])
        )
        collision = self._m.UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
        collision.CreateCollisionEnabledAttr().Set(True)

    def _configure_high_level_initial_root_pose(
        self,
        entity: EntitySpec,
        asset: Any,
        physical_root_paths: tuple[str, ...],
    ) -> None:
        """Translate the public entity-Prim pose into Isaac's physical-root initial state."""

        if len(physical_root_paths) != self._spec.environments.count:
            raise ValueError(
                f"physical root count for {entity.path.value} does not match the environment count"
            )
        targets = tuple(
            _retarget_physical_root_pose(
                entity.pose,
                self._read_usd_entity_prim_pose(entity.path, environment),
                self._read_usd_prim_pose(physical_root_path, environment),
            )
            for environment, physical_root_path in enumerate(physical_root_paths)
        )
        first = targets[0]
        for target in targets[1:]:
            position_error = max(abs(left - right) for left, right in zip(first.position, target.position, strict=True))
            orientation_dot = abs(
                sum(
                    left * right
                    for left, right in zip(
                        first.orientation_xyzw,
                        target.orientation_xyzw,
                        strict=True,
                    )
                )
            )
            if position_error > 1.0e-6 or abs(orientation_dot - 1.0) > 1.0e-6:
                raise ValueError(
                    f"authored entity-to-root transform for {entity.path.value} differs across environments"
                )
        asset.cfg.init_state = asset.cfg.init_state.replace(
            pos=first.position,
            rot=first.orientation_xyzw,
        )

    def _author_articulation(self, entity: EntitySpec) -> None:
        assert entity.asset_uri is not None
        if self._has_fluid:
            self._author_usd_articulation(entity)
            return
        cfg = self._m.ArticulationCfg(
            prim_path=f"/World/env_.*/{_native_name(entity.path)}",
            spawn=_usd_file_cfg(self._m, entity),
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
        asset = self._m.Articulation(cfg)
        self._articulations[entity.path] = asset
        root_paths = tuple(
            root + _articulation_mount_body_suffix(self._m, root, None)
            for root in (
                f"/World/env_{environment}/{_native_name(entity.path)}"
                for environment in range(self._spec.environments.count)
            )
        )
        self._configure_high_level_initial_root_pose(entity, asset, root_paths)

    def _author_usd_articulation(self, entity: EntitySpec) -> None:
        """Author an articulation through USD for particle-readback worlds."""

        assert entity.asset_uri is not None
        name = _native_name(entity.path)
        articulations: list[_UsdArticulation] = []
        for index in range(self._spec.environments.count):
            root = f"/World/env_{index}/{name}"
            cfg = _usd_file_cfg(self._m, entity)
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
                    size=_scaled_dimensions(entity.box.dimensions_m, entity.scale_xyz),
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
            self._kinematic_rigids[entity.path] = False
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
            spawn=_usd_file_cfg(self._m, entity, activate_contact_sensors=True),
            init_state=self._m.RigidObjectCfg.InitialStateCfg(
                pos=entity.pose.position,
                rot=entity.pose.orientation_xyzw,
            ),
        )
        asset = self._m.RigidObject(cfg)
        self._rigids[entity.path] = asset
        body_suffix: str | None = None
        kinematic: bool | None = None
        root_paths: list[str] = []
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
            root_paths.append(rigid_prim.GetPath().pathString)
            current_kinematic = _is_kinematic_rigid(rigid_prim, self._m.UsdPhysics)
            if kinematic is None:
                kinematic = current_kinematic
            elif current_kinematic != kinematic:
                raise ValueError("rigid body must have the same kinematic mode in every environment")
            if "PhysxContactReportAPI" not in rigid_prim.GetAppliedSchemas():
                rigid_prim.AddAppliedSchema("PhysxContactReportAPI")
            suffix = rigid_prim.GetPath().pathString.removeprefix(root)
            if body_suffix is None:
                body_suffix = suffix
            elif suffix != body_suffix:
                raise ValueError("rigid body prim must have the same relative path in every environment")
        assert body_suffix is not None and kinematic is not None
        self._configure_high_level_initial_root_pose(entity, asset, tuple(root_paths))
        self._kinematic_rigids[entity.path] = kinematic
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
        kinematic: bool | None = None
        for index in range(self._spec.environments.count):
            root = f"/World/env_{index}/{name}"
            cfg = _usd_file_cfg(self._m, entity, activate_contact_sensors=True)
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
            current_kinematic = _is_kinematic_rigid(rigid_prim, self._m.UsdPhysics)
            if kinematic is None:
                kinematic = current_kinematic
            elif current_kinematic != kinematic:
                raise ValueError("rigid body must have the same kinematic mode in every environment")
            if "PhysxContactReportAPI" not in rigid_prim.GetAppliedSchemas():
                rigid_prim.AddAppliedSchema("PhysxContactReportAPI")
            bodies.append(_UsdRigid(rigid_prim=rigid_prim))
        assert kinematic is not None
        self._usd_rigids[entity.path] = tuple(bodies)
        self._kinematic_rigids[entity.path] = kinematic

    def _author_static_scene(self, entity: EntitySpec) -> None:
        """Compose one immutable USD scene per environment without asset wrappers."""

        assert entity.asset_uri is not None
        name = _native_name(entity.path)
        roots: list[str] = []
        for index in range(self._spec.environments.count):
            root = f"/World/env_{index}/{name}"
            cfg = _usd_file_cfg(self._m, entity)
            cfg.func(
                root,
                cfg,
                translation=entity.pose.position,
                orientation=entity.pose.orientation_xyzw,
            )
            forbidden = self._m.sim_utils.get_all_matching_child_prims(
                root,
                lambda prim: (
                    prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                    or prim.HasAPI(self._m.UsdPhysics.ArticulationRootAPI)
                    or prim.IsA(self._m.UsdPhysics.Joint)
                ),
            )
            if forbidden:
                paths = tuple(str(prim.GetPath()) for prim in forbidden[:8])
                raise ValueError(
                    f"static-scene USD must not contain rigid bodies, articulations, or physics joints; found={paths}"
                )
            roots.append(root)
        self._static_scene_roots[entity.path] = tuple(roots)

    def _author_composite_scene(self, entity: EntitySpec) -> None:
        """Compose one mixed-physics USD container per environment exactly once."""

        assert entity.asset_uri is not None
        unbound_rigid_mode = _composite_unbound_rigid_mode(entity)
        name = _native_name(entity.path)
        roots: list[str] = []
        for index in range(self._spec.environments.count):
            root = f"/World/env_{index}/{name}"
            cfg = _usd_file_cfg(self._m, entity)
            cfg.func(
                root,
                cfg,
                translation=entity.pose.position,
                orientation=entity.pose.orientation_xyzw,
            )
            roots.append(root)
        if unbound_rigid_mode in {"kinematic", "static"}:
            self._author_unbound_composite_rigids(entity, tuple(roots), mode=unbound_rigid_mode)
        self._validate_authored_composite_scale(entity, tuple(roots))
        self._composite_scene_roots[entity.path] = tuple(roots)
        if not hasattr(self, "_composite_scene_modes"):
            self._composite_scene_modes = {}
        self._composite_scene_modes[entity.path] = unbound_rigid_mode

    def _validate_authored_composite_scale(self, entity: EntitySpec, roots: tuple[str, ...]) -> None:
        """Fail closed when native USD cannot represent an anisotropic physical scene."""

        if len(set(entity.scale_xyz)) == 1:
            return
        for root in roots:
            invalid_physics = self._m.sim_utils.get_all_matching_child_prims(
                root,
                lambda prim: (
                    prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                    or prim.HasAPI(self._m.UsdPhysics.ArticulationRootAPI)
                    or (
                        prim.IsA(self._m.UsdPhysics.Joint)
                        and self._m.UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False
                    )
                ),
            )
            if invalid_physics:
                paths = tuple(self._prim_path_string(prim) for prim in invalid_physics[:8])
                raise ValueError(
                    "non-uniform composite-scene scale requires a static physical scene after authoring; "
                    f"remaining rigid/articulation/joint prims={paths}"
                )
            collision_prims = self._m.sim_utils.get_all_matching_child_prims(
                root,
                lambda prim: (
                    prim.HasAPI(self._m.UsdPhysics.CollisionAPI)
                    and self._m.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
                ),
            )
            unsupported = tuple(
                self._prim_path_string(prim)
                for prim in collision_prims
                if not (prim.IsA(self._m.UsdGeom.Mesh) or prim.IsA(self._m.UsdGeom.Cube))
            )
            if unsupported:
                raise ValueError(
                    "non-uniform composite-scene scale supports Mesh and Cube collision only; "
                    f"unsupported collision prims={unsupported[:8]}"
                )

    def _author_unbound_composite_rigids(
        self,
        entity: EntitySpec,
        roots: tuple[str, ...],
        *,
        mode: str,
    ) -> None:
        """Disable private joints and freeze bodies not owned by embedded FastSim entities."""

        declared_embedded_body_paths = {
            prim.relative_prim_path
            for candidate in self._spec.entities
            if candidate.embedded_binding is not None and candidate.embedded_binding.container_path == entity.path
            for prim in candidate.embedded_binding.link_prims
        }
        declared_embedded_joint_paths = {
            prim.relative_prim_path
            for candidate in self._spec.entities
            if candidate.embedded_binding is not None and candidate.embedded_binding.container_path == entity.path
            for prim in candidate.embedded_binding.joint_prims
        }
        selected_rows: list[dict[str, Any]] = []
        protected_rows: list[tuple[str, ...]] = []
        disabled_joint_rows: list[dict[str, Any]] = []
        for root in roots:
            physical_prims = self._m.sim_utils.get_all_matching_child_prims(
                root,
                lambda prim: prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI) or prim.IsA(self._m.UsdPhysics.Joint),
            )
            rigid_by_relative_path: dict[str, Any] = {}
            joint_by_relative_path: dict[str, Any] = {}
            for prim in physical_prims:
                absolute = self._prim_path_string(prim)
                relative = absolute.removeprefix(f"{root}/")
                if not relative or relative == absolute:
                    raise ValueError("composite physical Prim paths are ambiguous below their container")
                if prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI):
                    if relative in rigid_by_relative_path:
                        raise ValueError("composite rigid-body paths are ambiguous below their container")
                    rigid_by_relative_path[relative] = prim
                if prim.IsA(self._m.UsdPhysics.Joint):
                    if relative in joint_by_relative_path:
                        raise ValueError("composite joint paths are ambiguous below their container")
                    joint_by_relative_path[relative] = prim

            rigid_body_paths = {self._prim_path_string(prim) for prim in rigid_by_relative_path.values()}
            protected_body_paths: set[str] = set()
            for relative in declared_embedded_body_paths:
                declared_path = f"{root}/{relative}"
                owner_path = _nearest_body_path(declared_path, rigid_body_paths, root)
                if owner_path is None:
                    raise ValueError(
                        "embedded link did not resolve to a rigid-body carrier while applying composite "
                        f"unbound-rigid mode: {declared_path}"
                    )
                protected_body_paths.add(owner_path)
            protected_rows.append(tuple(sorted(path.removeprefix(f"{root}/") for path in protected_body_paths)))
            missing_embedded_joints = declared_embedded_joint_paths - joint_by_relative_path.keys()
            if missing_embedded_joints:
                raise ValueError(
                    "embedded joint paths did not resolve while applying composite unbound-rigid mode: "
                    f"{tuple(sorted(missing_embedded_joints))[:8]}"
                )
            disabled_joint_rows.append(
                {
                    relative: prim
                    for relative, prim in joint_by_relative_path.items()
                    if relative not in declared_embedded_joint_paths
                }
            )

            selected_rows.append(
                {
                    relative: prim
                    for relative, prim in rigid_by_relative_path.items()
                    if self._prim_path_string(prim) not in protected_body_paths
                }
            )
        expected_protected_paths = protected_rows[0] if protected_rows else ()
        for environment, actual_protected_paths in enumerate(protected_rows[1:], start=1):
            if actual_protected_paths != expected_protected_paths:
                raise ValueError(
                    "composite embedded rigid-body protection changed across environments; "
                    f"first={expected_protected_paths[:8]}, environment_{environment}="
                    f"{actual_protected_paths[:8]}"
                )
        expected_relative_paths = tuple(sorted(selected_rows[0])) if selected_rows else ()
        for environment, row in enumerate(selected_rows[1:], start=1):
            actual_relative_paths = tuple(sorted(row))
            if actual_relative_paths != expected_relative_paths:
                raise ValueError(
                    "composite unbound rigid-body selection changed across environments; "
                    f"first_count={len(expected_relative_paths)}, environment_{environment}_count="
                    f"{len(actual_relative_paths)}, first_sample={expected_relative_paths[:8]}, "
                    f"environment_{environment}_sample={actual_relative_paths[:8]}"
                )
        expected_disabled_joint_paths = tuple(sorted(disabled_joint_rows[0])) if disabled_joint_rows else ()
        for environment, row in enumerate(disabled_joint_rows[1:], start=1):
            actual_disabled_joint_paths = tuple(sorted(row))
            if actual_disabled_joint_paths != expected_disabled_joint_paths:
                raise ValueError(
                    "composite private-joint selection changed across environments; "
                    f"first_count={len(expected_disabled_joint_paths)}, environment_{environment}_count="
                    f"{len(actual_disabled_joint_paths)}, first_sample={expected_disabled_joint_paths[:8]}, "
                    f"environment_{environment}_sample={actual_disabled_joint_paths[:8]}"
                )
        for row in disabled_joint_rows:
            for relative in expected_disabled_joint_paths:
                prim = row[relative]
                joint = self._m.UsdPhysics.Joint(prim)
                result = joint.CreateJointEnabledAttr().Set(False)
                if result is False or bool(joint.GetJointEnabledAttr().Get()):
                    raise RuntimeError(
                        "failed to author jointEnabled=false on a private composite joint: "
                        f"{self._prim_path_string(prim)}"
                    )
        for row in selected_rows:
            for relative in expected_relative_paths:
                prim = row[relative]
                rigid = self._m.UsdPhysics.RigidBodyAPI(prim)
                if mode == "kinematic":
                    result = rigid.CreateKinematicEnabledAttr().Set(True)
                    accepted = bool(rigid.GetKinematicEnabledAttr().Get())
                    attribute = "kinematicEnabled=true"
                elif mode == "static":
                    result = prim.RemoveAPI(self._m.UsdPhysics.RigidBodyAPI)
                    accepted = not prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                    attribute = "RigidBodyAPI removal"
                else:
                    raise ValueError(f"unsupported composite unbound rigid mode: {mode!r}")
                if result is False or not accepted:
                    raise RuntimeError(
                        f"failed to author {attribute} on an unbound composite rigid body: "
                        f"{self._prim_path_string(prim)}"
                    )

    @staticmethod
    def _prim_path_string(prim: Any) -> str:
        path = prim.GetPath()
        return str(getattr(path, "pathString", path))

    def _embedded_prim(self, root: str, relative_path: str, *, entity: EntitySpec, role: str) -> Any:
        stage = self._m.sim_utils.get_current_stage()
        absolute_path = f"{root}/{relative_path}"
        prim = stage.GetPrimAtPath(absolute_path)
        if not prim or not prim.IsValid() or self._prim_path_string(prim) != absolute_path:
            raise ValueError(
                f"embedded {role} Prim for {entity.path.value} does not resolve exactly below its container: "
                f"{absolute_path}"
            )
        return prim

    def _bind_embedded_entities(self) -> None:
        """Bind declared logical entities to already composed Prim trees without spawning."""

        claimed: dict[tuple[EntityPath, str], EntityPath] = {}
        for entity in self._spec.entities:
            binding = entity.embedded_binding
            if binding is None:
                continue
            roots = self._composite_scene_roots.get(binding.container_path)
            if roots is None:
                raise ValueError(
                    f"embedded entity {entity.path.value} references a composite container that was not composed"
                )
            for prim_binding in (*binding.link_prims, *binding.joint_prims):
                key = (binding.container_path, prim_binding.relative_prim_path)
                owner = claimed.get(key)
                if owner is not None and owner != entity.path:
                    raise ValueError(
                        f"embedded Prim {prim_binding.relative_prim_path!r} is claimed by both "
                        f"{owner.value} and {entity.path.value}"
                    )
                claimed[key] = entity.path

            root_binding = next(
                item for item in binding.link_prims if item.relative_prim_path == binding.root_body_prim_path
            )
            del root_binding  # Core validation proves membership; native validation below proves physics type.
            root_prims: list[Any] = []
            joint_paths_by_environment: list[tuple[str, ...]] = []
            kinematic: bool | None = None
            for root in roots:
                link_prims = tuple(
                    self._embedded_prim(
                        root,
                        item.relative_prim_path,
                        entity=entity,
                        role=f"link {item.logical_name!r}",
                    )
                    for item in binding.link_prims
                )
                if any(not prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI) for prim in link_prims):
                    raise ValueError(f"all embedded links for {entity.path.value} must be rigid-body Prims")
                root_prim = self._embedded_prim(
                    root,
                    binding.root_body_prim_path,
                    entity=entity,
                    role="root body",
                )
                root_prims.append(root_prim)
                if entity.kind is EntityKind.ARTICULATION:
                    if not root_prim.HasAPI(self._m.UsdPhysics.ArticulationRootAPI):
                        raise ValueError(
                            f"embedded articulation root body for {entity.path.value} must have "
                            "UsdPhysics.ArticulationRootAPI"
                        )
                    joint_prims = tuple(
                        self._embedded_prim(
                            root,
                            item.relative_prim_path,
                            entity=entity,
                            role=f"joint {item.logical_name!r}",
                        )
                        for item in binding.joint_prims
                    )
                    if any(not prim.IsA(self._m.UsdPhysics.Joint) for prim in joint_prims):
                        raise ValueError(f"all embedded joints for {entity.path.value} must be UsdPhysics.Joint Prims")
                    joint_paths_by_environment.append(tuple(self._prim_path_string(prim) for prim in joint_prims))
                elif entity.kind is EntityKind.RIGID_BODY:
                    if len(binding.link_prims) != 1:
                        raise ValueError("an embedded rigid body must bind exactly one rigid-body link Prim")
                    current_kinematic = _is_kinematic_rigid(root_prim, self._m.UsdPhysics)
                    if kinematic is None:
                        kinematic = current_kinematic
                    elif current_kinematic != kinematic:
                        raise ValueError("embedded rigid body kinematic mode changed across environments")
                else:
                    raise ValueError("only rigid bodies and articulations can use an embedded binding")

            if entity.kind is EntityKind.ARTICULATION:
                self._usd_articulations[entity.path] = tuple(_UsdArticulation(root_prim=prim) for prim in root_prims)
                self._embedded_joint_paths[entity.path] = tuple(joint_paths_by_environment)
            else:
                assert kinematic is not None
                self._usd_rigids[entity.path] = tuple(_UsdRigid(rigid_prim=prim) for prim in root_prims)
                self._kinematic_rigids[entity.path] = kinematic

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
            system.CreateEnableCCDAttr().Set(True)
            if self._config.fluid_max_velocity_m_s is not None:
                system.CreateMaxVelocityAttr().Set(self._config.fluid_max_velocity_m_s)
            if self._config.fluid_max_depenetration_velocity_m_s is not None:
                system.CreateMaxDepenetrationVelocityAttr().Set(
                    self._config.fluid_max_depenetration_velocity_m_s
                )
            system.CreateGlobalSelfCollisionEnabledAttr().Set(True)
            system.CreateNonParticleCollisionEnabledAttr().Set(True)

            if self._config.fluid_render_mode == "isosurface":
                material = self._author_fluid_surface_material(stage, root, fluid)
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
            fluid_color = getattr(fluid, "color_rgba", None) or (0.1, 0.45, 1.0, 1.0)
            points.CreateDisplayColorPrimvar(self._m.UsdGeom.Tokens.constant).Set(
                self._m.Vt.Vec3fArray((tuple(fluid_color[:3]),))
            )
            points.CreateDisplayOpacityPrimvar(self._m.UsdGeom.Tokens.constant).Set(
                self._m.Vt.FloatArray((float(fluid_color[3]),))
            )
            if self._config.fluid_render_mode == "isosurface":
                # The particle set remains active for PhysX, while RTX renders only
                # the reconstructed surface.  Rendering both is the characteristic
                # "beads inside a surface" artifact this mode is intended to avoid.
                points.CreateVisibilityAttr().Set(self._m.UsdGeom.Tokens.invisible)
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
            sets.append(_FluidSet(system, points, local_positions, local_velocities))
        self._fluids[entity.path] = tuple(sets)

    def _apply_pbd_material(self, material: Any, fluid: Any) -> None:
        material_api = self._m.PhysxSchema.PhysxPBDMaterialAPI.Apply(material.GetPrim())
        material_api.CreateDensityAttr().Set(fluid.rest_density_kg_m3)
        material_api.CreateViscosityAttr().Set(fluid.dynamic_viscosity_pa_s)
        material_api.CreateSurfaceTensionAttr().Set(fluid.surface_tension_n_m)
        material_api.CreateDampingAttr().Set(self._config.fluid_damping)
        material_api.CreateCohesionAttr().Set(self._config.fluid_cohesion)
        material_api.CreateAdhesionAttr().Set(self._config.fluid_adhesion)
        material_api.CreateAdhesionOffsetScaleAttr().Set(0.5 if self._config.fluid_adhesion > 0.0 else 0.0)
        material_api.CreateFrictionAttr().Set(self._config.fluid_friction)
        material_api.CreateCflCoefficientAttr().Set(self._config.fluid_cfl_coefficient)

    def _author_fluid_surface_material(self, stage: Any, root: str, fluid: Any) -> Any:
        """Create one material carrying surface rendering and PBD properties."""

        entity_color = getattr(fluid, "color_rgba", None)
        color = (
            entity_color
            if entity_color is not None
            else None
            if self._config.fluid_surface_color_rgb is None
            else (*self._config.fluid_surface_color_rgb, 1.0)
        )
        if color is not None:
            material_path = f"{root}/fluid_surface_material"
            material = self._m.UsdShade.Material.Define(stage, material_path)
            shader = self._m.UsdShade.Shader.Define(stage, f"{material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", self._m.Sdf.ValueTypeNames.Color3f).Set(
                self._m.Gf.Vec3f(*color[:3])
            )
            shader.CreateInput("roughness", self._m.Sdf.ValueTypeNames.Float).Set(0.28)
            shader.CreateInput("metallic", self._m.Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("opacity", self._m.Sdf.ValueTypeNames.Float).Set(float(color[3]))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self._apply_pbd_material(material, fluid)
            return material

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
        smoothing.CreateStrengthAttr().Set(1.0)

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
        isosurface.CreateSurfaceDistanceAttr().Set(
            fluid_rest_offset * self._config.fluid_surface_distance_scale
        )
        isosurface.CreateGridFilteringPassesAttr().Set("")
        isosurface.CreateGridSmoothingRadiusAttr().Set(
            fluid_rest_offset * self._config.fluid_surface_smoothing_scale
        )
        isosurface.CreateNumMeshSmoothingPassesAttr().Set(5)
        isosurface.CreateNumMeshNormalSmoothingPassesAttr().Set(6)
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
        data_types = [_camera_native_data_type(modality) for modality in camera.modalities]
        prim_path = f"/World/env_.*/{_native_name(entity.path)}"
        if entity.mount is not None:
            parent = next(item for item in self._spec.entities if item.path == entity.mount.parent_path)
            if parent.kind is not EntityKind.ARTICULATION:
                raise ValueError("mounted cameras require an articulation parent in this adapter profile")
            parent_name = _native_name(parent.path)
            suffix: str | None = None
            for environment in range(self._spec.environments.count):
                root = f"/World/env_{environment}/{parent_name}"
                current = _articulation_mount_body_suffix(self._m, root, entity.mount.parent_link_name)
                if suffix is None:
                    suffix = current
                elif current != suffix:
                    raise ValueError("camera mount body must have the same relative path in every environment")
            assert suffix is not None
            prim_path = f"/World/env_.*/{parent_name}{suffix}/{_native_name(entity.path)}"
            self._mounted_cameras[entity.path] = _MountedCamera(
                parent_path=parent.path,
                body_name=suffix.rsplit("/", 1)[-1],
                body_suffix=suffix,
                local_pose=entity.pose,
            )
        cfg = self._m.CameraCfg(
            prim_path=prim_path,
            update_period=0.0,
            # Isaac Lab otherwise freezes ``CameraData.pos_w`` and quaternion
            # at initialization.  Pay the FrameView pose-refresh cost only for
            # cameras whose parent can move.
            update_latest_camera_pose=entity.mount is not None,
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

    def _mounted_parent_pose(self, binding: _MountedCamera) -> tuple[Any, Any]:
        if binding.parent_path in self._articulations:
            articulation = self._articulations[binding.parent_path]
            if binding.body_index is None:
                matches = tuple(
                    index for index, name in enumerate(articulation.body_names) if name == binding.body_name
                )
                if len(matches) != 1:
                    raise RuntimeError(
                        f"mounted camera body {binding.body_name!r} must match exactly one native articulation body"
                    )
                binding.body_index = matches[0]
            return (
                articulation.data.body_pos_w.torch[:, binding.body_index],
                articulation.data.body_quat_w.torch[:, binding.body_index],
            )
        if binding.parent_path in self._usd_articulation_views:
            if binding.raw_body_view is None:
                parent_name = _native_name(binding.parent_path)
                expected_paths = [
                    f"/World/env_{environment}/{parent_name}{binding.body_suffix}"
                    for environment in range(self._spec.environments.count)
                ]
                view = self._usd_simulation_view().create_rigid_body_view(expected_paths)
                if view.count != self._spec.environments.count or tuple(view.prim_paths) != tuple(expected_paths):
                    raise RuntimeError("mounted camera raw body view did not preserve environment order")
                binding.raw_body_view = view
            transforms = binding.raw_body_view.get_transforms()
            return transforms[:, :3], transforms[:, 3:7]
        raise RuntimeError("mounted camera parent articulation is not initialized")

    def _sync_mounted_camera(self, path: EntityPath) -> None:
        binding = self._mounted_cameras.get(path)
        if binding is None:
            return
        parent_position, parent_orientation = self._mounted_parent_pose(binding)
        local_position = self._m.torch.tensor(
            binding.local_pose.position,
            device=parent_position.device,
            dtype=parent_position.dtype,
        ).expand_as(parent_position)
        local_orientation = self._m.torch.tensor(
            binding.local_pose.orientation_xyzw,
            device=parent_orientation.device,
            dtype=parent_orientation.dtype,
        ).expand_as(parent_orientation)
        parent_vector = parent_orientation[:, :3]
        twice_cross = 2.0 * self._m.torch.cross(parent_vector, local_position, dim=-1)
        position = (
            parent_position
            + local_position
            + parent_orientation[:, 3:4] * twice_cross
            + self._m.torch.cross(parent_vector, twice_cross, dim=-1)
        )
        parent_x, parent_y, parent_z, parent_w = parent_orientation.unbind(dim=-1)
        local_x, local_y, local_z, local_w = local_orientation.unbind(dim=-1)
        orientation = self._m.torch.stack(
            (
                parent_w * local_x + parent_x * local_w + parent_y * local_z - parent_z * local_y,
                parent_w * local_y - parent_x * local_z + parent_y * local_w + parent_z * local_x,
                parent_w * local_z + parent_x * local_y - parent_y * local_x + parent_z * local_w,
                parent_w * local_w - parent_x * local_x - parent_y * local_y - parent_z * local_z,
            ),
            dim=-1,
        )
        self._cameras[path].set_world_poses(position, orientation, convention="opengl")

    def _sync_all_mounted_cameras(self) -> None:
        for path in self._mounted_cameras:
            self._sync_mounted_camera(path)

    def _invalidate_render(self) -> None:
        self._render_revision = getattr(self, "_render_revision", 0) + 1

    def _mark_rendered(self) -> None:
        self._rendered_revision = getattr(self, "_render_revision", 0)

    def _ensure_camera_render(self) -> None:
        revision = getattr(self, "_render_revision", 0)
        if getattr(self, "_rendered_revision", -1) == revision:
            return
        assert self._sim is not None
        # A render is global to the USD stage, not local to one Camera object.
        # Synchronize every mounted camera before that shared render so all
        # camera reads at this simulation revision consume one coherent frame.
        self._sync_all_mounted_cameras()
        self._sim.render()
        self._rendered_revision = revision

    def _initialize_articulations(self) -> None:
        torch = self._m.torch
        assert self._sim is not None
        for path, asset in self._articulations.items():
            entity = next(item for item in self._spec.entities if item.path == path)
            native_names = tuple(asset.joint_names)
            joint_map = _declared_joint_map(path, native_names, entity.joint_names)
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
            self._initial_articulation_gains[path] = (
                asset.data.joint_stiffness.torch.clone(),
                asset.data.joint_damping.torch.clone(),
            )
            self._articulation_control_modes[path] = [
                [None] * len(native_names) for _ in range(self._spec.environments.count)
            ]

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
            if path in self._embedded_joint_paths:
                joint_map = _declared_joint_path_map(
                    path,
                    view.dof_paths,
                    self._embedded_joint_paths[path],
                )
            else:
                native_names = tuple(view.shared_metatype.dof_names)
                joint_map = _declared_joint_map(path, native_names, entity.joint_names)
            self._joint_maps[path] = joint_map
            root_pose = view.get_root_transforms().clone()
            root_velocity = view.get_root_velocities().clone()
            positions = view.get_dof_positions().clone()
            for public_index, native_index in enumerate(joint_map):
                positions[:, native_index] = entity.initial_joint_positions[public_index]
            velocities = self._m.torch.zeros_like(positions)
            stiffness = view.get_dof_stiffnesses().clone()
            damping = view.get_dof_dampings().clone()
            if self._config.position_stiffness is not None:
                stiffness[:, list(joint_map)] = self._config.position_stiffness
            if self._config.position_damping is not None:
                damping[:, list(joint_map)] = self._config.position_damping
            if self._config.position_stiffness is not None or self._config.position_damping is not None:
                stiffness_indices = self._m.torch.arange(
                    view.count,
                    device=stiffness.device,
                    dtype=self._m.torch.int64,
                )
                damping_indices = self._m.torch.arange(
                    view.count,
                    device=damping.device,
                    dtype=self._m.torch.int64,
                )
                view.set_dof_stiffnesses(stiffness[stiffness_indices], stiffness_indices)
                view.set_dof_dampings(damping[damping_indices], damping_indices)
            self._initial_usd_articulation[path] = (root_pose, root_velocity, positions, velocities)
            self._initial_usd_articulation_gains[path] = (stiffness, damping)
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

    def _initialize_composite_physics(self) -> None:
        """Capture every unbound physical body contained by composite scenes."""

        if not self._composite_scene_roots:
            return
        tensor_view = self._usd_simulation_view()
        bound_articulation_roots = {
            self._prim_path_string(articulation.root_prim)
            for articulations in self._usd_articulations.values()
            for articulation in articulations
        }

        # Capture provider-authored articulation roots that were not admitted as
        # public embedded entities. They remain lifecycle-managed, but private.
        root_rows: list[dict[str, str]] = []
        for path, roots in self._composite_scene_roots.items():
            if getattr(self, "_composite_scene_modes", {}).get(path, "authored") == "static":
                continue
            for root in roots:
                root_row: dict[str, str] = {}
                prims = self._m.sim_utils.get_all_matching_child_prims(
                    root,
                    lambda prim: prim.HasAPI(self._m.UsdPhysics.ArticulationRootAPI),
                )
                for prim in prims:
                    absolute = self._prim_path_string(prim)
                    if absolute in bound_articulation_roots:
                        continue
                    relative = absolute.removeprefix(f"{root}/")
                    if not relative or relative == absolute or relative in root_row:
                        raise ValueError("composite articulation roots are ambiguous below their container")
                    root_row[relative] = absolute
                root_rows.append(root_row)
        if root_rows:
            expected_relative_roots = tuple(sorted(root_rows[0]))
            if any(tuple(sorted(row)) != expected_relative_roots for row in root_rows[1:]):
                raise ValueError("composite articulation-root topology changed across environments")
            # One native view per topology avoids combining unrelated mechanisms
            # that can have different DOF/link counts.
            for relative in expected_relative_roots:
                paths = tuple(row[relative] for row in root_rows)
                view = tensor_view.create_articulation_view(list(paths))
                if view.count != len(paths) or tuple(view.prim_paths) != paths:
                    raise RuntimeError(
                        "composite articulation view did not preserve environment order; "
                        f"expected={paths}, actual={tuple(view.prim_paths)}"
                    )
                self._composite_articulation_states.append(
                    _CompositeArticulationState(
                        view=view,
                        initial_root_transforms=view.get_root_transforms().clone(),
                        initial_root_velocities=view.get_root_velocities().clone(),
                        initial_dof_positions=view.get_dof_positions().clone(),
                        initial_dof_velocities=view.get_dof_velocities().clone(),
                        initial_position_targets=view.get_dof_position_targets().clone(),
                        initial_velocity_targets=view.get_dof_velocity_targets().clone(),
                        initial_actuation_forces=view.get_dof_actuation_forces().clone(),
                        initial_stiffnesses=view.get_dof_stiffnesses().clone(),
                        initial_dampings=view.get_dof_dampings().clone(),
                    )
                )

        articulation_links: set[str] = set()
        articulation_views = [
            *(state.view for state in self._composite_articulation_states),
            *self._usd_articulation_views.values(),
        ]
        for view in articulation_views:
            try:
                articulation_links.update(str(path) for row in view.link_paths for path in row)
            except TypeError as exc:
                raise RuntimeError("native articulation view did not expose exact link Prim paths") from exc

        # Every remaining rigid body is captured in a homogeneous motion-mode
        # view. The environment map permits deterministic partial reset without
        # assuming PhysX preserves the caller's path order.
        rigid_by_mode: dict[bool, list[tuple[str, int]]] = {False: [], True: []}
        for path, roots in self._composite_scene_roots.items():
            if getattr(self, "_composite_scene_modes", {}).get(path, "authored") == "static":
                continue
            reference_relative_paths: tuple[str, ...] | None = None
            reference_modes: dict[str, bool] = {}
            for environment, root in enumerate(roots):
                rigid_row: dict[str, Any] = {}
                prims = self._m.sim_utils.get_all_matching_child_prims(
                    root,
                    lambda prim: prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI),
                )
                for prim in prims:
                    absolute = self._prim_path_string(prim)
                    relative = absolute.removeprefix(f"{root}/")
                    if not relative or relative == absolute or relative in rigid_row:
                        raise ValueError("composite rigid-body paths are ambiguous below their container")
                    rigid_row[relative] = prim
                relative_paths = tuple(sorted(rigid_row))
                if reference_relative_paths is None:
                    reference_relative_paths = relative_paths
                    reference_modes = {
                        relative: _is_kinematic_rigid(rigid_row[relative], self._m.UsdPhysics)
                        for relative in relative_paths
                    }
                elif relative_paths != reference_relative_paths:
                    raise ValueError("composite rigid-body topology changed across environments")
                for relative in relative_paths:
                    prim = rigid_row[relative]
                    absolute = self._prim_path_string(prim)
                    mode = _is_kinematic_rigid(prim, self._m.UsdPhysics)
                    if mode != reference_modes[relative]:
                        raise ValueError("composite rigid-body kinematic mode changed across environments")
                    if absolute not in articulation_links:
                        rigid_by_mode[mode].append((absolute, environment))

        for kinematic, path_rows in rigid_by_mode.items():
            if not path_rows:
                continue
            requested_paths = tuple(path for path, _ in path_rows)
            environment_by_path = {path: environment for path, environment in path_rows}
            if len(environment_by_path) != len(requested_paths):
                raise ValueError("composite rigid-body paths are not globally unique")
            view = tensor_view.create_rigid_body_view(list(requested_paths))
            actual_paths = tuple(view.prim_paths)
            if view.count != len(requested_paths) or set(actual_paths) != set(requested_paths):
                raise RuntimeError(
                    "composite rigid-body view did not preserve the requested Prim set; "
                    f"requested_count={len(requested_paths)}, actual_count={view.count}"
                )
            transforms = view.get_transforms().clone()
            velocities = view.get_velocities().clone()
            self._composite_rigid_states.append(
                _CompositeRigidState(
                    view=view,
                    initial_transforms=transforms,
                    initial_velocities=velocities,
                    environment_by_index=tuple(environment_by_path[path] for path in actual_paths),
                    kinematic=kinematic,
                )
            )

    def _initialize_rigids(self) -> None:
        assert self._sim is not None
        for path, asset in self._rigids.items():
            root_pose = asset.data.default_root_pose.torch.clone()
            assert self._origins is not None
            root_pose[:, :3] += self._origins
            root_velocity = asset.data.default_root_vel.torch.clone()
            self._initial_rigid[path] = (root_pose, root_velocity)
            self._rigid_wrenches[path] = (
                self._m.torch.zeros(
                    (self._spec.environments.count, 3),
                    device=root_pose.device,
                    dtype=root_pose.dtype,
                ),
                self._m.torch.zeros(
                    (self._spec.environments.count, 3),
                    device=root_pose.device,
                    dtype=root_pose.dtype,
                ),
            )

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
        physics_activation = getattr(self, "_physics_activation", None)
        if physics_activation is not None:
            # Restore managed bodies while enabled; the forced update at the
            # end freezes only bodies that remain outside every robot radius.
            physics_activation.reset_dynamic_state()
        if getattr(self, "_runtime_attachments", None):
            selected = frozenset(environment_indices)
            stage = self._m.sim_utils.get_current_stage()
            for key, attachment in tuple(self._runtime_attachments.items()):
                if attachment.environment_index in selected:
                    stage.RemovePrim(attachment.joint_prim_path)
                    del self._runtime_attachments[key]
        env_ids = list(environment_indices)
        for state in getattr(self, "_composite_rigid_states", ()):
            selected = tuple(
                index
                for index, environment in enumerate(state.environment_by_index)
                if environment in environment_indices
            )
            if not selected:
                continue
            indices = self._m.torch.tensor(
                selected,
                device=state.initial_transforms.device,
                dtype=self._m.torch.int64,
            )
            _write_usd_rigid_state(
                state.view,
                state.initial_transforms[indices],
                state.initial_velocities[indices],
                indices,
                kinematic=state.kinematic,
            )
        for state in getattr(self, "_composite_articulation_states", ()):
            indices = self._m.torch.tensor(
                environment_indices,
                device=state.initial_dof_positions.device,
                dtype=self._m.torch.int64,
            )
            state.view.set_root_transforms(state.initial_root_transforms[indices], indices)
            state.view.set_root_velocities(state.initial_root_velocities[indices], indices)
            state.view.set_dof_positions(state.initial_dof_positions[indices], indices)
            state.view.set_dof_velocities(state.initial_dof_velocities[indices], indices)
            state.view.set_dof_position_targets(state.initial_position_targets[indices], indices)
            state.view.set_dof_velocity_targets(state.initial_velocity_targets[indices], indices)
            state.view.set_dof_actuation_forces(state.initial_actuation_forces[indices], indices)
            stiffness_indices = self._m.torch.tensor(
                environment_indices,
                device=state.initial_stiffnesses.device,
                dtype=self._m.torch.int64,
            )
            damping_indices = self._m.torch.tensor(
                environment_indices,
                device=state.initial_dampings.device,
                dtype=self._m.torch.int64,
            )
            state.view.set_dof_stiffnesses(
                state.initial_stiffnesses[stiffness_indices],
                stiffness_indices,
            )
            state.view.set_dof_dampings(
                state.initial_dampings[damping_indices],
                damping_indices,
            )
        for path, asset in self._articulations.items():
            root_pose, positions, velocities = self._initial_articulation[path]
            asset.write_root_pose_to_sim_index(root_pose=root_pose[env_ids], env_ids=env_ids)
            asset.write_joint_position_to_sim_index(position=positions[env_ids], env_ids=env_ids)
            asset.write_joint_velocity_to_sim_index(velocity=velocities[env_ids], env_ids=env_ids)
            asset.set_joint_position_target_index(target=positions[env_ids], env_ids=env_ids)
            asset.set_joint_velocity_target_index(target=velocities[env_ids], env_ids=env_ids)
            asset.set_joint_effort_target_index(target=self._m.torch.zeros_like(positions[env_ids]), env_ids=env_ids)
            asset.reset(env_ids=env_ids)
            stiffness, damping = self._initial_articulation_gains[path]
            asset.write_joint_stiffness_to_sim_index(stiffness=stiffness[env_ids], env_ids=env_ids)
            asset.write_joint_damping_to_sim_index(damping=damping[env_ids], env_ids=env_ids)
            control_modes = self._articulation_control_modes[path]
            for environment in environment_indices:
                control_modes[environment] = [None] * len(control_modes[environment])
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
            stiffness, damping = self._initial_usd_articulation_gains[path]
            stiffness_indices = self._m.torch.tensor(
                environment_indices,
                device=stiffness.device,
                dtype=self._m.torch.int64,
            )
            damping_indices = self._m.torch.tensor(
                environment_indices,
                device=damping.device,
                dtype=self._m.torch.int64,
            )
            view.set_dof_stiffnesses(stiffness[stiffness_indices], stiffness_indices)
            view.set_dof_dampings(damping[damping_indices], damping_indices)
        for path, asset in self._rigids.items():
            root_pose, root_velocity = self._initial_rigid[path]
            asset.reset(env_ids=env_ids)
            _write_high_level_rigid_state(
                asset,
                root_pose[env_ids],
                root_velocity[env_ids],
                env_ids,
                kinematic=self._kinematic_rigids[path],
            )
            self._contacts[path].reset(env_ids=env_ids)
            stored_wrench = getattr(self, "_rigid_wrenches", {}).get(path)
            if stored_wrench is not None:
                forces, torques = stored_wrench
                forces[env_ids] = 0.0
                torques[env_ids] = 0.0
        for path, view in self._usd_rigid_views.items():
            transforms, velocities = self._initial_usd_rigid[path]
            indices = self._m.torch.tensor(
                environment_indices,
                device=transforms.device,
                dtype=self._m.torch.int64,
            )
            _write_usd_rigid_state(
                view,
                transforms[indices],
                velocities[indices],
                indices,
                kinematic=self._kinematic_rigids[path],
            )
            stored_wrench = getattr(self, "_usd_rigid_wrenches", {}).get(path)
            if stored_wrench is not None:
                forces, torques = stored_wrench
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
        for path, poses in getattr(self, "_initial_entity_prim_poses", {}).items():
            for environment in environment_indices:
                if not self._entity_prim_is_physical_root(path, environment):
                    self._write_entity_prim_pose(path, environment, poses[environment])
        reset_debug_keys = tuple(
            key for key, mode in self._debug_lifetimes.items() if mode is not DebugLifetimeMode.MANUAL
        )
        if reset_debug_keys:
            self._remove_debug_keys(reset_debug_keys)
        assert self._sim is not None
        self._sim.forward()
        self._update_assets(0.0)
        if physics_activation is not None:
            physics_activation.update(self._step_index, force=True)
        self._sync_all_mounted_cameras()
        self._invalidate_render()

    def apply_render_state(self, frame: NativeRenderStateFrame) -> None:
        """Apply one fully prevalidated frame without calling the physics step API."""

        if type(frame) is not NativeRenderStateFrame:
            raise ValueError("native render state requires a NativeRenderStateFrame")
        if (
            any(type(update) is not NativeRenderArticulationState for update in frame.articulations)
            or any(type(update) is not NativeRenderRigidBodyState for update in frame.rigid_bodies)
            or any(type(update) is not NativeRenderParticleFluidState for update in frame.particle_fluids)
        ):
            raise ValueError("native render state contains an invalid update type")
        paths = tuple(
            update.path
            for updates in (frame.articulations, frame.rigid_bodies, frame.particle_fluids)
            for update in updates
        )
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("native render state must contain unique entity paths")

        environment_count = self._spec.environments.count
        torch = self._m.torch

        def selection(values: tuple[int, ...], size: int, field: str) -> tuple[int, ...]:
            if (
                type(values) is not tuple
                or not values
                or any(type(index) is not int or index < 0 or index >= size for index in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"native render state {field} selection is invalid")
            return values

        def tensor_payload(value: object, shape: tuple[int, ...], *, device: object = "cpu") -> Any:
            if isinstance(value, PackedFloat32Array):
                if sys.byteorder != "little" or value.shape != shape:
                    raise ValueError("native packed float32 render payload shape or byte order is invalid")
                tensor = torch.frombuffer(value.data, dtype=torch.float32).clone().reshape(shape)
                if str(device) != "cpu":
                    tensor = tensor.to(device=device)
            else:
                try:
                    tensor = torch.tensor(value, device=device, dtype=torch.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError("native render state contains an invalid numeric payload") from exc
                if tuple(int(size) for size in tensor.shape) != shape:
                    raise ValueError("native render state payload shape is invalid")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError("native render state payload must contain only finite values")
            return tensor

        articulation_stages: list[
            tuple[str, EntityPath, Any, Any, Any, Any, tuple[int, ...], Any | None, Any | None]
        ] = []
        for articulation_update in frame.articulations:
            if articulation_update.path not in self._joint_maps:
                raise ValueError("native render state references an unknown articulation")
            environments = selection(
                articulation_update.environment_indices,
                environment_count,
                "environment",
            )
            degrees = selection(
                articulation_update.degree_of_freedom_indices,
                len(self._joint_maps[articulation_update.path]),
                "degree-of-freedom",
            )
            joint_shape = (len(environments), len(degrees))
            native_degrees = tuple(
                self._joint_maps[articulation_update.path][index] for index in degrees
            )
            root_values = (
                articulation_update.root_positions_m,
                articulation_update.root_orientations_xyzw,
                articulation_update.root_linear_velocities_m_s,
                articulation_update.root_angular_velocities_rad_s,
            )
            if any(value is not None for value in root_values) and any(value is None for value in root_values):
                raise ValueError("native render articulation root state must be supplied as one complete state")
            if articulation_update.path in self._usd_articulation_views:
                view = self._usd_articulation_views[articulation_update.path]
                indices = torch.tensor(environments, device=view.get_dof_positions().device, dtype=torch.int64)
                positions = view.get_dof_positions()[indices].clone()
                velocities = view.get_dof_velocities()[indices].clone()
                target_positions = tensor_payload(
                    articulation_update.joint_positions,
                    joint_shape,
                    device=positions.device,
                )
                target_velocities = tensor_payload(
                    articulation_update.joint_velocities,
                    joint_shape,
                    device=velocities.device,
                )
                positions[:, list(native_degrees)] = target_positions
                velocities[:, list(native_degrees)] = target_velocities
                root_pose = None
                root_velocity = None
                if articulation_update.root_positions_m is not None:
                    assert articulation_update.root_orientations_xyzw is not None
                    assert articulation_update.root_linear_velocities_m_s is not None
                    assert articulation_update.root_angular_velocities_rad_s is not None
                    root_positions = tensor_payload(
                        articulation_update.root_positions_m,
                        (len(environments), 3),
                        device=positions.device,
                    )
                    root_orientations = tensor_payload(
                        articulation_update.root_orientations_xyzw,
                        (len(environments), 4),
                        device=positions.device,
                    )
                    origins = torch.tensor(
                        self._origins_cpu,
                        device=positions.device,
                        dtype=positions.dtype,
                    )[indices]
                    root_pose = torch.cat((root_positions + origins, root_orientations), dim=1)
                    root_velocity = torch.cat(
                        (
                            tensor_payload(
                                articulation_update.root_linear_velocities_m_s,
                                (len(environments), 3),
                                device=positions.device,
                            ),
                            tensor_payload(
                                articulation_update.root_angular_velocities_rad_s,
                                (len(environments), 3),
                                device=positions.device,
                            ),
                        ),
                        dim=1,
                    )
                    root_norms = torch.linalg.vector_norm(root_orientations, dim=1)
                    if not bool(
                        torch.allclose(root_norms, torch.ones_like(root_norms), rtol=0.0, atol=1.0e-6)
                    ):
                        raise ValueError("native render articulation root orientations must be unit quaternions")
                articulation_stages.append(
                    (
                        "usd",
                        articulation_update.path,
                        view,
                        indices,
                        positions,
                        velocities,
                        native_degrees,
                        root_pose,
                        root_velocity,
                    )
                )
            else:
                asset = self._articulations.get(articulation_update.path)
                if asset is None:
                    raise ValueError("native render state references an unavailable articulation")
                env_ids = list(environments)
                positions = asset.data.joint_pos.torch[env_ids].clone()
                velocities = asset.data.joint_vel.torch[env_ids].clone()
                target_positions = tensor_payload(
                    articulation_update.joint_positions,
                    joint_shape,
                    device=positions.device,
                )
                target_velocities = tensor_payload(
                    articulation_update.joint_velocities,
                    joint_shape,
                    device=velocities.device,
                )
                positions[:, list(native_degrees)] = target_positions
                velocities[:, list(native_degrees)] = target_velocities
                root_pose = None
                root_velocity = None
                if articulation_update.root_positions_m is not None:
                    assert articulation_update.root_orientations_xyzw is not None
                    assert articulation_update.root_linear_velocities_m_s is not None
                    assert articulation_update.root_angular_velocities_rad_s is not None
                    if self._origins is None:
                        raise ValueError("native environment origins are unavailable for articulation root state")
                    root_positions = tensor_payload(
                        articulation_update.root_positions_m,
                        (len(environments), 3),
                        device=positions.device,
                    )
                    root_orientations = tensor_payload(
                        articulation_update.root_orientations_xyzw,
                        (len(environments), 4),
                        device=positions.device,
                    )
                    root_pose = torch.cat((root_positions + self._origins[env_ids], root_orientations), dim=1)
                    root_velocity = torch.cat(
                        (
                            tensor_payload(
                                articulation_update.root_linear_velocities_m_s,
                                (len(environments), 3),
                                device=positions.device,
                            ),
                            tensor_payload(
                                articulation_update.root_angular_velocities_rad_s,
                                (len(environments), 3),
                                device=positions.device,
                            ),
                        ),
                        dim=1,
                    )
                    root_norms = torch.linalg.vector_norm(root_orientations, dim=1)
                    if not bool(
                        torch.allclose(root_norms, torch.ones_like(root_norms), rtol=0.0, atol=1.0e-6)
                    ):
                        raise ValueError("native render articulation root orientations must be unit quaternions")
                articulation_stages.append(
                    (
                        "high",
                        articulation_update.path,
                        asset,
                        env_ids,
                        positions,
                        velocities,
                        native_degrees,
                        root_pose,
                        root_velocity,
                    )
                )

        rigid_stages: list[tuple[str, EntityPath, Any, Any, Any, Any]] = []
        for rigid_update in frame.rigid_bodies:
            if rigid_update.path not in self._kinematic_rigids:
                raise ValueError("native render state references an unknown rigid body")
            if self._kinematic_rigids[rigid_update.path]:
                raise ValueError("native render state rigid updates require a dynamic body")
            environments = selection(rigid_update.environment_indices, environment_count, "environment")
            row_count = len(environments)
            if rigid_update.path in self._usd_rigid_views:
                target = self._usd_rigid_views[rigid_update.path]
                transforms = target.get_transforms()
                indices = torch.tensor(environments, device=transforms.device, dtype=torch.int64)
                positions = tensor_payload(rigid_update.positions_m, (row_count, 3), device=transforms.device)
                orientations = tensor_payload(
                    rigid_update.orientations_xyzw,
                    (row_count, 4),
                    device=transforms.device,
                )
                linear = tensor_payload(
                    rigid_update.linear_velocities_m_s,
                    (row_count, 3),
                    device=transforms.device,
                )
                angular = tensor_payload(
                    rigid_update.angular_velocities_rad_s,
                    (row_count, 3),
                    device=transforms.device,
                )
                origins = torch.tensor(self._origins_cpu, device=transforms.device, dtype=transforms.dtype)[indices]
                poses = torch.cat((positions + origins, orientations), dim=1)
                velocities = torch.cat((linear, angular), dim=1)
                rigid_stages.append(("usd", rigid_update.path, target, indices, poses, velocities))
            else:
                target = self._rigids.get(rigid_update.path)
                if target is None or self._sim is None or self._origins is None:
                    raise ValueError("native render state references an unavailable rigid body")
                env_ids = list(environments)
                positions = tensor_payload(rigid_update.positions_m, (row_count, 3), device=self._sim.device)
                orientations = tensor_payload(
                    rigid_update.orientations_xyzw,
                    (row_count, 4),
                    device=self._sim.device,
                )
                linear = tensor_payload(
                    rigid_update.linear_velocities_m_s,
                    (row_count, 3),
                    device=self._sim.device,
                )
                angular = tensor_payload(
                    rigid_update.angular_velocities_rad_s,
                    (row_count, 3),
                    device=self._sim.device,
                )
                poses = torch.cat((positions + self._origins[env_ids], orientations), dim=1)
                velocities = torch.cat((linear, angular), dim=1)
                rigid_stages.append(("high", rigid_update.path, target, env_ids, poses, velocities))
            quaternion_norms = torch.linalg.vector_norm(orientations, dim=1)
            if not bool(torch.allclose(quaternion_norms, torch.ones_like(quaternion_norms), rtol=0.0, atol=1.0e-6)):
                raise ValueError("native render state rigid orientations must be unit quaternions")

        fluid_stages: list[tuple[Any, Any, Any | None]] = []
        for fluid_update in frame.particle_fluids:
            sets = self._fluids.get(fluid_update.path)
            if sets is None:
                raise ValueError("native render state references an unknown particle fluid")
            environments = selection(fluid_update.environment_indices, environment_count, "environment")
            particle_count = len(sets[0].initial_positions)
            range_count = (
                fluid_update.positions_m.shape[1]
                if isinstance(fluid_update.positions_m, PackedFloat32Array)
                else 0
            )
            if not isinstance(fluid_update.positions_m, PackedFloat32Array):
                try:
                    range_count = len(fluid_update.positions_m[0])
                except (IndexError, TypeError) as exc:
                    raise ValueError("native render particle payload is invalid") from exc
            fluid_shape = (len(environments), range_count, 3)
            if (
                type(fluid_update.first_particle_index) is not int
                or fluid_update.first_particle_index < 0
                or range_count <= 0
                or fluid_update.first_particle_index + range_count > particle_count
            ):
                raise ValueError("native render particle range is invalid")
            positions = tensor_payload(fluid_update.positions_m, fluid_shape)
            velocities = (
                None
                if fluid_update.velocities_m_s is None
                else tensor_payload(fluid_update.velocities_m_s, fluid_shape)
            )
            first = fluid_update.first_particle_index
            last = first + range_count
            for row_index, environment in enumerate(environments):
                fluid_set = sets[environment]
                current_positions = torch.tensor(
                    fluid_set.points.GetPointsAttr().Get(),
                    device="cpu",
                    dtype=torch.float32,
                )
                current_velocities = torch.tensor(
                    fluid_set.points.GetVelocitiesAttr().Get(),
                    device="cpu",
                    dtype=torch.float32,
                )
                if (
                    tuple(int(size) for size in current_positions.shape) != (particle_count, 3)
                    or tuple(int(size) for size in current_velocities.shape) != (particle_count, 3)
                    or not bool(torch.isfinite(current_positions).all().item())
                    or not bool(torch.isfinite(current_velocities).all().item())
                ):
                    raise ValueError("native particle fluid storage is invalid before render-state application")
                current_positions[first:last] = positions[row_index]
                if velocities is not None:
                    current_velocities[first:last] = velocities[row_index]
                fluid_stages.append(
                    (
                        fluid_set,
                        current_positions,
                        current_velocities if velocities is not None else None,
                    )
                )

        for (
            kind,
            path,
            target,
            indices,
            positions,
            velocities,
            native_degrees,
            root_pose,
            root_velocity,
        ) in articulation_stages:
            zeros = torch.zeros_like(positions)
            if kind == "usd":
                if root_pose is not None:
                    assert root_velocity is not None
                    target.set_root_transforms(root_pose, indices)
                    target.set_root_velocities(root_velocity, indices)
                target.set_dof_positions(positions, indices)
                target.set_dof_velocities(velocities, indices)
                target.set_dof_position_targets(positions, indices)
                target.set_dof_velocity_targets(velocities, indices)
                target.set_dof_actuation_forces(zeros, indices)
            else:
                if root_pose is not None:
                    assert root_velocity is not None
                    target.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=indices)
                    target.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=indices)
                target.write_joint_position_to_sim_index(position=positions, env_ids=indices)
                target.write_joint_velocity_to_sim_index(velocity=velocities, env_ids=indices)
                target.set_joint_position_target_index(target=positions, env_ids=indices)
                target.set_joint_velocity_target_index(target=velocities, env_ids=indices)
                target.set_joint_effort_target_index(target=zeros, env_ids=indices)
                control_modes = self._articulation_control_modes[path]
                for environment in indices:
                    for degree in native_degrees:
                        control_modes[environment][degree] = CommandMode.POSITION

        for kind, path, target, indices, poses, velocities in rigid_stages:
            if kind == "usd":
                _write_usd_rigid_state(target, poses, velocities, indices, kinematic=False)
                forces, torques = self._usd_rigid_wrenches[path]
                forces[indices] = 0.0
                torques[indices] = 0.0
            else:
                _write_high_level_rigid_state(target, poses, velocities, indices, kinematic=False)
                zeros = torch.zeros((len(indices), 1, 3), device=poses.device, dtype=poses.dtype)
                target.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=zeros,
                    torques=zeros,
                    env_ids=torch.tensor(indices, device=poses.device, dtype=torch.int64),
                    is_global=True,
                )

        assert self._sim is not None
        # Particle-fluid worlds deliberately disable Isaac Lab Fabric so PhysX can
        # publish particle state through USD.  In that profile SimulationContext.forward()
        # is a no-op, which leaves raw USD-backed articulation link transforms at the
        # previous pose after set_dof_positions().  Refresh PhysX forward kinematics
        # explicitly before RTX consumes the frame.  This does not simulate or fetch a
        # physics step and therefore preserves the render-only Replay contract.
        if any(kind == "usd" for kind, *_ in articulation_stages):
            self._usd_simulation_view().update_articulations_kinematic()
        if any(kind == "usd" for kind, *_ in articulation_stages) or any(
            kind == "usd" for kind, *_ in rigid_stages
        ):
            # RTX reads this profile from USD rather than Fabric.  The tensor
            # setters above mutate PhysX state only; publish those already-computed
            # transforms to USD without simulate()/fetch_results().
            self._m.omni_physx.get_physx_interface().update_transformations(
                False,
                True,
                True,
                False,
            )

        # Publish recorded particle positions last.  A PhysX-to-USD transform
        # update can otherwise restore the previous live particle payload over
        # the state selected by Replay.
        vec3_array = self._m.Vt.Vec3fArray
        from_numpy = getattr(vec3_array, "FromNumpy", None)
        for fluid_set, positions, velocities in fluid_stages:
            if (
                self._config.fluid_render_mode == "isosurface"
                and not fluid_set.render_state_visualization_enabled
            ):
                # A PhysX isosurface is generated only by simulation.  Render-only
                # Replay must not run physics merely to rebuild it, so expose the
                # recorded particles as the deterministic RTX representation and
                # disable the stale live-simulation surface for this world.
                isosurface = self._m.PhysxSchema.PhysxParticleIsosurfaceAPI(fluid_set.system.GetPrim())
                isosurface.GetIsosurfaceEnabledAttr().Set(False)
                fluid_set.points.GetVisibilityAttr().Set(self._m.UsdGeom.Tokens.inherited)
                fluid_set.render_state_visualization_enabled = True
            position_numpy = positions.contiguous().numpy()
            position_value = from_numpy(position_numpy) if callable(from_numpy) else vec3_array(position_numpy)
            fluid_set.points.GetPointsAttr().Set(position_value)
            if velocities is not None:
                velocity_numpy = velocities.contiguous().numpy()
                velocity_value = from_numpy(velocity_numpy) if callable(from_numpy) else vec3_array(velocity_numpy)
                fluid_set.points.GetVelocitiesAttr().Set(velocity_value)
        self._sim.forward()
        self._update_assets(0.0)
        self._sync_all_mounted_cameras()
        self._invalidate_render()

    def apply_articulation(
        self,
        path: EntityPath,
        mode: CommandMode,
        targets: Matrix,
        environment_indices: tuple[int, ...],
        degree_of_freedom_indices: tuple[int, ...],
    ) -> None:
        self._pin_physics_activation(path)
        if path in self._usd_articulation_views:
            self._apply_usd_articulation(path, mode, targets, environment_indices, degree_of_freedom_indices)
            return
        asset = self._articulations[path]
        assert self._sim is not None
        env_ids = list(environment_indices)
        joint_ids = [self._joint_maps[path][index] for index in degree_of_freedom_indices]
        target = self._m.torch.tensor(targets, device=self._sim.device, dtype=self._m.torch.float32)
        zeros = self._m.torch.zeros_like(target)
        control_modes = self._articulation_control_modes[path]
        gains_changed = any(
            control_modes[environment][joint] is not mode for environment in environment_indices for joint in joint_ids
        )
        if mode is CommandMode.POSITION:
            asset.set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids)
            asset.set_joint_velocity_target_index(target=zeros, joint_ids=joint_ids, env_ids=env_ids)
            asset.set_joint_effort_target_index(target=zeros, joint_ids=joint_ids, env_ids=env_ids)
            if gains_changed:
                initial_stiffness, initial_damping = self._initial_articulation_gains[path]
                stiffness = initial_stiffness[env_ids][:, joint_ids]
                damping = initial_damping[env_ids][:, joint_ids]
                stiffness, damping = _position_command_gains(
                    stiffness,
                    damping,
                    fallback_authored_zero=self._config.position_stiffness is None,
                    fallback_authored_damping=self._config.position_damping is None,
                )
        elif mode is CommandMode.VELOCITY:
            asset.set_joint_velocity_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids)
            asset.set_joint_effort_target_index(target=zeros, joint_ids=joint_ids, env_ids=env_ids)
            if gains_changed:
                stiffness = zeros
                damping = self._m.torch.full_like(target, self._config.velocity_damping)
        else:
            asset.set_joint_effort_target_index(target=target, joint_ids=joint_ids, env_ids=env_ids)
            if gains_changed:
                stiffness = zeros
                damping = zeros
        if gains_changed:
            asset.write_joint_stiffness_to_sim_index(stiffness=stiffness, joint_ids=joint_ids, env_ids=env_ids)
            asset.write_joint_damping_to_sim_index(damping=damping, joint_ids=joint_ids, env_ids=env_ids)
            for environment in environment_indices:
                for joint in joint_ids:
                    control_modes[environment][joint] = mode

    def _validate_articulation_command(self, command: NativeArticulationCommand) -> None:
        if type(command) is not NativeArticulationCommand or command.path not in self._joint_maps:
            raise ValueError("articulation command references an unknown native articulation")
        environment_count = self._spec.environments.count
        joint_count = len(self._joint_maps[command.path])
        if (
            type(command.mode) is not CommandMode
            or not command.environment_indices
            or not command.degree_of_freedom_indices
            or len(command.environment_indices) != len(set(command.environment_indices))
            or len(command.degree_of_freedom_indices) != len(set(command.degree_of_freedom_indices))
            or any(index < 0 or index >= environment_count for index in command.environment_indices)
            or any(index < 0 or index >= joint_count for index in command.degree_of_freedom_indices)
            or len(command.targets) != len(command.environment_indices)
            or any(len(row) != len(command.degree_of_freedom_indices) for row in command.targets)
            or any(not math.isfinite(value) for row in command.targets for value in row)
        ):
            raise ValueError("articulation command failed native batch prevalidation")

    def apply_articulation_commands_and_step(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
    ) -> None:
        """Validate a complete control tick, apply every setter, then step once."""

        if type(commands) is not tuple or type(count) is not int or count <= 0:
            raise ValueError("native articulation batch and step count are invalid")
        for command in commands:
            self._validate_articulation_command(command)
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
        """Advance once and return demanded articulation state in the same worker transaction."""

        if type(paths) is not tuple or any(path not in self._joint_maps for path in paths):
            raise ValueError("native articulation state batch references an unknown articulation")
        self.apply_articulation_commands_and_step(commands, count)
        return tuple(self.read_articulation(path) for path in paths)

    def apply_articulation_commands_step_and_read_sensors(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
        paths: tuple[EntityPath, ...],
        sensor_paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[tuple[Matrix, Matrix], ...], NativeSensorBatch]:
        """Advance once, then acquire selected state and cameras at the resulting tick."""

        states = self.apply_articulation_commands_step_and_read(commands, count, paths)
        return states, self.read_sensors(sensor_paths)

    def apply_articulation_commands_step_and_read_encoded_sensors(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
        paths: tuple[EntityPath, ...],
        sensor_requests: tuple[Any, ...],
    ) -> tuple[tuple[tuple[Matrix, Matrix], ...], tuple[Any, ...]]:
        """Advance once, then batch-encode selected cameras at the resulting tick."""

        states = self.apply_articulation_commands_step_and_read(commands, count, paths)
        return states, self.read_encoded_sensors(sensor_requests)

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
        position_stiffness = None
        position_damping = None
        if mode is CommandMode.POSITION:
            initial_stiffness, initial_damping = self._initial_usd_articulation_gains[path]
            position_stiffness = initial_stiffness[list(environment_indices)][:, list(joint_ids)]
            position_damping = initial_damping[list(environment_indices)][:, list(joint_ids)]
            position_stiffness, position_damping = _position_command_gains(
                position_stiffness,
                position_damping,
                fallback_authored_zero=self._config.position_stiffness is None,
                fallback_authored_damping=self._config.position_damping is None,
            )
        for row_index, environment in enumerate(environment_indices):
            for column_index, joint in enumerate(joint_ids):
                target = float(targets[row_index][column_index])
                if mode is CommandMode.POSITION:
                    assert position_stiffness is not None and position_damping is not None
                    positions[environment, joint] = target
                    velocities[environment, joint] = 0.0
                    efforts[environment, joint] = 0.0
                    stiffness[environment, joint] = position_stiffness[row_index, column_index]
                    damping[environment, joint] = position_damping[row_index, column_index]
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
        stiffness_indices = self._m.torch.tensor(
            environment_indices,
            device=stiffness.device,
            dtype=self._m.torch.int64,
        )
        damping_indices = self._m.torch.tensor(
            environment_indices,
            device=damping.device,
            dtype=self._m.torch.int64,
        )
        view.set_dof_stiffnesses(stiffness[stiffness_indices], stiffness_indices)
        view.set_dof_dampings(damping[damping_indices], damping_indices)

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

    def read_selected_kinematics(
        self,
        targets: tuple[KinematicTarget, ...],
        environment_index: int = 0,
    ) -> tuple[NativeKinematicState, ...]:
        """Read only requested articulation bodies without admitting geometry."""

        if not 0 <= environment_index < self._spec.environments.count:
            raise IndexError("selected kinematics environment index is out of range")
        origin = self._origins_cpu[environment_index]
        result: list[NativeKinematicState] = []
        for target in targets:
            path = target.entity_path
            if path in self._articulations:
                asset = self._articulations[path]
                if target.link_name is None:
                    pose_row = asset.data.root_link_pose_w.torch[environment_index]
                    velocity_row = asset.data.root_link_vel_w.torch[environment_index]
                else:
                    matches = tuple(
                        index for index, name in enumerate(asset.body_names) if name == target.link_name
                    )
                    if len(matches) != 1:
                        raise KeyError(
                            f"selected link {target.link_name!r} must match exactly one body on {path.value}"
                        )
                    body_index = matches[0]
                    pose_row = asset.data.body_link_pose_w.torch[environment_index, body_index]
                    velocity_row = asset.data.body_link_vel_w.torch[environment_index, body_index]
                row = pose_row.detach().cpu().tolist()
                velocity = velocity_row.detach().cpu().tolist()
            elif path in self._usd_articulation_views:
                articulation_view = self._usd_articulation_views[path]
                if target.link_name is None:
                    row = articulation_view.get_root_transforms()[environment_index].detach().cpu().tolist()
                    velocity = articulation_view.get_root_velocities()[environment_index].detach().cpu().tolist()
                else:
                    key = (path, target.link_name)
                    body_view = self._selected_link_views.get(key)
                    if body_view is None:
                        expected_paths = tuple(
                            self._attachment_body_path(path, target.link_name, environment)
                            for environment in range(self._spec.environments.count)
                        )
                        body_view = self._usd_simulation_view().create_rigid_body_view(list(expected_paths))
                        if (
                            body_view.count != self._spec.environments.count
                            or tuple(body_view.prim_paths) != expected_paths
                        ):
                            raise RuntimeError("selected body view did not preserve environment order")
                        self._selected_link_views[key] = body_view
                    row = body_view.get_transforms()[environment_index].detach().cpu().tolist()
                    velocity = body_view.get_velocities()[environment_index].detach().cpu().tolist()
            else:
                raise KeyError(f"selected kinematics entity {path.value!r} is not an articulation")
            result.append(
                NativeKinematicState(
                    target.target_id,
                    path,
                    target.link_name,
                    tuple(float(row[index]) - float(origin[index]) for index in range(3)),
                    tuple(float(row[index]) for index in range(3, 7)),
                    tuple(float(velocity[index]) for index in range(3)),
                    tuple(float(velocity[index]) for index in range(3, 6)),
                )
            )
        return tuple(result)

    def apply_rigid_body_wrench(
        self,
        path: EntityPath,
        forces_n: Matrix,
        torques_n_m: Matrix,
        environment_indices: tuple[int, ...],
    ) -> None:
        self._pin_physics_activation(path)
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
        stored_forces, stored_torques = self._rigid_wrenches[path]
        stored_forces[list(environment_indices)] = forces[:, 0, :].to(
            device=stored_forces.device,
            dtype=stored_forces.dtype,
        )
        stored_torques[list(environment_indices)] = torques[:, 0, :].to(
            device=stored_torques.device,
            dtype=stored_torques.dtype,
        )
        asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            env_ids=env_ids,
            is_global=True,
        )

    @staticmethod
    def _checkpoint_values(tensor: Any) -> object:
        return tensor.detach().cpu().tolist()

    def _checkpoint_tensor(self, value: object, reference: Any, label: str) -> Any:
        try:
            tensor = self._m.torch.tensor(value, device=reference.device, dtype=reference.dtype)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"checkpoint {label} is not a numeric tensor") from error
        if tuple(int(size) for size in tensor.shape) != tuple(int(size) for size in reference.shape):
            raise ValueError(f"checkpoint {label} shape is invalid")
        if not bool(self._m.torch.isfinite(tensor).all().item()):
            raise ValueError(f"checkpoint {label} contains a non-finite value")
        return tensor

    @staticmethod
    def _checkpoint_map(value: object, expected: set[str], label: str) -> dict[str, object]:
        if type(value) is not dict or set(value) != expected:
            raise ValueError(f"checkpoint {label} identity set is invalid")
        return cast(dict[str, object], value)

    def capture_checkpoint(self) -> dict[str, object]:
        """Capture all mutable physical state without advancing physics or rendering."""

        values = self._checkpoint_values
        articulations: dict[str, object] = {}
        for path, asset in self._articulations.items():
            articulations[path.value] = {
                "kind": "isaaclab",
                "root_pose": values(asset.data.root_link_pose_w.torch),
                "root_velocity": values(asset.data.root_link_vel_w.torch),
                "joint_position": values(asset.data.joint_pos.torch),
                "joint_velocity": values(asset.data.joint_vel.torch),
                "position_target": values(asset.data.joint_pos_target.torch),
                "velocity_target": values(asset.data.joint_vel_target.torch),
                "effort_target": values(asset.data.joint_effort_target.torch),
                "stiffness": values(asset.data.joint_stiffness.torch),
                "damping": values(asset.data.joint_damping.torch),
                "control_modes": [
                    [None if mode is None else mode.value for mode in environment]
                    for environment in self._articulation_control_modes[path]
                ],
            }
        for path, view in self._usd_articulation_views.items():
            articulations[path.value] = {
                "kind": "usd",
                "root_pose": values(view.get_root_transforms()),
                "root_velocity": values(view.get_root_velocities()),
                "joint_position": values(view.get_dof_positions()),
                "joint_velocity": values(view.get_dof_velocities()),
                "position_target": values(view.get_dof_position_targets()),
                "velocity_target": values(view.get_dof_velocity_targets()),
                "effort_target": values(view.get_dof_actuation_forces()),
                "stiffness": values(view.get_dof_stiffnesses()),
                "damping": values(view.get_dof_dampings()),
            }

        rigids: dict[str, object] = {}
        for path, asset in self._rigids.items():
            forces, torques = self._rigid_wrenches[path]
            rigids[path.value] = {
                "kind": "isaaclab",
                "pose": values(asset.data.root_link_pose_w.torch),
                "velocity": values(asset.data.root_link_vel_w.torch),
                "force": values(forces),
                "torque": values(torques),
            }
        for path, view in self._usd_rigid_views.items():
            forces, torques = self._usd_rigid_wrenches[path]
            rigids[path.value] = {
                "kind": "usd",
                "pose": values(view.get_transforms()),
                "velocity": values(view.get_velocities()),
                "force": values(forces),
                "torque": values(torques),
            }

        deformables = {
            path.value: {
                "state": values(asset.data.nodal_state_w.torch),
                "kinematic_target": (
                    None
                    if self._initial_deformable[path][1] is None
                    else values(asset.data.nodal_kinematic_target.torch)
                ),
            }
            for path, asset in self._deformables.items()
        }
        fluids: dict[str, object] = {}
        for path in self._fluids:
            positions, velocities = self.read_particle_fluid(path)
            fluids[path.value] = {"positions": positions, "velocities": velocities}

        composite_rigids = tuple(
            {
                "pose": values(state.view.get_transforms()),
                "velocity": values(state.view.get_velocities()),
            }
            for state in self._composite_rigid_states
        )
        composite_articulations = tuple(
            {
                "root_pose": values(state.view.get_root_transforms()),
                "root_velocity": values(state.view.get_root_velocities()),
                "joint_position": values(state.view.get_dof_positions()),
                "joint_velocity": values(state.view.get_dof_velocities()),
                "position_target": values(state.view.get_dof_position_targets()),
                "velocity_target": values(state.view.get_dof_velocity_targets()),
                "effort_target": values(state.view.get_dof_actuation_forces()),
                "stiffness": values(state.view.get_dof_stiffnesses()),
                "damping": values(state.view.get_dof_dampings()),
            }
            for state in self._composite_articulation_states
        )
        attachments = tuple(
            {
                "attachment_id": attachment.attachment_id,
                "environment_index": attachment.environment_index,
                "parent_path": attachment.parent_path.value,
                "parent_link_name": attachment.parent_link_name,
                "child_path": attachment.child_path.value,
                "child_link_name": attachment.child_link_name,
                "parent_T_child": {
                    "position": attachment.parent_T_child.position,
                    "orientation_xyzw": attachment.parent_T_child.orientation_xyzw,
                },
            }
            for _, attachment in sorted(self._runtime_attachments.items())
        )
        return {
            "schema": "nvidia.isaaclab.native-state/1",
            "articulations": articulations,
            "rigids": rigids,
            "deformables": deformables,
            "fluids": fluids,
            "composite_rigids": composite_rigids,
            "composite_articulations": composite_articulations,
            "attachments": attachments,
        }

    def _stage_checkpoint(self, state: dict[str, object]) -> dict[str, object]:
        if state.get("schema") != "nvidia.isaaclab.native-state/1" or set(state) != {
            "schema",
            "articulations",
            "rigids",
            "deformables",
            "fluids",
            "composite_rigids",
            "composite_articulations",
            "attachments",
        }:
            raise ValueError("checkpoint native schema is invalid")
        staged: dict[str, object] = {
            "articulations": {},
            "rigids": {},
            "deformables": {},
            "fluids": {},
            "composite_rigids": [],
            "composite_articulations": [],
            "attachments": [],
        }
        articulation_records = self._checkpoint_map(
            state.get("articulations"),
            {path.value for path in self._articulations} | {path.value for path in self._usd_articulation_views},
            "articulation",
        )
        staged_articulations = cast(dict[EntityPath, dict[str, Any]], staged["articulations"])
        for path in (*self._articulations, *self._usd_articulation_views):
            record = articulation_records[path.value]
            if type(record) is not dict:
                raise ValueError("checkpoint articulation record is invalid")
            high_level = path in self._articulations
            asset_or_view = self._articulations[path] if high_level else self._usd_articulation_views[path]
            references = (
                {
                    "root_pose": asset_or_view.data.root_link_pose_w.torch,
                    "root_velocity": asset_or_view.data.root_link_vel_w.torch,
                    "joint_position": asset_or_view.data.joint_pos.torch,
                    "joint_velocity": asset_or_view.data.joint_vel.torch,
                    "position_target": asset_or_view.data.joint_pos_target.torch,
                    "velocity_target": asset_or_view.data.joint_vel_target.torch,
                    "effort_target": asset_or_view.data.joint_effort_target.torch,
                    "stiffness": asset_or_view.data.joint_stiffness.torch,
                    "damping": asset_or_view.data.joint_damping.torch,
                }
                if high_level
                else {
                    "root_pose": asset_or_view.get_root_transforms(),
                    "root_velocity": asset_or_view.get_root_velocities(),
                    "joint_position": asset_or_view.get_dof_positions(),
                    "joint_velocity": asset_or_view.get_dof_velocities(),
                    "position_target": asset_or_view.get_dof_position_targets(),
                    "velocity_target": asset_or_view.get_dof_velocity_targets(),
                    "effort_target": asset_or_view.get_dof_actuation_forces(),
                    "stiffness": asset_or_view.get_dof_stiffnesses(),
                    "damping": asset_or_view.get_dof_dampings(),
                }
            )
            expected_kind = "isaaclab" if high_level else "usd"
            expected_fields = {"kind", *references}
            if high_level:
                expected_fields.add("control_modes")
            if record.get("kind") != expected_kind or set(record) != expected_fields:
                raise ValueError("checkpoint articulation representation is invalid")
            staged_record = {
                name: self._checkpoint_tensor(record[name], reference, f"{path.value}.{name}")
                for name, reference in references.items()
            }
            if high_level:
                raw_modes = record["control_modes"]
                expected_modes = self._articulation_control_modes[path]
                if (
                    type(raw_modes) is not list
                    or len(raw_modes) != len(expected_modes)
                    or any(type(row) is not list or len(row) != len(expected_modes[index])
                           for index, row in enumerate(raw_modes))
                ):
                    raise ValueError("checkpoint articulation control modes are invalid")
                try:
                    staged_record["control_modes"] = [
                        [None if value is None else CommandMode(value) for value in row]
                        for row in raw_modes
                    ]
                except (TypeError, ValueError) as error:
                    raise ValueError("checkpoint articulation control modes are invalid") from error
            staged_articulations[path] = staged_record

        rigid_records = self._checkpoint_map(
            state.get("rigids"),
            {path.value for path in self._rigids} | {path.value for path in self._usd_rigid_views},
            "rigid",
        )
        staged_rigids = cast(dict[EntityPath, dict[str, Any]], staged["rigids"])
        for path in (*self._rigids, *self._usd_rigid_views):
            record = rigid_records[path.value]
            if type(record) is not dict:
                raise ValueError("checkpoint rigid record is invalid")
            high_level = path in self._rigids
            asset_or_view = self._rigids[path] if high_level else self._usd_rigid_views[path]
            forces, torques = self._rigid_wrenches[path] if high_level else self._usd_rigid_wrenches[path]
            references = {
                "pose": asset_or_view.data.root_link_pose_w.torch if high_level else asset_or_view.get_transforms(),
                "velocity": (
                    asset_or_view.data.root_link_vel_w.torch if high_level else asset_or_view.get_velocities()
                ),
                "force": forces,
                "torque": torques,
            }
            expected_kind = "isaaclab" if high_level else "usd"
            if record.get("kind") != expected_kind or set(record) != {"kind", *references}:
                raise ValueError("checkpoint rigid representation is invalid")
            staged_rigids[path] = {
                name: self._checkpoint_tensor(record[name], reference, f"{path.value}.{name}")
                for name, reference in references.items()
            }

        deformable_records = self._checkpoint_map(
            state.get("deformables"),
            {path.value for path in self._deformables},
            "deformable",
        )
        staged_deformables = cast(dict[EntityPath, dict[str, Any]], staged["deformables"])
        for path, asset in self._deformables.items():
            record = deformable_records[path.value]
            if type(record) is not dict or set(record) != {"state", "kinematic_target"}:
                raise ValueError("checkpoint deformable record is invalid")
            target_reference = self._initial_deformable[path][1]
            raw_target = record["kinematic_target"]
            if (target_reference is None) != (raw_target is None):
                raise ValueError("checkpoint deformable kinematic target is invalid")
            staged_deformables[path] = {
                "state": self._checkpoint_tensor(
                    record["state"], asset.data.nodal_state_w.torch, f"{path.value}.state"
                ),
                "kinematic_target": (
                    None
                    if target_reference is None
                    else self._checkpoint_tensor(raw_target, target_reference, f"{path.value}.kinematic_target")
                ),
            }

        fluid_records = self._checkpoint_map(
            state.get("fluids"),
            {path.value for path in self._fluids},
            "fluid",
        )
        staged_fluids = cast(dict[EntityPath, tuple[PointBatch, PointBatch]], staged["fluids"])
        for path, sets in self._fluids.items():
            record = fluid_records[path.value]
            if type(record) is not dict or set(record) != {"positions", "velocities"}:
                raise ValueError("checkpoint fluid record is invalid")
            position_rows: list[tuple[Vector3, ...]] = []
            velocity_rows: list[tuple[Vector3, ...]] = []
            for name, output in (("positions", position_rows), ("velocities", velocity_rows)):
                raw = record[name]
                if type(raw) not in {list, tuple} or len(raw) != len(sets):
                    raise ValueError("checkpoint fluid environment count is invalid")
                for environment, row in enumerate(raw):
                    if type(row) not in {list, tuple} or len(row) != len(sets[environment].initial_positions):
                        raise ValueError("checkpoint fluid particle count is invalid")
                    vectors: list[Vector3] = []
                    for vector in row:
                        if type(vector) not in {list, tuple} or len(vector) != 3:
                            raise ValueError("checkpoint fluid vector is invalid")
                        converted = tuple(float(component) for component in vector)
                        if not all(math.isfinite(component) for component in converted):
                            raise ValueError("checkpoint fluid vector is non-finite")
                        vectors.append(cast(Vector3, converted))
                    output.append(tuple(vectors))
            staged_fluids[path] = (tuple(position_rows), tuple(velocity_rows))

        def stage_composites(key: str, states: list[Any], fields: tuple[str, ...]) -> list[dict[str, Any]]:
            raw_records = state.get(key)
            if not isinstance(raw_records, (list, tuple)) or len(raw_records) != len(states):
                raise ValueError(f"checkpoint {key} count is invalid")
            output: list[dict[str, Any]] = []
            for index, (raw, composite) in enumerate(zip(raw_records, states, strict=True)):
                if type(raw) is not dict or set(raw) != set(fields):
                    raise ValueError(f"checkpoint {key} record is invalid")
                getters = {
                    "pose": composite.view.get_transforms,
                    "velocity": composite.view.get_velocities,
                    "root_pose": composite.view.get_root_transforms,
                    "root_velocity": composite.view.get_root_velocities,
                    "joint_position": composite.view.get_dof_positions,
                    "joint_velocity": composite.view.get_dof_velocities,
                    "position_target": composite.view.get_dof_position_targets,
                    "velocity_target": composite.view.get_dof_velocity_targets,
                    "effort_target": composite.view.get_dof_actuation_forces,
                    "stiffness": composite.view.get_dof_stiffnesses,
                    "damping": composite.view.get_dof_dampings,
                }
                output.append(
                    {
                        field: self._checkpoint_tensor(raw[field], getters[field](), f"{key}.{index}.{field}")
                        for field in fields
                    }
                )
            return output

        staged["composite_rigids"] = stage_composites(
            "composite_rigids",
            self._composite_rigid_states,
            ("pose", "velocity"),
        )
        staged["composite_articulations"] = stage_composites(
            "composite_articulations",
            self._composite_articulation_states,
            (
                "root_pose",
                "root_velocity",
                "joint_position",
                "joint_velocity",
                "position_target",
                "velocity_target",
                "effort_target",
                "stiffness",
                "damping",
            ),
        )

        raw_attachments = state.get("attachments")
        if not isinstance(raw_attachments, (list, tuple)):
            raise ValueError("checkpoint attachments are invalid")
        staged_attachments = cast(list[_RuntimeAttachment], staged["attachments"])
        attachment_keys: set[tuple[int, str]] = set()
        attachment_children: set[tuple[int, EntityPath, str | None]] = set()
        expected_attachment_fields = {
            "attachment_id",
            "environment_index",
            "parent_path",
            "parent_link_name",
            "child_path",
            "child_link_name",
            "parent_T_child",
        }
        for raw in raw_attachments:
            if type(raw) is not dict or set(raw) != expected_attachment_fields:
                raise ValueError("checkpoint attachment record is invalid")
            attachment_id = raw["attachment_id"]
            environment = raw["environment_index"]
            parent_value = raw["parent_path"]
            child_value = raw["child_path"]
            parent_link = raw["parent_link_name"]
            child_link = raw["child_link_name"]
            relative_value = raw["parent_T_child"]
            if (
                type(attachment_id) is not str
                or not attachment_id
                or type(environment) is not int
                or not 0 <= environment < self._spec.environments.count
                or type(parent_value) is not str
                or type(child_value) is not str
                or (parent_link is not None and type(parent_link) is not str)
                or (child_link is not None and type(child_link) is not str)
                or type(relative_value) is not dict
                or set(relative_value) != {"position", "orientation_xyzw"}
            ):
                raise ValueError("checkpoint attachment record is invalid")
            try:
                parent = EntityPath(parent_value)
                child = EntityPath(child_value)
                relative = Pose(
                    tuple(relative_value["position"]),
                    tuple(relative_value["orientation_xyzw"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError("checkpoint attachment record is invalid") from error
            key = (environment, attachment_id)
            child_key = (environment, child, child_link)
            if (
                parent not in self._entity_specs
                or child not in self._entity_specs
                or key in attachment_keys
                or child_key in attachment_children
            ):
                raise ValueError("checkpoint attachment identity is invalid")
            attachment_keys.add(key)
            attachment_children.add(child_key)
            staged_attachments.append(
                _RuntimeAttachment(
                    attachment_id,
                    environment,
                    parent,
                    parent_link,
                    child,
                    child_link,
                    relative,
                    "",
                )
            )
        return staged

    def _apply_staged_checkpoint(self, staged: dict[str, object]) -> None:
        stage = self._m.sim_utils.get_current_stage()
        for attachment in tuple(self._runtime_attachments.values()):
            stage.RemovePrim(attachment.joint_prim_path)
        self._runtime_attachments.clear()

        env_ids = list(range(self._spec.environments.count))
        articulations = cast(dict[EntityPath, dict[str, Any]], staged["articulations"])
        for path, asset in self._articulations.items():
            record = articulations[path]
            asset.reset(env_ids=env_ids)
            asset.write_root_pose_to_sim_index(root_pose=record["root_pose"], env_ids=env_ids)
            asset.write_root_link_velocity_to_sim_index(root_velocity=record["root_velocity"], env_ids=env_ids)
            asset.write_joint_position_to_sim_index(position=record["joint_position"], env_ids=env_ids)
            asset.write_joint_velocity_to_sim_index(velocity=record["joint_velocity"], env_ids=env_ids)
            asset.set_joint_position_target_index(target=record["position_target"], env_ids=env_ids)
            asset.set_joint_velocity_target_index(target=record["velocity_target"], env_ids=env_ids)
            asset.set_joint_effort_target_index(target=record["effort_target"], env_ids=env_ids)
            asset.write_joint_stiffness_to_sim_index(stiffness=record["stiffness"], env_ids=env_ids)
            asset.write_joint_damping_to_sim_index(damping=record["damping"], env_ids=env_ids)
            asset.write_data_to_sim()
            self._articulation_control_modes[path] = [list(row) for row in record["control_modes"]]
        for path, view in self._usd_articulation_views.items():
            record = articulations[path]
            indices = self._m.torch.arange(
                view.count,
                device=record["joint_position"].device,
                dtype=self._m.torch.int64,
            )
            view.set_root_transforms(record["root_pose"], indices)
            view.set_root_velocities(record["root_velocity"], indices)
            view.set_dof_positions(record["joint_position"], indices)
            view.set_dof_velocities(record["joint_velocity"], indices)
            view.set_dof_position_targets(record["position_target"], indices)
            view.set_dof_velocity_targets(record["velocity_target"], indices)
            view.set_dof_actuation_forces(record["effort_target"], indices)
            view.set_dof_stiffnesses(record["stiffness"], indices)
            view.set_dof_dampings(record["damping"], indices)

        rigids = cast(dict[EntityPath, dict[str, Any]], staged["rigids"])
        for path, asset in self._rigids.items():
            record = rigids[path]
            _write_high_level_rigid_state(
                asset,
                record["pose"],
                record["velocity"],
                env_ids,
                kinematic=self._kinematic_rigids[path],
            )
            forces, torques = self._rigid_wrenches[path]
            forces.copy_(record["force"])
            torques.copy_(record["torque"])
            asset.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces.unsqueeze(1),
                torques=torques.unsqueeze(1),
                env_ids=env_ids,
                is_global=True,
            )
            self._contacts[path].reset(env_ids=env_ids)
        for path, view in self._usd_rigid_views.items():
            record = rigids[path]
            indices = self._m.torch.arange(view.count, device=record["pose"].device, dtype=self._m.torch.int64)
            _write_usd_rigid_state(
                view,
                record["pose"],
                record["velocity"],
                indices,
                kinematic=self._kinematic_rigids[path],
            )
            forces, torques = self._usd_rigid_wrenches[path]
            forces.copy_(record["force"])
            torques.copy_(record["torque"])

        deformables = cast(dict[EntityPath, dict[str, Any]], staged["deformables"])
        for path, asset in self._deformables.items():
            record = deformables[path]
            asset.write_nodal_state_to_sim_index(record["state"], env_ids=env_ids)
            if record["kinematic_target"] is not None:
                asset.write_nodal_kinematic_target_to_sim_index(record["kinematic_target"], env_ids=env_ids)
            asset.reset(env_ids=env_ids)

        fluids = cast(dict[EntityPath, tuple[PointBatch, PointBatch]], staged["fluids"])
        for path, (positions, velocities) in fluids.items():
            for environment, fluid_set in enumerate(self._fluids[path]):
                fluid_set.points.GetPointsAttr().Set(self._m.Vt.Vec3fArray(positions[environment]))
                fluid_set.points.GetVelocitiesAttr().Set(self._m.Vt.Vec3fArray(velocities[environment]))

        for rigid_state, record in zip(
            self._composite_rigid_states,
            cast(list[dict[str, Any]], staged["composite_rigids"]),
            strict=True,
        ):
            indices = self._m.torch.arange(
                rigid_state.view.count, device=record["pose"].device, dtype=self._m.torch.int64
            )
            _write_usd_rigid_state(
                rigid_state.view,
                record["pose"],
                record["velocity"],
                indices,
                kinematic=rigid_state.kinematic,
            )
        for articulation_state, record in zip(
            self._composite_articulation_states,
            cast(list[dict[str, Any]], staged["composite_articulations"]),
            strict=True,
        ):
            indices = self._m.torch.arange(
                articulation_state.view.count,
                device=record["joint_position"].device,
                dtype=self._m.torch.int64,
            )
            articulation_state.view.set_root_transforms(record["root_pose"], indices)
            articulation_state.view.set_root_velocities(record["root_velocity"], indices)
            articulation_state.view.set_dof_positions(record["joint_position"], indices)
            articulation_state.view.set_dof_velocities(record["joint_velocity"], indices)
            articulation_state.view.set_dof_position_targets(record["position_target"], indices)
            articulation_state.view.set_dof_velocity_targets(record["velocity_target"], indices)
            articulation_state.view.set_dof_actuation_forces(record["effort_target"], indices)
            articulation_state.view.set_dof_stiffnesses(record["stiffness"], indices)
            articulation_state.view.set_dof_dampings(record["damping"], indices)

        assert self._sim is not None
        self._sim.forward()
        self._update_assets(0.0)
        for attachment in cast(list[_RuntimeAttachment], staged["attachments"]):
            self.attach_rigid_body(
                attachment.attachment_id,
                attachment.parent_path,
                attachment.parent_link_name,
                attachment.child_path,
                attachment.child_link_name,
                attachment.environment_index,
                attachment.parent_T_child,
            )
        self._update_assets(0.0)
        self._sync_all_mounted_cameras()
        self._invalidate_render()

    def restore_checkpoint(self, state: dict[str, object]) -> None:
        """Preflight a complete payload and roll back if native application fails."""

        if type(state) is not dict:
            raise ValueError("checkpoint native state must be a dictionary")
        staged = self._stage_checkpoint(state)
        rollback = self._stage_checkpoint(self.capture_checkpoint())
        try:
            self._apply_staged_checkpoint(staged)
        except BaseException:
            self._apply_staged_checkpoint(rollback)
            raise

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

    def _entity_prim_path(self, path: EntityPath, environment_index: int) -> str:
        key = (path, environment_index)
        cached = self._entity_prim_path_cache.get(key)
        if cached is not None:
            return cached
        entity = self._entity_specs.get(path)
        if entity is None:
            raise KeyError(f"entity {path.value!r} does not exist")
        if not 0 <= environment_index < self._spec.environments.count:
            raise IndexError("entity Prim environment index is out of range")
        binding = entity.embedded_binding
        if binding is None:
            result = f"/World/env_{environment_index}/{_native_name(path)}"
            self._entity_prim_path_cache[key] = result
            return result
        roots = self._composite_scene_roots.get(binding.container_path)
        if roots is None or environment_index >= len(roots):
            raise KeyError(f"embedded entity container for {path.value!r} is unavailable")
        relative_paths = tuple(
            item.relative_prim_path for item in (*binding.link_prims, *binding.joint_prims)
        )
        parts = tuple(relative.split("/") for relative in relative_paths)
        common: list[str] = []
        for values in zip(*parts, strict=False):
            if len(set(values)) != 1:
                break
            common.append(values[0])
        if not common:
            raise ValueError(f"embedded entity {path.value!r} has no distinct common USD Prim")
        result = f"{roots[environment_index]}/{'/'.join(common)}"
        self._entity_prim_path_cache[key] = result
        return result

    def _entity_prim_is_physical_root(self, path: EntityPath, environment_index: int) -> bool:
        key = (path, environment_index)
        cached = self._entity_prim_physical_root_cache.get(key)
        if cached is not None:
            return cached
        entity = self._entity_specs[path]
        if entity.kind not in {EntityKind.RIGID_BODY, EntityKind.ARTICULATION}:
            self._entity_prim_physical_root_cache[key] = False
            return False
        try:
            result = self._entity_prim_path(path, environment_index) == self._attachment_body_path(
                path, None, environment_index
            )
        except (KeyError, ValueError):
            result = False
        self._entity_prim_physical_root_cache[key] = result
        return result

    def _read_usd_prim_pose(self, prim_path: str, environment_index: int) -> Pose:
        stage = self._m.sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not self._m.UsdGeom.Xformable(prim):
            raise ValueError(f"USD Prim is absent or not transformable: {prim_path}")
        matrix = self._m.UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        origin = self._origins_cpu[environment_index]
        return _pose_from_world_matrix(
            matrix,
            (float(origin[0]), float(origin[1]), float(origin[2])),
        )

    def _read_usd_entity_prim_pose(self, path: EntityPath, environment_index: int) -> Pose:
        return self._read_usd_prim_pose(
            self._entity_prim_path(path, environment_index),
            environment_index,
        )

    def _write_entity_prim_pose(self, path: EntityPath, environment_index: int, pose: Pose) -> None:
        stage = self._m.sim_utils.get_current_stage()
        prim_path = self._entity_prim_path(path, environment_index)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not self._m.UsdGeom.Xformable(prim):
            raise ValueError(f"entity USD Prim is absent or not transformable: {prim_path}")
        self._m.sim_utils.standardize_xform_ops(prim)
        parent = prim.GetParent()
        origin = self._origins_cpu[environment_index]
        absolute_position = tuple(float(pose.position[index]) + float(origin[index]) for index in range(3))
        desired = self._m.Gf.Matrix4d(1.0)
        desired.SetRotateOnly(
            self._m.Gf.Quatd(
                float(pose.orientation_xyzw[3]),
                self._m.Gf.Vec3d(*pose.orientation_xyzw[:3]),
            )
        )
        desired.SetTranslateOnly(self._m.Gf.Vec3d(*absolute_position))
        if parent and parent.IsValid() and str(parent.GetPath()) != "/":
            parent_world = self._m.UsdGeom.XformCache().GetLocalToWorldTransform(parent)
            local = desired * parent_world.GetInverse()
        else:
            local = desired
        translation = local.ExtractTranslation()
        orientation_xyzw = _normalized_quaternion_xyzw(
            local.RemoveScaleShear().ExtractRotationQuat()
        )
        translate_attribute = prim.GetAttribute("xformOp:translate")
        orient_attribute = prim.GetAttribute("xformOp:orient")
        if not translate_attribute or not orient_attribute:
            raise ValueError(f"entity USD Prim has no standardized pose attributes: {prim_path}")
        translate_attribute.Set(translation)
        orientation_value = orient_attribute.Get()
        if isinstance(orientation_value, self._m.Gf.Quatf):
            orientation = self._m.Gf.Quatf(
                orientation_xyzw[3],
                self._m.Gf.Vec3f(*orientation_xyzw[:3]),
            )
        elif isinstance(orientation_value, self._m.Gf.Quath):
            orientation = self._m.Gf.Quath(
                orientation_xyzw[3],
                self._m.Gf.Vec3h(*orientation_xyzw[:3]),
            )
        else:
            orientation = self._m.Gf.Quatd(
                orientation_xyzw[3],
                self._m.Gf.Vec3d(*orientation_xyzw[:3]),
            )
        orient_attribute.Set(orientation)

    def read_entity_prim_states(
        self,
        paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[NativeEntityPrimState, ...], ...]:
        result: list[tuple[NativeEntityPrimState, ...]] = []
        for path in paths:
            if path not in self._entity_specs:
                raise KeyError(f"entity {path.value!r} does not exist")
            entity = self._entity_specs[path]
            physical_root = tuple(
                self._entity_prim_is_physical_root(path, environment)
                for environment in range(self._spec.environments.count)
            )
            rigid_state = (
                self.read_rigid_body(path)
                if entity.kind is EntityKind.RIGID_BODY and any(physical_root)
                else None
            )
            row: list[NativeEntityPrimState] = []
            for environment in range(self._spec.environments.count):
                if physical_root[environment]:
                    if entity.kind is EntityKind.ARTICULATION:
                        state = self.read_selected_kinematics(
                            (KinematicTarget("entity-prim", path, None),), environment
                        )[0]
                        pose = Pose(state.position_m, state.orientation_xyzw)
                        linear = state.linear_velocity_m_s
                        angular = state.angular_velocity_rad_s
                    else:
                        assert rigid_state is not None
                        positions, orientations, linear_rows, angular_rows = rigid_state
                        position_row = positions[environment]
                        orientation_row = orientations[environment]
                        linear_row = linear_rows[environment]
                        angular_row = angular_rows[environment]
                        pose = Pose(
                            (float(position_row[0]), float(position_row[1]), float(position_row[2])),
                            (
                                float(orientation_row[0]),
                                float(orientation_row[1]),
                                float(orientation_row[2]),
                                float(orientation_row[3]),
                            ),
                        )
                        linear = (float(linear_row[0]), float(linear_row[1]), float(linear_row[2]))
                        angular = (float(angular_row[0]), float(angular_row[1]), float(angular_row[2]))
                else:
                    pose = self._read_usd_entity_prim_pose(path, environment)
                    linear = angular = (0.0, 0.0, 0.0)
                row.append(NativeEntityPrimState(pose, linear, angular))
            result.append(tuple(row))
        return tuple(result)

    def _set_physical_rigid_body_pose(
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
            _write_usd_rigid_state(
                view,
                transforms[indices],
                self._m.torch.zeros((1, 6), device=transforms.device, dtype=transforms.dtype),
                indices,
                kinematic=self._kinematic_rigids[path],
            )
        else:
            asset = self._rigids[path]
            assert self._sim is not None and self._origins is not None
            pose = self._m.torch.tensor(
                ((*position_m, *orientation_xyzw),), device=self._sim.device, dtype=self._m.torch.float32
            )
            pose[:, :3] += self._origins[environment_index : environment_index + 1]
            env_ids = self._m.torch.tensor((environment_index,), device=self._sim.device, dtype=self._m.torch.int64)
            _write_high_level_rigid_state(
                asset,
                pose,
                self._m.torch.zeros((1, 6), device=self._sim.device, dtype=self._m.torch.float32),
                env_ids,
                kinematic=self._kinematic_rigids[path],
            )

    def set_rigid_body_pose(
        self,
        path: EntityPath,
        position_m: Vector3,
        orientation_xyzw: tuple[float, float, float, float],
        environment_index: int,
    ) -> None:
        """Private physical-body helper retained for adapter unit tests.

        Entity-level callers must use :meth:`set_entity_prim_pose`.
        """

        self._pin_physics_activation(path)
        self._set_physical_rigid_body_pose(path, position_m, orientation_xyzw, environment_index)
        assert self._sim is not None
        self._sim.forward()
        self._update_assets(0.0)
        self._sync_all_mounted_cameras()
        self._invalidate_render()

    def _set_physical_articulation_root_pose(
        self,
        path: EntityPath,
        pose: Pose,
        environment_index: int,
    ) -> None:
        if path in self._usd_articulation_views:
            view = self._usd_articulation_views[path]
            transforms = view.get_root_transforms().clone()
            origin = self._m.torch.tensor(
                self._origins_cpu[environment_index], device=transforms.device, dtype=transforms.dtype
            )
            transforms[environment_index, :3] = (
                self._m.torch.tensor(pose.position, device=transforms.device, dtype=transforms.dtype) + origin
            )
            transforms[environment_index, 3:] = self._m.torch.tensor(
                pose.orientation_xyzw, device=transforms.device, dtype=transforms.dtype
            )
            indices = self._m.torch.tensor((environment_index,), device=transforms.device, dtype=self._m.torch.int64)
            view.set_root_transforms(transforms[indices], indices)
            view.set_root_velocities(
                self._m.torch.zeros((1, 6), device=transforms.device, dtype=transforms.dtype), indices
            )
            return
        asset = self._articulations[path]
        assert self._sim is not None and self._origins is not None
        root_pose = self._m.torch.tensor(
            ((*pose.position, *pose.orientation_xyzw),), device=self._sim.device, dtype=self._m.torch.float32
        )
        root_pose[:, :3] += self._origins[environment_index : environment_index + 1]
        env_ids = self._m.torch.tensor((environment_index,), device=self._sim.device, dtype=self._m.torch.int64)
        asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(
            root_velocity=self._m.torch.zeros((1, 6), device=self._sim.device, dtype=self._m.torch.float32),
            env_ids=env_ids,
        )

    def set_entity_prim_pose(
        self,
        path: EntityPath,
        position_m: Vector3,
        orientation_xyzw: tuple[float, float, float, float],
        environment_index: int,
    ) -> None:
        entity = self._entity_specs.get(path)
        if entity is None or entity.kind not in {EntityKind.RIGID_BODY, EntityKind.ARTICULATION}:
            raise ValueError("entity Prim pose can be set only for a rigid body or articulation")
        self._pin_physics_activation(path)
        target = Pose(position_m, orientation_xyzw)
        old_entity = self.read_entity_prim_states((path,))[0][environment_index].pose
        old_physical_root = self._attachment_endpoint_pose(path, None, environment_index)
        target_physical_root = _retarget_physical_root_pose(target, old_entity, old_physical_root)
        if not self._entity_prim_is_physical_root(path, environment_index):
            self._write_entity_prim_pose(path, environment_index, target)
        if entity.kind is EntityKind.ARTICULATION:
            self._set_physical_articulation_root_pose(path, target_physical_root, environment_index)
        else:
            self._set_physical_rigid_body_pose(
                path,
                target_physical_root.position,
                target_physical_root.orientation_xyzw,
                environment_index,
            )
        assert self._sim is not None
        self._sim.forward()
        self._update_assets(0.0)
        self._sync_all_mounted_cameras()
        self._invalidate_render()

    def _attachment_body_path(
        self,
        path: EntityPath,
        link_name: str | None,
        environment_index: int,
    ) -> str:
        entity = self._entity_specs.get(path)
        if entity is None:
            raise KeyError(f"attachment entity {path.value!r} does not exist")
        if entity.embedded_binding is not None:
            binding = entity.embedded_binding
            roots = self._composite_scene_roots.get(binding.container_path)
            if roots is None or environment_index >= len(roots):
                raise KeyError(f"attachment container for {path.value!r} is unavailable")
            if link_name is None:
                relative_path = binding.root_body_prim_path
            else:
                matches = tuple(
                    item.relative_prim_path
                    for item in binding.link_prims
                    if item.logical_name == link_name
                )
                if len(matches) != 1:
                    raise ValueError(
                        f"attachment link {link_name!r} on {path.value!r} must identify exactly one "
                        f"embedded rigid body; found {len(matches)}"
                    )
                relative_path = matches[0]
            prim = self._embedded_prim(
                roots[environment_index],
                relative_path,
                entity=entity,
                role=f"attachment link {link_name!r}",
            )
            if not prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI):
                raise ValueError("an embedded attachment endpoint must be a rigid-body Prim")
            return self._prim_path_string(prim)
        root = f"/World/env_{environment_index}/{_native_name(path)}"
        if entity.kind is EntityKind.ARTICULATION:
            return f"{root}{_articulation_mount_body_suffix(self._m, root, link_name)}"
        if entity.kind is not EntityKind.RIGID_BODY:
            raise ValueError("attachment endpoints must be rigid bodies or articulations")
        bodies = tuple(
            self._m.sim_utils.get_all_matching_child_prims(
                root,
                lambda prim: prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI),
            )
        )
        if link_name is None:
            matches = bodies
        else:
            matches = tuple(prim for prim in bodies if prim.GetName() == link_name)
        if len(matches) != 1:
            raise ValueError(
                f"attachment endpoint {path.value!r} link {link_name!r} must identify exactly one rigid body; "
                f"found {len(matches)}"
            )
        return self._prim_path_string(matches[0])

    def _attachment_endpoint_pose(
        self,
        path: EntityPath,
        link_name: str | None,
        environment_index: int,
    ) -> Pose:
        entity = self._entity_specs[path]
        if entity.kind is EntityKind.ARTICULATION:
            state = self.read_selected_kinematics(
                (KinematicTarget("attachment-endpoint", path, link_name),),
                environment_index,
            )[0]
            return Pose(state.position_m, state.orientation_xyzw)
        positions, orientations, _linear, _angular = self.read_rigid_body(path)
        return Pose(
            cast(tuple[float, float, float], positions[environment_index]),
            cast(tuple[float, float, float, float], orientations[environment_index]),
        )

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
        if not 0 <= environment_index < self._spec.environments.count:
            raise IndexError("attachment environment index is out of range")
        key = (environment_index, attachment_id)
        if key in self._runtime_attachments:
            raise ValueError("attachment ID is already active")
        if any(
            attachment.environment_index == environment_index
            and attachment.child_path == child_path
            and attachment.child_link_name == child_link_name
            for attachment in self._runtime_attachments.values()
        ):
            raise ValueError("attachment child already has an active parent")
        parent_body_path = self._attachment_body_path(parent_path, parent_link_name, environment_index)
        child_body_path = self._attachment_body_path(child_path, child_link_name, environment_index)
        if parent_body_path == child_body_path:
            raise ValueError("attachment endpoints resolve to the same physical body")
        self._pin_physics_activation(parent_path)
        self._pin_physics_activation(child_path)
        parent_endpoint_pose = self._attachment_endpoint_pose(
            parent_path, parent_link_name, environment_index
        )
        child_endpoint_pose = self._attachment_endpoint_pose(
            child_path, child_link_name, environment_index
        )
        relative, parent_body_T_joint, child_body_T_joint = _attachment_joint_frames(
            parent_endpoint_pose,
            child_endpoint_pose,
            parent_T_child,
        )
        digest = hashlib.sha256(attachment_id.encode("utf-8")).hexdigest()[:24]
        joint_root = f"/World/env_{environment_index}/unirobosim_runtime_attachments"
        joint_path = f"{joint_root}/fixed_{digest}"
        stage = self._m.sim_utils.get_current_stage()
        stage.DefinePrim(joint_root, "Scope")
        if stage.GetPrimAtPath(joint_path).IsValid():
            raise ValueError("attachment joint Prim path is already occupied")
        fixed = self._m.UsdPhysics.FixedJoint.Define(stage, joint_path)
        fixed.CreateBody0Rel().SetTargets((self._m.Sdf.Path(parent_body_path),))
        fixed.CreateBody1Rel().SetTargets((self._m.Sdf.Path(child_body_path),))
        fixed.CreateLocalPos0Attr().Set(self._m.Gf.Vec3f(*parent_body_T_joint.position))
        fixed.CreateLocalRot0Attr().Set(
            self._m.Gf.Quatf(
                parent_body_T_joint.orientation_xyzw[3],
                self._m.Gf.Vec3f(*parent_body_T_joint.orientation_xyzw[:3]),
            )
        )
        fixed.CreateLocalPos1Attr().Set(self._m.Gf.Vec3f(*child_body_T_joint.position))
        fixed.CreateLocalRot1Attr().Set(
            self._m.Gf.Quatf(
                child_body_T_joint.orientation_xyzw[3],
                self._m.Gf.Vec3f(*child_body_T_joint.orientation_xyzw[:3]),
            )
        )
        fixed.CreateJointEnabledAttr().Set(True)
        # The two constrained bodies must not simultaneously repel one another;
        # contacts with every other scene body remain enabled.
        fixed.CreateCollisionEnabledAttr().Set(False)
        fixed.CreateExcludeFromArticulationAttr().Set(True)
        self._runtime_attachments[key] = _RuntimeAttachment(
            attachment_id,
            environment_index,
            parent_path,
            parent_link_name,
            child_path,
            child_link_name,
            relative,
            joint_path,
        )
        assert self._sim is not None
        self._sim.forward()
        return relative

    def detach_rigid_body(
        self,
        attachment_id: str,
        child_path: EntityPath,
        environment_index: int,
    ) -> None:
        key = (environment_index, attachment_id)
        attachment = self._runtime_attachments.get(key)
        if attachment is None or attachment.child_path != child_path:
            raise KeyError("attachment is missing or belongs to another child")
        stage = self._m.sim_utils.get_current_stage()
        if not stage.RemovePrim(attachment.joint_prim_path):
            raise RuntimeError("failed to remove the runtime fixed joint")
        del self._runtime_attachments[key]
        assert self._sim is not None
        self._sim.forward()
        self._update_assets(0.0)
        self._invalidate_render()

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
        self._invalidate_render()

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

    def _read_camera_channels(self, camera: Any, entity: EntitySpec) -> NativeSensorSample:
        assert entity.camera is not None
        channels = []
        for modality in entity.camera.modalities:
            native_name = _camera_native_data_type(modality)
            value = camera.data.output[native_name]
            tensor = getattr(value, "torch", value)
            channel_values: tuple[float | int, ...] | bytes
            if modality is CameraModality.RGB:
                # Keep conversion and layout normalization on the GPU, then cross
                # the worker boundary as one compact CPU buffer.  Expanding a
                # 1080p image into millions of Python integers dominated camera
                # acquisition and doubled the multiprocessing payload.
                tensor = tensor.to(dtype=self._m.torch.uint8).contiguous()
                shape = tuple(int(size) for size in tensor.shape)
                channel_values = tensor.detach().cpu().numpy().tobytes(order="C")
            elif modality is CameraModality.DEPTH:
                tensor = tensor[..., 0]
                valid = self._m.torch.isfinite(tensor)
                valid &= tensor >= entity.camera.near_plane_m
                valid &= tensor <= entity.camera.far_plane_m
                tensor = self._m.torch.where(valid, tensor, self._m.torch.zeros_like(tensor))
                tensor = tensor.to(dtype=self._m.torch.float32)
                shape = tuple(int(size) for size in tensor.shape)
                channel_values = tuple(float(item) for item in tensor.detach().cpu().reshape(-1).tolist())
            else:
                if tensor.shape[-1] != 3:
                    raise RuntimeError("native camera normals must have exactly three channels")
                tensor = tensor.to(dtype=self._m.torch.float32)
                tensor = self._m.torch.where(
                    self._m.torch.isfinite(tensor),
                    tensor,
                    self._m.torch.zeros_like(tensor),
                )
                shape = tuple(int(size) for size in tensor.shape)
                channel_values = tuple(float(item) for item in tensor.detach().cpu().reshape(-1).tolist())
            channels.append((modality, shape, channel_values))
        return tuple(channels)

    def read_sensor(self, path: EntityPath) -> NativeSensorSample:
        camera = self._cameras[path]
        entity = next(item for item in self._spec.entities if item.path == path)
        assert entity.camera is not None
        assert self._sim is not None
        self._ensure_camera_render()
        camera.update(0.0, force_recompute=True)
        return self._read_camera_channels(camera, entity)

    def read_sensors(self, paths: tuple[EntityPath, ...]) -> NativeSensorBatch:
        """Read ordered cameras after one shared render and, when possible, one RGB transfer."""

        entity_by_path = {
            entity.path: entity
            for entity in self._spec.entities
            if entity.kind is EntityKind.CAMERA_SENSOR and entity.camera is not None
        }
        targets = []
        for path in paths:
            camera = self._cameras.get(path)
            entity = entity_by_path.get(path)
            if camera is None or entity is None:
                raise KeyError(f"native camera does not exist: {path.value}")
            targets.append((camera, entity))
        if not targets:
            return ()

        assert self._sim is not None
        self._ensure_camera_render()
        for camera, _ in targets:
            camera.update(0.0, force_recompute=True)

        if all(
            entity.camera is not None and entity.camera.modalities == (CameraModality.RGB,) for _, entity in targets
        ):
            rgb_tensors = tuple(
                getattr(camera.data.output["rgb"], "torch", camera.data.output["rgb"]) for camera, _ in targets
            )
            packed = _pack_compatible_rgb_tensors(self._m.torch, rgb_tensors, self._rgb_host_staging)
            if packed is not None:
                shape, payloads = packed
                return tuple(((CameraModality.RGB, shape, payload),) for payload in payloads)

        # Mixed modalities or image shapes retain the generic conversion path.
        # Rendering and camera updates have already happened once for the whole
        # request, so fallback cannot create additional frames or skew the tick.
        return tuple(self._read_camera_channels(camera, entity) for camera, entity in targets)

    def read_sensors_into_shared(
        self,
        paths: tuple[EntityPath, ...],
        target: memoryview,
    ) -> tuple[tuple[tuple[int, ...], int, int], ...] | None:
        """Write an RGB-only camera batch directly into registered shared host memory."""

        entity_by_path = {
            entity.path: entity
            for entity in self._spec.entities
            if entity.kind is EntityKind.CAMERA_SENSOR and entity.camera is not None
        }
        targets = []
        for path in paths:
            camera = self._cameras.get(path)
            entity = entity_by_path.get(path)
            if camera is None or entity is None:
                raise KeyError(f"native camera does not exist: {path.value}")
            targets.append((camera, entity))
        if not targets or not all(
            entity.camera is not None and entity.camera.modalities == (CameraModality.RGB,)
            for _, entity in targets
        ):
            return None

        assert self._sim is not None
        self._ensure_camera_render()
        for camera, _ in targets:
            camera.update(0.0, force_recompute=True)
        tensors = tuple(
            getattr(camera.data.output["rgb"], "torch", camera.data.output["rgb"])
            for camera, _ in targets
        )
        devices = tuple(getattr(tensor, "device", None) for tensor in tensors)
        if any(getattr(device, "type", None) != "cuda" for device in devices):
            return None
        device = devices[0]
        if any(str(candidate) != str(device) for candidate in devices[1:]):
            return None
        normalized = tuple(tensor.to(dtype=self._m.torch.uint8).contiguous() for tensor in tensors)
        shape = tuple(int(size) for size in normalized[0].shape)
        if any(tuple(int(size) for size in tensor.shape) != shape for tensor in normalized[1:]):
            return None
        sample_size = math.prod(shape)
        required = sample_size * len(normalized)
        if required > len(target):
            raise RuntimeError("native RGB batch exceeds its shared host allocation")

        host = self._rgb_shared_host
        if host is None:
            host = self._m.torch.frombuffer(target, dtype=self._m.torch.uint8)
            result = self._m.torch.cuda.cudart().cudaHostRegister(host.data_ptr(), host.numel(), 0)
            if result != 0:
                raise RuntimeError(f"CUDA could not register the RGB shared host allocation: {result}")
            self._rgb_shared_host = host
            self._rgb_shared_host_size = int(host.numel())
        elif self._rgb_shared_host_size != len(target):
            raise RuntimeError("native RGB shared host allocation changed while the world was live")

        for index, tensor in enumerate(normalized):
            offset = index * sample_size
            host[offset : offset + sample_size].view(shape).copy_(tensor, non_blocking=True)
        self._m.torch.cuda.current_stream(device=device).synchronize()
        return tuple((shape, index * sample_size, sample_size) for index in range(len(normalized)))

    def read_encoded_sensors(self, requests: tuple[Any, ...]) -> tuple[Any, ...]:
        """Encode one coherent RGB camera batch on CUDA and return compressed bytes."""

        from .native_protocols import NativeEncodedSensorFrame

        if not requests:
            return ()
        entity_by_path = {
            entity.path: entity
            for entity in self._spec.entities
            if entity.kind is EntityKind.CAMERA_SENSOR and entity.camera is not None
        }
        targets: list[tuple[Any, Any, Any]] = []
        for request in requests:
            path = request.path
            camera = self._cameras.get(path)
            entity = entity_by_path.get(path)
            if camera is None or entity is None or entity.camera is None:
                raise KeyError(f"native camera does not exist: {path.value}")
            if entity.camera.modalities != (CameraModality.RGB,):
                raise RuntimeError("backend JPEG encoding requires an RGB-only camera")
            if (
                request.encoding != "jpeg"
                or type(request.quality) is not int
                or not 1 <= request.quality <= 100
                or request.color_space != "srgb"
                or request.chroma_subsampling != "4:4:4"
            ):
                raise RuntimeError("Isaac encoded camera path supports only JPEG sRGB 4:4:4")
            targets.append((camera, entity, request))

        assert self._sim is not None
        self._ensure_camera_render()
        for camera, _, _ in targets:
            camera.update(0.0, force_recompute=True)

        try:
            from torchvision.io import encode_jpeg
        except Exception as exc:
            raise RuntimeError("torchvision CUDA JPEG encoder is unavailable") from exc

        tensors = []
        for camera, _, _ in targets:
            tensor = getattr(camera.data.output["rgb"], "torch", camera.data.output["rgb"])
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 3 or tensor.shape[-1] != 3 or tensor.device.type != "cuda":
                raise RuntimeError("native camera RGB must be one CUDA-resident HWC image")
            tensors.append(tensor.to(dtype=self._m.torch.uint8).permute(2, 0, 1).contiguous())

        payload_by_index: dict[int, bytes] = {}
        for quality in sorted({request.quality for _, _, request in targets}):
            assert type(quality) is int
            indices = [index for index, (_, _, request) in enumerate(targets) if request.quality == quality]
            encoded = encode_jpeg([tensors[index] for index in indices], quality=quality)
            if not isinstance(encoded, list) or len(encoded) != len(indices):
                raise RuntimeError("CUDA JPEG encoder returned an invalid batch")
            for index, payload in zip(indices, encoded, strict=True):
                data = bytes(payload.cpu().numpy())
                if not data.startswith(b"\xff\xd8\xff"):
                    raise RuntimeError("CUDA JPEG encoder returned an invalid marker")
                payload_by_index[index] = data

        return tuple(
            NativeEncodedSensorFrame(
                path=request.path,
                payload=payload_by_index[index],
                width_px=int(tensor.shape[2]),
                height_px=int(tensor.shape[1]),
                encoding=request.encoding,
                quality=request.quality,
                color_space=request.color_space,
                chroma_subsampling=request.chroma_subsampling,
            )
            for index, (tensor, (_, _, request)) in enumerate(zip(tensors, targets, strict=True))
        )

    def camera_calibration(self, path: EntityPath) -> NativeCameraCalibration:
        """Read the effective camera state authored into Isaac/USD."""

        camera = self._cameras[path]
        self._sync_mounted_camera(path)
        camera.update(0.0, force_recompute=True)
        data = camera.data
        if data.image_shape is None:
            raise RuntimeError("native camera has no effective image shape")
        height, width = (int(value) for value in data.image_shape)
        matrices = data.intrinsic_matrices.torch.detach().cpu()
        positions = data.pos_w.torch.detach().cpu()
        orientations = data.quat_w_opengl.torch.detach().cpu()
        if matrices.shape != (1, 3, 3) or positions.shape != (1, 3) or orientations.shape != (1, 4):
            raise RuntimeError("native camera calibration requires exactly one initialized view")
        sensor_prims = tuple(camera._sensor_prims)
        if len(sensor_prims) != 1:
            raise RuntimeError("native camera calibration requires exactly one USD camera prim")
        sensor_prim = sensor_prims[0]
        clipping = sensor_prim.GetClippingRangeAttr().Get()
        return NativeCameraCalibration(
            resolution_px=(width, height),
            intrinsic_matrix=tuple(float(value) for value in matrices[0].reshape(-1).tolist()),
            projection=str(sensor_prim.GetProjectionAttr().Get()),
            focal_length=float(sensor_prim.GetFocalLengthAttr().Get()),
            horizontal_aperture=float(sensor_prim.GetHorizontalApertureAttr().Get()),
            clipping_range_m=(float(clipping[0]), float(clipping[1])),
            position_m=cast(tuple[float, float, float], tuple(float(value) for value in positions[0].tolist())),
            orientation_opengl_xyzw=cast(
                tuple[float, float, float, float],
                tuple(float(value) for value in orientations[0].tolist()),
            ),
        )

    def publish_debug(self, batch: DebugBatch) -> NativeDebugReport:
        for resource in batch.mesh_resources:
            cached = self._debug_mesh_resources.get(resource.resource_id)
            if cached is not None and cached.content_sha256 != resource.content_sha256:
                raise RuntimeError(f"debug mesh resource changed under stable ID: {resource.resource_id}")
            self._debug_mesh_resources[resource.resource_id] = resource
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
        portable_primitives = tuple(
            primitive for primitive in batch.primitives if primitive.kind is not DebugPrimitiveKind.MESH_INSTANCE
        )
        mesh_primitives = tuple(
            primitive for primitive in batch.primitives if primitive.kind is DebugPrimitiveKind.MESH_INSTANCE
        )
        if portable_primitives:
            self._native_debug_overlay().upsert(portable_primitives)
        for primitive in mesh_primitives:
            self._upsert_debug_mesh(primitive)
        self._flush_debug_render()
        active_keys = set(self._debug_mesh_paths)
        if self._debug_overlay is not None:
            active_keys.update(self._debug_overlay.keys)
        return len(batch.primitives), 0, len(active_keys)

    def _flush_debug_render(self) -> None:
        self._invalidate_render()
        if self._config.render:
            assert self._sim is not None
            self._sync_all_mounted_cameras()
            self._sim.render()
            self._mark_rendered()

    def _remove_debug_keys(self, keys: tuple[tuple[str, str, str], ...]) -> None:
        changed = False
        stage = self._m.sim_utils.get_current_stage()
        for key in keys:
            self._debug_expirations.pop(key, None)
            self._debug_lifetimes.pop(key, None)
            path = self._debug_mesh_paths.pop(key, None)
            self._debug_mesh_resource_ids.pop(key, None)
            self._debug_mesh_signatures.pop(key, None)
            if path is not None:
                stage.RemovePrim(path)
                changed = True
        if self._debug_overlay is not None and self._debug_overlay.remove(keys):
            changed = True
        if changed:
            self._flush_debug_render()

    def clear_debug(self, layer: str | None, group: str | None, primitive_id: str | None) -> int:
        active_keys = set(self._debug_mesh_paths)
        if self._debug_overlay is not None:
            active_keys.update(self._debug_overlay.keys)
        keys = tuple(
            key
            for key in active_keys
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
            physics_activation = getattr(self, "_physics_activation", None)
            if physics_activation is not None:
                physics_activation.update(self._step_index)
            for substep_index in range(self._spec.physics.substeps):
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
                native_step_index = self._step_index * self._spec.physics.substeps + substep_index + 1
                render = _render_step_enabled(
                    self._config,
                    native_step_index,
                    self._render_interval_steps,
                )
                if render:
                    self._sync_all_mounted_cameras()
                self._sim.step(render=render)
                self._invalidate_render()
                if render:
                    self._mark_rendered()
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
        self._selected_link_views.clear()
        self._initial_usd_articulation.clear()
        self._initial_usd_articulation_gains.clear()
        self._rigids.clear()
        self._usd_rigids.clear()
        self._usd_rigid_views.clear()
        self._initial_usd_rigid.clear()
        self._usd_rigid_wrenches.clear()
        self._kinematic_rigids.clear()
        self._static_scene_roots.clear()
        self._composite_scene_roots.clear()
        self._composite_scene_modes.clear()
        self._embedded_joint_paths.clear()
        self._composite_rigid_states.clear()
        self._composite_articulation_states.clear()
        self._physics_activation = None
        self._physics_activation_live_state = False
        self._mounted_cameras.clear()
        self._usd_tensor_view = None
        self._contacts.clear()
        self._deformables.clear()
        self._fluids.clear()
        self._cameras.clear()
        self._rgb_host_staging.clear()
        shared_host = self._rgb_shared_host
        self._rgb_shared_host = None
        self._rgb_shared_host_size = 0
        if shared_host is not None:
            result = self._m.torch.cuda.cudart().cudaHostUnregister(shared_host.data_ptr())
            if result != 0:
                raise RuntimeError(f"CUDA could not unregister the RGB shared host allocation: {result}")
        self._runtime_attachments.clear()
        self._debug_expirations.clear()
        self._debug_lifetimes.clear()
        if self._debug_mesh_paths:
            stage = self._m.sim_utils.get_current_stage()
            stage.RemovePrim("/World/UniRoboSimDebug")
        self._debug_mesh_resources.clear()
        self._debug_mesh_paths.clear()
        self._debug_mesh_resource_ids.clear()
        self._debug_mesh_signatures.clear()
        if self._debug_overlay is not None:
            self._debug_overlay.close()
            self._debug_overlay = None
        if self._debug_draw_interface is not None:
            self._m.debug_draw.release_debug_draw_interface(self._debug_draw_interface)
            self._debug_draw_interface = None
        self._initial_articulation.clear()
        self._initial_articulation_gains.clear()
        self._articulation_control_modes.clear()
        self._initial_rigid.clear()
        self._rigid_wrenches.clear()
        self._initial_entity_prim_poses.clear()
        self._entity_prim_path_cache.clear()
        self._entity_prim_physical_root_cache.clear()
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

    def physics_diagnostics(self) -> NativePhysicsDiagnostics:
        """Read the effective timestep from the live Isaac Lab context."""

        assert self._sim is not None
        native_dt = float(self._sim.get_physics_dt())
        substeps = self._spec.physics.substeps
        world_dt = native_dt * substeps
        if (
            not math.isfinite(native_dt)
            or native_dt <= 0.0
            or type(substeps) is not int
            or substeps <= 0
            or not math.isfinite(world_dt)
            or world_dt <= 0.0
        ):
            raise RuntimeError("Isaac Lab reported invalid effective physics timing")
        return NativePhysicsDiagnostics(
            native_step_dt_seconds=native_dt,
            substeps=substeps,
            world_step_dt_seconds=world_dt,
            source="Isaac Lab SimulationContext.get_physics_dt() after native world build",
        )
