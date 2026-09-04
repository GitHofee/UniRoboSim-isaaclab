"""Demand-only native planning-scene admission for Isaac Lab.

The ordinary native World never imports this module.  A planning-demanded World
uses the composed USD stage as structural authority, Isaac Lab tensor state as
dynamic authority, and PhysX cooking as mesh authority.  Unsupported or
ambiguous facts fail admission instead of being approximated.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from unirobosim import (
    PLANNING_SYSTEM_ENTITY_ID,
    PLANNING_SYSTEM_ENTITY_PATH,
    EntityKind,
    EntityPath,
    PlanningArticulationState,
    PlanningAttachment,
    PlanningEntityDescriptor,
    PlanningEntityKind,
    PlanningEntityState,
    PlanningFrameDescriptor,
    PlanningFrameKind,
    PlanningFrameSourceKind,
    PlanningFrameState,
    PlanningGeometryContentProfile,
    PlanningGeometryDescriptor,
    PlanningGeometryDType,
    PlanningGeometryLocalPose,
    PlanningGeometryMotionClass,
    PlanningGeometryPurpose,
    PlanningGeometryRepresentation,
    PlanningGeometryResourceLayout,
    PlanningGeometryTransform,
    PlanningHalfspaceGeometry,
    PlanningJointDescriptor,
    PlanningJointType,
    PlanningLinkDescriptor,
    PlanningLinkState,
    PlanningPose,
    PlanningPrimitiveGeometry,
    PlanningTwist,
    parse_planning_frame_declarations,
)

from ._collision_mesh import single_exact_convex_mesh
from ._planning_cache import PlanningMeshCache
from ._planning_cache import cache_key as planning_cache_key
from .native import IsaacLabNativeWorld, _native_name
from .native_protocols import (
    NativePlanningCatalog,
    NativePlanningError,
    NativePlanningResource,
    NativePlanningState,
)

_WORLD_FRAME_ID = "frame.world"
_SYSTEM_FRAME_ID = "frame.system.simulator_effective"
_SYSTEM_GROUND_GEOMETRY_ID = "geometry.system.ground"
_DEFAULT_COLLISION_GROUP = 1
_DEFAULT_COLLISION_MASK = 2**32 - 1
_MAX_GEOMETRY_RESOURCE_BYTES = 64 * 1024 * 1024
_MAX_FILTERED_COLLISION_BODIES = 128
_MAX_FILTER_CLIQUE_SEARCH_STATES = 1_000_000
_COLLISION_FILTER_PROFILE = "usd-effective-owner-pairs-to-bilateral-bitmask-v1"
_PROVENANCE_PROFILE = "isaaclab-3.0-complete-collision-forest-v5"
_MESH_CANONICALIZATION = "float32-le-vertices-then-uint32-le-triangles-v1"
_COMPOSITE_ENTITY_POSE = "__container_entity_pose__"
_SUPPORTED_COLLISION_SCHEMAS = frozenset(
    {
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
        "PhysxCollisionAPI",
        "PhysxContactReportAPI",
        # Isaac Sim 6 materializes these effective cooking schemas on the
        # runtime stage from PhysicsMeshCollisionAPI:approximation.  The
        # approximation token remains the authoritative representation below;
        # accepting only this closed set keeps unknown collision behavior
        # fail-closed without rejecting the simulator's effective stage.
        "PhysxConvexDecompositionCollisionAPI",
        "PhysxConvexHullCollisionAPI",
        "PhysxSDFMeshCollisionAPI",
        "PhysxSphereFillCollisionAPI",
        "PhysxTriangleMeshCollisionAPI",
        "PhysxTriangleMeshSimplificationCollisionAPI",
    }
)
_IDENTITY_POSE = PlanningPose(_WORLD_FRAME_ID, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
_ZERO_TWIST = PlanningTwist(_WORLD_FRAME_ID)


@dataclass(frozen=True, slots=True)
class _JointBinding:
    descriptor: PlanningJointDescriptor
    parent_name: str
    child_name: str
    local_pose: PlanningGeometryLocalPose
    movable_name: str | None


@dataclass(frozen=True, slots=True)
class _FrameBinding:
    frame_id: str
    entity_path: EntityPath | None
    source: str
    source_name: str | None = None
    local_pose: PlanningGeometryLocalPose | None = None


@dataclass(frozen=True, slots=True)
class _GeometryBinding:
    descriptor: PlanningGeometryDescriptor
    entity_path: EntityPath | None
    owner_link_name: str | None


@dataclass(frozen=True, slots=True)
class _TensorBodyBinding:
    view: Any
    row_by_environment_and_name: tuple[dict[str, int], ...]


@dataclass(frozen=True, slots=True)
class _EntityBinding:
    path: EntityPath
    entity_id: str
    root_link_name: str
    root_link_id: str
    link_id_by_name: dict[str, str]
    joint_bindings: tuple[_JointBinding, ...]
    entity_prim_paths: tuple[str, ...]
    entity_prim_link_name: str | None = None
    state_source: str = "asset"
    tensor_bodies: _TensorBodyBinding | None = None
    static_poses: tuple[dict[str, PlanningPose], ...] = ()


@dataclass(frozen=True, slots=True)
class _MeshInput:
    vertices: tuple[tuple[float, float, float], ...]
    face_counts: tuple[int, ...]
    face_indices: tuple[int, ...]
    triangles: tuple[tuple[int, int, int], ...]
    hole_faces: tuple[int, ...]
    subdivision: str
    orientation: str
    source_sha256: str


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}.{digest}"


def _local_asset(uri: str | None) -> Path | None:
    if uri is None:
        return None
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri)
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mesh_source_sha256(
    vertices: tuple[tuple[float, float, float], ...],
    face_counts: tuple[int, ...],
    face_indices: tuple[int, ...],
    hole_faces: tuple[int, ...],
    subdivision: str,
    orientation: str,
) -> str:
    """Hash canonical numeric mesh input without materializing a huge JSON document."""

    digest = hashlib.sha256(b"unirobosim-planning-mesh-input-v1\0")

    def update_numeric(typecode: str, values: Any) -> None:
        packed = array(typecode, values)
        if sys.byteorder != "little":
            packed.byteswap()
        digest.update(len(packed).to_bytes(8, "little"))
        digest.update(memoryview(packed).cast("B"))

    update_numeric("f", (component for vertex in vertices for component in vertex))
    update_numeric("I", face_counts)
    update_numeric("I", face_indices)
    update_numeric("I", hole_faces)
    for text in (subdivision, orientation):
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _float_tuple(value: Any, size: int) -> tuple[float, ...]:
    result = tuple(float(value[index]) for index in range(size))
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise NativePlanningError("native_failure")
    return tuple(0.0 if item == 0.0 else item for item in result)


def _xyzw(value: Any) -> tuple[float, float, float, float]:
    if hasattr(value, "GetImaginary") and hasattr(value, "GetReal"):
        imaginary = value.GetImaginary()
        result = (float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(value.GetReal()))
    else:
        result = _float_tuple(value, 4)  # type: ignore[assignment]
    norm = math.sqrt(sum(component * component for component in result))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise NativePlanningError("native_failure")
    normalized = tuple(component / norm for component in result)
    if normalized[3] < 0.0 or (
        normalized[3] == 0.0 and next((item for item in normalized[:3] if item != 0.0), 0.0) < 0.0
    ):
        normalized = tuple(-item for item in normalized)
    return tuple(0.0 if item == 0.0 else item for item in normalized)  # type: ignore[return-value]


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _rotate(
    vector: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
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


def _compose_pose(parent: PlanningPose, local: PlanningGeometryLocalPose) -> PlanningPose:
    offset = _rotate(local.position_m, parent.orientation_xyzw)
    return PlanningPose(
        parent.frame_id,
        tuple(parent.position_m[index] + offset[index] for index in range(3)),  # type: ignore[arg-type]
        _quat_multiply(parent.orientation_xyzw, local.orientation_xyzw),
    )


def _relative_pose(parent: PlanningPose, child: PlanningPose, parent_frame_id: str) -> PlanningPose:
    inverse_orientation = (
        -parent.orientation_xyzw[0],
        -parent.orientation_xyzw[1],
        -parent.orientation_xyzw[2],
        parent.orientation_xyzw[3],
    )
    world_delta = tuple(
        child.position_m[index] - parent.position_m[index] for index in range(3)
    )
    return PlanningPose(
        parent_frame_id,
        _rotate(world_delta, inverse_orientation),  # type: ignore[arg-type]
        _quat_multiply(inverse_orientation, child.orientation_xyzw),
    )


def _compose_scaled_local_pose(
    parent: PlanningGeometryLocalPose,
    scale: tuple[float, float, float],
    offset: tuple[float, float, float],
) -> PlanningGeometryLocalPose:
    scaled_offset = (scale[0] * offset[0], scale[1] * offset[1], scale[2] * offset[2])
    rotated = _rotate(scaled_offset, parent.orientation_xyzw)
    return PlanningGeometryLocalPose(
        tuple(parent.position_m[index] + rotated[index] for index in range(3)),  # type: ignore[arg-type]
        parent.orientation_xyzw,
    )


def _cylinder_pose_scale(
    pose: PlanningGeometryLocalPose,
    scale: tuple[float, float, float],
    axis: str,
) -> tuple[PlanningGeometryLocalPose, tuple[float, float, float]]:
    half_sqrt_two = math.sqrt(0.5)
    if axis == "Z":
        axis_orientation = (0.0, 0.0, 0.0, 1.0)
        cylinder_scale = scale
    elif axis == "X":
        axis_orientation = (0.0, half_sqrt_two, 0.0, half_sqrt_two)
        cylinder_scale = (scale[2], scale[1], scale[0])
    elif axis == "Y":
        axis_orientation = (-half_sqrt_two, 0.0, 0.0, half_sqrt_two)
        cylinder_scale = (scale[0], scale[2], scale[1])
    else:
        raise NativePlanningError("collision_geometry_unsupported")
    return (
        PlanningGeometryLocalPose(
            pose.position_m,
            _quat_multiply(pose.orientation_xyzw, axis_orientation),
        ),
        cylinder_scale,
    )


def _path_is_at_or_under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _canonical_body_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise NativePlanningError("collision_filter_unsupported")
    return (left, right) if left < right else (right, left)


def _maximum_filtered_clique(
    body_paths: tuple[str, ...],
    filtered_pairs: frozenset[tuple[str, str]],
) -> tuple[str, ...]:
    """Return the largest lexicographically first clique with bounded search."""

    best: tuple[str, ...] = ()
    search_states = 0

    def visit(clique: tuple[str, ...], candidates: tuple[str, ...]) -> None:
        nonlocal best, search_states
        search_states += 1
        if search_states > _MAX_FILTER_CLIQUE_SEARCH_STATES:
            raise NativePlanningError("collision_filter_unsupported")
        if len(clique) > len(best) or (len(clique) == len(best) and clique < best):
            best = clique
        remaining = candidates
        while remaining:
            if len(clique) + len(remaining) < len(best):
                return
            vertex = remaining[0]
            tail = remaining[1:]
            visit(
                (*clique, vertex),
                tuple(item for item in tail if _canonical_body_pair(vertex, item) in filtered_pairs),
            )
            remaining = tail

    visit((), body_paths)
    if not best:
        raise NativePlanningError("collision_filter_unsupported")
    return best


def _exact_filtered_pair_encoding(
    body_paths: tuple[str, ...],
    filtered_pairs: frozenset[tuple[str, str]],
    *,
    first_class_bit: int,
    self_filtered_bodies: frozenset[str] = frozenset(),
) -> tuple[dict[str, tuple[int, int]], int, tuple[tuple[str, ...], ...]]:
    """Encode one articulation's filtered-body graph exactly into group/mask bits."""

    bodies = tuple(sorted(set(body_paths)))
    if len(bodies) > _MAX_FILTERED_COLLISION_BODIES or not 1 <= first_class_bit <= 31:
        raise NativePlanningError("collision_filter_unsupported")
    body_set = frozenset(bodies)
    if not self_filtered_bodies.issubset(body_set) or any(
        left not in body_set or right not in body_set or left >= right for left, right in filtered_pairs
    ):
        raise NativePlanningError("collision_filter_unsupported")
    participants = tuple(sorted({item for pair in filtered_pairs for item in pair} | set(self_filtered_bodies)))
    if not participants:
        return {}, first_class_bit, ()

    classes: list[tuple[str, ...]] = []
    remaining = participants
    while remaining:
        clique = _maximum_filtered_clique(remaining, filtered_pairs)
        classes.append(clique)
        claimed = frozenset(clique)
        remaining = tuple(item for item in remaining if item not in claimed)
    classes.sort(key=lambda item: item[0])

    def allowed(left: str, right: str) -> bool:
        if left == right:
            return left not in participants
        return _canonical_body_pair(left, right) not in filtered_pairs

    while True:
        class_by_body = {body: index for index, members in enumerate(classes) for body in members}
        allowed_classes = {
            body: frozenset(class_by_body[other] for other in participants if allowed(body, other))
            for body in participants
        }
        mismatch: tuple[str, str] | None = None
        for left_index, left in enumerate(participants):
            for right in participants[left_index + 1 :]:
                predicted = (
                    class_by_body[right] in allowed_classes[left] and class_by_body[left] in allowed_classes[right]
                )
                if predicted != allowed(left, right):
                    mismatch = (left, right)
                    break
            if mismatch is not None:
                break
        if mismatch is None:
            break
        left, right = mismatch
        right_class = class_by_body[right]
        members = classes[right_class]
        filtered_members = tuple(item for item in members if not allowed(left, item))
        allowed_members = tuple(item for item in members if allowed(left, item))
        if not filtered_members or not allowed_members:
            raise NativePlanningError("collision_filter_unsupported")
        classes[right_class : right_class + 1] = [filtered_members, allowed_members]
        classes.sort(key=lambda item: item[0])

    if first_class_bit + len(classes) > 32:
        raise NativePlanningError("collision_filter_unsupported")
    class_by_body = {body: index for index, members in enumerate(classes) for body in members}
    group_by_class = tuple(1 << (first_class_bit + index) for index in range(len(classes)))
    encoding: dict[str, tuple[int, int]] = {}
    for body in participants:
        mask = _DEFAULT_COLLISION_MASK
        for class_index, members in enumerate(classes):
            if not any(allowed(body, other) for other in members):
                mask &= ~group_by_class[class_index]
        encoding[body] = (group_by_class[class_by_body[body]], mask)

    def encoded(body: str) -> tuple[int, int]:
        return encoding.get(body, (_DEFAULT_COLLISION_GROUP, _DEFAULT_COLLISION_MASK))

    for left_index, left in enumerate(bodies):
        left_group, left_mask = encoded(left)
        if bool(left_group & left_mask) != allowed(left, left):
            raise NativePlanningError("collision_filter_unsupported")
        for right in bodies[left_index + 1 :]:
            right_group, right_mask = encoded(right)
            predicted = bool(left_group & right_mask) and bool(right_group & left_mask)
            if predicted != allowed(left, right):
                raise NativePlanningError("collision_filter_unsupported")
        if not (left_group & _DEFAULT_COLLISION_MASK) or not (_DEFAULT_COLLISION_GROUP & left_mask):
            raise NativePlanningError("collision_filter_unsupported")
    return encoding, first_class_bit + len(classes), tuple(classes)


def _triangulate_faces(
    counts: tuple[int, ...],
    indices: tuple[int, ...],
    *,
    vertex_count: int,
    orientation: str,
    hole_faces: frozenset[int] = frozenset(),
) -> tuple[tuple[int, int, int], ...]:
    if not counts or sum(counts) != len(indices) or any(count < 0 for count in counts):
        raise NativePlanningError("collision_cooking_failed")
    if any(face < 0 or face >= len(counts) for face in hole_faces):
        raise NativePlanningError("collision_cooking_failed")
    if any(index < 0 or index >= vertex_count for index in indices):
        raise NativePlanningError("collision_cooking_failed")
    if orientation not in {"rightHanded", "leftHanded"}:
        raise NativePlanningError("collision_cooking_failed")
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for face_index, count in enumerate(counts):
        face = indices[cursor : cursor + count]
        cursor += count
        # USD permits authored degenerate faces and PhysX ignores them while
        # cooking.  Mirror that effective collision topology instead of
        # rejecting an otherwise valid mesh because one face has 0--2
        # vertices.  A mesh with no valid triangle still fails below.
        if face_index in hole_faces or count < 3:
            continue
        for offset in range(1, count - 1):
            triangle = (face[0], face[offset], face[offset + 1])
            triangles.append(triangle if orientation == "rightHanded" else (triangle[0], triangle[2], triangle[1]))
    if not triangles:
        raise NativePlanningError("collision_cooking_failed")
    return tuple(triangles)


def _reflect_mesh_input(
    mesh_input: _MeshInput,
    reflection: tuple[int, int, int],
) -> _MeshInput:
    if reflection == (1, 1, 1):
        return mesh_input
    if reflection not in {(-1, 1, 1), (1, -1, 1), (1, 1, -1)}:
        raise NativePlanningError("collision_geometry_unsupported")
    vertices = tuple(tuple(vertex[axis] * reflection[axis] for axis in range(3)) for vertex in mesh_input.vertices)
    return _MeshInput(
        vertices,  # type: ignore[arg-type]
        mesh_input.face_counts,
        mesh_input.face_indices,
        mesh_input.triangles,
        mesh_input.hole_faces,
        mesh_input.subdivision,
        mesh_input.orientation,
        _json_sha256(
            {
                "mesh_source_sha256": mesh_input.source_sha256,
                "baked_reflection": reflection,
            }
        ),
    )


def _bake_mesh_linear_transform(
    mesh_input: _MeshInput,
    linear: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> _MeshInput:
    vertices = tuple(
        tuple(sum(vertex[axis] * linear[axis][component] for axis in range(3)) for component in range(3))
        for vertex in mesh_input.vertices
    )
    if any(not math.isfinite(component) for vertex in vertices for component in vertex):
        raise NativePlanningError("collision_geometry_unsupported")
    return _MeshInput(
        vertices,  # type: ignore[arg-type]
        mesh_input.face_counts,
        mesh_input.face_indices,
        mesh_input.triangles,
        mesh_input.hole_faces,
        mesh_input.subdivision,
        mesh_input.orientation,
        _json_sha256(
            {
                "mesh_source_sha256": mesh_input.source_sha256,
                "baked_linear_transform": linear,
            }
        ),
    )


def _matrix_origin_basis(
    modules: Any,
    matrix: Any,
) -> tuple[
    tuple[float, float, float],
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
]:
    if any(abs(float(matrix[row][3])) > 1.0e-10 for row in range(3)) or not math.isclose(
        float(matrix[3][3]),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise NativePlanningError("collision_geometry_unsupported")
    origin = _float_tuple(matrix.Transform(modules.Gf.Vec3d(0.0, 0.0, 0.0)), 3)
    basis = tuple(
        tuple(
            value - origin[index]
            for index, value in enumerate(
                _float_tuple(
                    matrix.Transform(
                        modules.Gf.Vec3d(
                            1.0 if axis == 0 else 0.0,
                            1.0 if axis == 1 else 0.0,
                            1.0 if axis == 2 else 0.0,
                        )
                    ),
                    3,
                )
            )
        )
        for axis in range(3)
    )
    return origin, basis  # type: ignore[return-value]


def _matrix_pose_scale_reflection(
    modules: Any,
    matrix: Any,
) -> tuple[PlanningGeometryLocalPose, tuple[float, float, float], tuple[int, int, int]]:
    origin, basis = _matrix_origin_basis(modules, matrix)
    scale = (
        math.sqrt(sum(component * component for component in basis[0])),
        math.sqrt(sum(component * component for component in basis[1])),
        math.sqrt(sum(component * component for component in basis[2])),
    )
    if any(not math.isfinite(item) or item <= 0.0 for item in scale):
        raise NativePlanningError("collision_geometry_unsupported")
    tolerance = 1.0e-7 * max(1.0, *scale)
    if any(
        abs(sum(basis[left][axis] * basis[right][axis] for axis in range(3))) > tolerance
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise NativePlanningError("collision_geometry_unsupported")
    normalized_basis = tuple(tuple(component / scale[axis] for component in basis[axis]) for axis in range(3))
    determinant = (
        normalized_basis[0][0]
        * (normalized_basis[1][1] * normalized_basis[2][2] - normalized_basis[1][2] * normalized_basis[2][1])
        - normalized_basis[0][1]
        * (normalized_basis[1][0] * normalized_basis[2][2] - normalized_basis[1][2] * normalized_basis[2][0])
        + normalized_basis[0][2]
        * (normalized_basis[1][0] * normalized_basis[2][1] - normalized_basis[1][1] * normalized_basis[2][0])
    )
    reflection = (1, 1, 1)
    if math.isclose(determinant, -1.0, rel_tol=0.0, abs_tol=1.0e-6):
        # The public geometry contract requires a proper quaternion and
        # positive scale.  Canonically move one reflection into the geometry
        # payload; callers that cannot bake geometry reject it below.
        reflection = (-1, 1, 1)
        normalized_basis = (
            tuple(-component for component in normalized_basis[0]),
            normalized_basis[1],
            normalized_basis[2],
        )
    elif not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise NativePlanningError("collision_geometry_unsupported")
    rotation_matrix = modules.Gf.Matrix3d(*(component for row in normalized_basis for component in row))
    orientation = _xyzw(rotation_matrix.ExtractRotation().GetQuat())
    for axis, unit in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
        rotated = _rotate(unit, orientation)
        if any(abs(rotated[index] - normalized_basis[axis][index]) > 1.0e-6 for index in range(3)):
            raise NativePlanningError("collision_geometry_unsupported")
    return PlanningGeometryLocalPose(origin, orientation), scale, reflection


def _collision_pose_scale_bake(
    modules: Any,
    matrix: Any,
    *,
    mesh_capable: bool,
) -> tuple[
    PlanningGeometryLocalPose,
    tuple[float, float, float],
    tuple[int, int, int],
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None,
]:
    try:
        pose, scale, reflection = _matrix_pose_scale_reflection(modules, matrix)
        return pose, scale, reflection, None
    except NativePlanningError:
        if not mesh_capable:
            raise
    origin, basis = _matrix_origin_basis(modules, matrix)
    determinant = (
        basis[0][0] * (basis[1][1] * basis[2][2] - basis[1][2] * basis[2][1])
        - basis[0][1] * (basis[1][0] * basis[2][2] - basis[1][2] * basis[2][0])
        + basis[0][2] * (basis[1][0] * basis[2][1] - basis[1][1] * basis[2][0])
    )
    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-12:
        raise NativePlanningError("collision_geometry_unsupported")
    return (
        PlanningGeometryLocalPose(origin, (0.0, 0.0, 0.0, 1.0)),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        basis,
    )


def _effective_collision_relative_transform(cache: Any, prim: Any, owner_body: Any) -> Any:
    """Keep ancestor scale while expressing collision geometry in the rigid pose frame.

    ``ComputeRelativeTransform(prim, owner_body)`` cancels every transform shared by
    the collision Prim and its rigid-body owner.  Isaac Lab authors an EntitySpec
    scale on such a shared asset root, so using that relative transform silently
    drops the physical entity scale from planning geometry.  Public rigid/link
    states expose position and orientation, not scale; therefore collision geometry
    must instead be relative to the owner's pose-only world transform.
    """

    _relative, resets = cache.ComputeRelativeTransform(prim, owner_body)
    if resets:
        raise NativePlanningError("collision_geometry_unsupported")
    collision_world = cache.GetLocalToWorldTransform(prim)
    owner_world = cache.GetLocalToWorldTransform(owner_body)
    owner_pose_world = owner_world.RemoveScaleShear()
    return collision_world * owner_pose_world.GetInverse()


def _matrix_local_pose(modules: Any, matrix: Any) -> tuple[PlanningGeometryLocalPose, tuple[float, float, float]]:
    pose, scale, reflection = _matrix_pose_scale_reflection(modules, matrix)
    if reflection != (1, 1, 1):
        # Frames and entity poses have no geometry payload in which a mirror can
        # be represented.  Only collision geometry uses the baking path.
        raise NativePlanningError("collision_geometry_unsupported")
    return pose, scale


def _relationship_target(relationship: Any) -> str | None:
    targets = tuple(str(item) for item in relationship.GetTargets())
    if len(targets) > 1:
        raise NativePlanningError("topology_unsupported")
    return targets[0] if targets else None


def _nearest_body(path: str | None, bodies: dict[str, Any]) -> str | None:
    if path is None:
        return None
    current = path
    while current:
        if current in bodies:
            return current
        if current == "/":
            break
        current = current.rsplit("/", 1)[0] or "/"
    return None


def _motion_class(modules: Any, body_prim: Any) -> PlanningGeometryMotionClass:
    rigid = modules.UsdPhysics.RigidBodyAPI(body_prim)
    kinematic = rigid.GetKinematicEnabledAttr().Get()
    enabled = rigid.GetRigidBodyEnabledAttr().Get()
    if enabled is False:
        return PlanningGeometryMotionClass.STATIC
    if kinematic is True:
        return PlanningGeometryMotionClass.KINEMATIC
    return PlanningGeometryMotionClass.DYNAMIC


class _PlanningAdmission:
    def __init__(self, world: IsaacLabNativePlanningWorld) -> None:
        self._world = world
        self._m = world._m
        self._stage = self._m.sim_utils.get_current_stage()
        self._resources: dict[str, NativePlanningResource] = {}
        self._entities: dict[EntityPath, _EntityBinding] = {}
        self._frames: dict[str, _FrameBinding] = {}
        self._geometries: dict[str, _GeometryBinding] = {}
        self._source_sha_by_entity: dict[EntityPath, str] = {}
        self._mesh_inputs: dict[tuple[str, str], _MeshInput] = {}
        self._convex_cache: dict[tuple[str, str], tuple[tuple[bytes, int, int], ...]] = {}
        self._triangle_cache: dict[str, tuple[bytes, int, int]] = {}
        self._accounted_bodies: set[str] = set()
        self._accounted_colliders: set[str] = set()
        self._accounted_constraints: set[str] = set()
        self._accounted_filtered_pair_sources: set[str] = set()
        self._next_collision_filter_bit = 1
        self._walk_cache: dict[str, tuple[Any, ...]] = {}
        self._xform_cache = self._m.UsdGeom.XformCache()
        self._persistent_mesh_cache = PlanningMeshCache()
        try:
            self._catalog = self._build_catalog()
        finally:
            self._persistent_mesh_cache.close()

    @property
    def catalog(self) -> NativePlanningCatalog:
        return self._catalog

    def resource(self, geometry_id: str) -> NativePlanningResource:
        try:
            return self._resources[geometry_id]
        except KeyError:
            raise NativePlanningError("resource_missing") from None

    def _walk(self, root: Any) -> tuple[Any, ...]:
        key = str(root.GetPath())
        cached = self._walk_cache.get(key)
        if cached is None:
            cached = tuple(self._m.Usd.PrimRange(root, self._m.Usd.TraverseInstanceProxies()))
            self._walk_cache[key] = cached
        return cached

    def _entity_root(self, spec: Any, environment_index: int) -> Any:
        root = self._stage.GetPrimAtPath(f"/World/env_{environment_index}/{_native_name(spec.path)}")
        if not root.IsValid():
            raise NativePlanningError("catalog_invalid")
        return root

    def _source_sha256(self, spec: Any) -> str:
        cached = self._source_sha_by_entity.get(spec.path)
        if cached is not None:
            return cached
        if spec.asset_uri is None:
            digest = hashlib.sha256(b"procedural").hexdigest()
        else:
            path = _local_asset(spec.asset_uri)
            if path is None or not path.is_file():
                raise NativePlanningError("catalog_invalid")
            digest = _file_sha256(path)
        self._source_sha_by_entity[spec.path] = digest
        return digest

    def _articulation_collision_filter_projection(
        self,
        root: Any,
        bodies: dict[str, Any],
        body_name_by_path: dict[str, str],
        collision_owner_paths: tuple[str, ...],
    ) -> tuple[dict[str, tuple[int, int]], str, str, int, int, bool]:
        """Project one articulation's effective body filters without approximation."""

        articulation_roots = tuple(
            prim for prim in self._walk(root) if prim.HasAPI(self._m.UsdPhysics.ArticulationRootAPI)
        )
        if len(articulation_roots) != 1:
            raise NativePlanningError("collision_filter_unsupported")
        physx_articulation = self._m.PhysxSchema.PhysxArticulationAPI(articulation_roots[0])
        enabled_self_collisions = physx_articulation.GetEnabledSelfCollisionsAttr().Get()
        if type(enabled_self_collisions) is not bool:
            raise NativePlanningError("collision_filter_unsupported")

        authored_pairs: set[tuple[str, str]] = set()
        for prim in self._walk(root):
            filtered = self._m.UsdPhysics.FilteredPairsAPI(prim)
            if not filtered:
                continue
            targets = tuple(filtered.GetFilteredPairsRel().GetTargets())
            if not targets:
                continue
            source_path = str(prim.GetPath())
            if source_path not in bodies:
                raise NativePlanningError("collision_filter_unsupported")
            for target in targets:
                target_prim = self._stage.GetPrimAtPath(target)
                target_path = str(target)
                if (
                    not target_prim
                    or not target_prim.IsValid()
                    or target_path not in bodies
                    or not target_prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                ):
                    raise NativePlanningError("collision_filter_unsupported")
                authored_pairs.add(_canonical_body_pair(source_path, target_path))
            self._accounted_filtered_pair_sources.add(source_path)

        owner_paths = tuple(sorted(set(collision_owner_paths)))
        owner_set = frozenset(owner_paths)
        if enabled_self_collisions:
            effective_pairs = frozenset(
                pair for pair in authored_pairs if pair[0] in owner_set and pair[1] in owner_set
            )
        else:
            effective_pairs = frozenset(
                (left, right) for left_index, left in enumerate(owner_paths) for right in owner_paths[left_index + 1 :]
            )
        collider_count_by_owner = {path: collision_owner_paths.count(path) for path in owner_paths}
        self_filtered_bodies = frozenset(path for path, count in collider_count_by_owner.items() if count > 1)
        encoding, next_bit, _classes = _exact_filtered_pair_encoding(
            owner_paths,
            effective_pairs,
            first_class_bit=self._next_collision_filter_bit,
            self_filtered_bodies=self_filtered_bodies,
        )
        self._next_collision_filter_bit = next_bit

        named_pairs = tuple(
            sorted(
                tuple(sorted((body_name_by_path[left], body_name_by_path[right]))) for left, right in effective_pairs
            )
        )
        self_owner_names = tuple(sorted(body_name_by_path[path] for path in self_filtered_bodies))
        effective_graph = tuple(sorted((*named_pairs, *((name, name) for name in self_owner_names))))
        return (
            encoding,
            _json_sha256(effective_graph),
            _json_sha256(named_pairs),
            len(named_pairs),
            len(self_owner_names),
            enabled_self_collisions,
        )

    def _build_catalog(self) -> NativePlanningCatalog:
        frames: list[PlanningFrameDescriptor] = [
            PlanningFrameDescriptor(_WORLD_FRAME_ID, PlanningFrameKind.WORLD, None, None, None)
        ]
        entities: list[PlanningEntityDescriptor] = []
        links: list[PlanningLinkDescriptor] = []
        joints: list[PlanningJointDescriptor] = []
        geometries: list[PlanningGeometryDescriptor] = []

        system_geometry = self._ground_geometry()
        if system_geometry is not None:
            frames.append(
                PlanningFrameDescriptor(
                    _SYSTEM_FRAME_ID,
                    PlanningFrameKind.ENTITY,
                    _WORLD_FRAME_ID,
                    PLANNING_SYSTEM_ENTITY_ID,
                    None,
                )
            )
            entities.append(
                PlanningEntityDescriptor(
                    PLANNING_SYSTEM_ENTITY_ID,
                    PLANNING_SYSTEM_ENTITY_PATH,
                    PlanningEntityKind.OTHER,
                    True,
                    _SYSTEM_FRAME_ID,
                    (),
                    (_SYSTEM_FRAME_ID,),
                    (_SYSTEM_GROUND_GEOMETRY_ID,),
                )
            )
            geometries.append(system_geometry)
            self._geometries[system_geometry.geometry_id] = _GeometryBinding(system_geometry, None, None)
        self._frames[_WORLD_FRAME_ID] = _FrameBinding(_WORLD_FRAME_ID, None, "world")
        if system_geometry is not None:
            self._frames[_SYSTEM_FRAME_ID] = _FrameBinding(_SYSTEM_FRAME_ID, None, "system")

        ordered_specs = tuple(
            sorted(
                self._world._spec.entities,
                key=lambda item: (
                    0 if item.kind is EntityKind.COMPOSITE_SCENE else 1 if item.embedded_binding is not None else 2,
                    item.path.value,
                ),
            )
        )
        for spec in ordered_specs:
            if spec.kind is EntityKind.CAMERA_SENSOR:
                self._verify_nonphysical_entity(spec)
                continue
            if spec.kind is EntityKind.PARTICLE_FLUID:
                # Particle fluids are simulated entities, but they are not a
                # planner-consumable collision resource.  Excluding them from
                # the planning catalog lets one world expose rigid/articulated
                # geometry and fluid state at the same time without pretending
                # thousands of transient particles are immutable obstacles.
                continue
            if spec.kind is EntityKind.STATIC_SCENE:
                # Static scenes deliberately retain the old fail-closed contract.
                raise NativePlanningError("topology_unsupported")
            if spec.kind not in {EntityKind.ARTICULATION, EntityKind.RIGID_BODY}:
                if spec.kind is not EntityKind.COMPOSITE_SCENE:
                    raise NativePlanningError("soft_matter_unsupported")
            if spec.kind is EntityKind.COMPOSITE_SCENE:
                result = self._composite_catalog(spec)
            elif spec.embedded_binding is not None:
                result = self._embedded_entity_catalog(spec)
            else:
                result = self._entity_catalog(spec)
            entity, entity_links, entity_joints, entity_frames, entity_geometries, binding = result
            entities.append(entity)
            links.extend(entity_links)
            joints.extend(entity_joints)
            frames.extend(entity_frames)
            geometries.extend(entity_geometries)
            self._entities[spec.path] = binding

        self._verify_complete_native_world()
        return NativePlanningCatalog(
            tuple(sorted(entities, key=lambda item: item.entity_id)),
            tuple(sorted(links, key=lambda item: item.link_id)),
            tuple(sorted(joints, key=lambda item: item.joint_id)),
            tuple(sorted(frames, key=lambda item: item.frame_id)),
            tuple(sorted(geometries, key=lambda item: item.geometry_id)),
        )

    def _verify_nonphysical_entity(self, spec: Any) -> None:
        for environment_index in range(self._world._spec.environments.count):
            root = self._entity_root(spec, environment_index)
            for prim in self._walk(root):
                filtered = self._m.UsdPhysics.FilteredPairsAPI(prim)
                if (
                    prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                    or prim.HasAPI(self._m.UsdPhysics.CollisionAPI)
                    or prim.IsA(self._m.UsdPhysics.Joint)
                    or prim.IsA(self._m.UsdPhysics.CollisionGroup)
                    or (filtered and tuple(filtered.GetFilteredPairsRel().GetTargets()))
                    or "Attachment" in prim.GetTypeName()
                    or "Constraint" in prim.GetTypeName()
                ):
                    raise NativePlanningError("catalog_invalid")

    def _ground_geometry(self) -> PlanningGeometryDescriptor | None:
        root = self._stage.GetPrimAtPath("/World/unirobosimGround")
        if not root or not root.IsValid():
            if any(item.kind is EntityKind.COMPOSITE_SCENE for item in self._world._spec.entities):
                return None
            raise NativePlanningError("collision_geometry_unsupported")
        candidates = tuple(
            prim
            for prim in self._walk(root)
            if prim.HasAPI(self._m.UsdPhysics.CollisionAPI)
            and self._m.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        )
        if len(candidates) != 1 or candidates[0].GetTypeName() != "Plane":
            raise NativePlanningError("collision_geometry_unsupported")
        plane = candidates[0]
        axis = str(self._m.UsdGeom.Plane(plane).GetAxisAttr().Get() or "Z").upper()
        if axis != "Z":
            raise NativePlanningError("collision_geometry_unsupported")
        self._validate_collision_common(plane)
        pose, scale = _matrix_local_pose(
            self._m,
            self._xform_cache.GetLocalToWorldTransform(plane),
        )
        if max(scale) - min(scale) > 1.0e-10:
            raise NativePlanningError("collision_geometry_unsupported")
        self._accounted_colliders.add(str(plane.GetPath()))
        provenance = _json_sha256(
            {
                "adapter": _PROVENANCE_PROFILE,
                "native_path": str(plane.GetPath()),
                "representation": "halfspace",
                "axis": axis,
                "pose": [pose.position_m, pose.orientation_xyzw],
                "native_uniform_scale": scale[0],
            }
        )
        return PlanningGeometryDescriptor(
            _SYSTEM_GROUND_GEOMETRY_ID,
            PLANNING_SYSTEM_ENTITY_ID,
            None,
            _SYSTEM_FRAME_ID,
            PlanningGeometryPurpose.COLLISION,
            PlanningGeometryRepresentation.HALFSPACE,
            pose,
            (1.0, 1.0, 1.0),
            PlanningGeometryMotionClass.STATIC,
            _DEFAULT_COLLISION_GROUP,
            _DEFAULT_COLLISION_MASK,
            provenance,
            inline=PlanningHalfspaceGeometry(),
        )

    def _entity_catalog(
        self,
        spec: Any,
    ) -> tuple[
        PlanningEntityDescriptor,
        tuple[PlanningLinkDescriptor, ...],
        tuple[PlanningJointDescriptor, ...],
        tuple[PlanningFrameDescriptor, ...],
        tuple[PlanningGeometryDescriptor, ...],
        _EntityBinding,
    ]:
        root = self._entity_root(spec, 0)
        entity_id = _stable_id("entity", spec.path.value)
        entity_frame_id = _stable_id("frame.entity", spec.path.value)
        body_prims = tuple(prim for prim in self._walk(root) if prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI))
        if not body_prims:
            raise NativePlanningError("topology_unsupported")
        locked_motion = spec.metadata.get("planning_motion_class")
        if locked_motion is not None:
            expected_motion = PlanningGeometryMotionClass(locked_motion)
            if any(_motion_class(self._m, prim) is not expected_motion for prim in body_prims):
                raise NativePlanningError("topology_unsupported")
        self._accounted_bodies.update(str(prim.GetPath()) for prim in body_prims)
        bodies = {str(prim.GetPath()): prim for prim in body_prims}
        body_name_by_path = {path: prim.GetName() for path, prim in bodies.items()}
        if len(set(body_name_by_path.values())) != len(body_name_by_path):
            raise NativePlanningError("topology_unsupported")
        link_id_by_name = {name: _stable_id("link", spec.path.value, path) for path, name in body_name_by_path.items()}
        link_frame_by_name = {
            name: _stable_id("frame.link", spec.path.value, path) for path, name in body_name_by_path.items()
        }

        joint_models: list[tuple[Any, str, str, str | None, PlanningJointType]] = []
        child_names: set[str] = set()
        for prim in self._walk(root):
            joint_type = self._joint_type(prim)
            if joint_type is None:
                continue
            joint = self._m.UsdPhysics.Joint(prim)
            body0_target = _relationship_target(joint.GetBody0Rel())
            body1_target = _relationship_target(joint.GetBody1Rel())
            body0 = _nearest_body(body0_target, bodies)
            body1 = _nearest_body(body1_target, bodies)
            path = str(prim.GetPath())
            if (body0_target is not None and body0 is None) or (body1_target is not None and body1 is None):
                raise NativePlanningError("constraint_unsupported")
            if body0 is None and body1 is not None:
                if joint_type is not PlanningJointType.FIXED:
                    raise NativePlanningError("constraint_unsupported")
                self._accounted_constraints.add(path)
                continue
            if body0 is None or body1 is None or body0 == body1:
                raise NativePlanningError("constraint_unsupported")
            parent_name = body_name_by_path[body0]
            child_name = body_name_by_path[body1]
            if child_name in child_names:
                raise NativePlanningError("topology_unsupported")
            child_names.add(child_name)
            movable = None if joint_type is PlanningJointType.FIXED else prim.GetName()
            joint_models.append((prim, parent_name, child_name, movable, joint_type))
            self._accounted_constraints.add(path)
        roots = tuple(sorted(set(link_id_by_name) - child_names))
        if len(roots) != 1:
            raise NativePlanningError("topology_unsupported")
        root_name = roots[0]

        ordered_joint_models = self._order_joints(root_name, joint_models)
        if len(ordered_joint_models) != len(body_prims) - 1:
            raise NativePlanningError("topology_unsupported")
        movable_names = tuple(model[3] for model in ordered_joint_models if model[3] is not None)
        if spec.kind is EntityKind.ARTICULATION:
            asset_names = tuple(self._world._articulations[spec.path].joint_names)
            if set(movable_names) != set(asset_names) or not set(spec.joint_names).issubset(asset_names):
                raise NativePlanningError("topology_unsupported")
        elif movable_names:
            raise NativePlanningError("topology_unsupported")

        collision_models: list[tuple[Any, str]] = []
        for prim in self._walk(root):
            if not prim.HasAPI(self._m.UsdPhysics.CollisionAPI):
                continue
            collision = self._m.UsdPhysics.CollisionAPI(prim)
            if collision.GetCollisionEnabledAttr().Get() is False:
                continue
            owner_path = _nearest_body(str(prim.GetPath()), bodies)
            if owner_path is None:
                raise NativePlanningError("catalog_invalid")
            collision_models.append((prim, owner_path))
        if spec.kind is EntityKind.ARTICULATION:
            (
                filter_encoding,
                filter_graph_sha256,
                filter_distinct_pairs_sha256,
                filter_pair_count,
                filter_self_owner_count,
                enabled_self_collisions,
            ) = self._articulation_collision_filter_projection(
                root,
                bodies,
                body_name_by_path,
                tuple(owner_path for _prim, owner_path in collision_models),
            )
        else:
            filter_encoding = {}
            filter_graph_sha256 = None
            filter_distinct_pairs_sha256 = None
            filter_pair_count = 0
            filter_self_owner_count = 0
            enabled_self_collisions = True

        geometry_by_link: dict[str, list[PlanningGeometryDescriptor]] = {name: [] for name in link_id_by_name}
        for prim, owner_path in collision_models:
            owner_name = body_name_by_path[owner_path]
            collision_group, collision_mask = filter_encoding.get(
                owner_path,
                (_DEFAULT_COLLISION_GROUP, _DEFAULT_COLLISION_MASK),
            )
            collision_geometries = self._collision_geometry(
                spec,
                entity_id,
                prim,
                bodies[owner_path],
                link_id_by_name[owner_name],
                link_frame_by_name[owner_name],
                collision_group=collision_group,
                collision_mask=collision_mask,
                collision_filter_graph_sha256=filter_graph_sha256,
                collision_filter_distinct_pairs_sha256=filter_distinct_pairs_sha256,
                collision_filter_pair_count=filter_pair_count,
                collision_filter_self_owner_count=filter_self_owner_count,
                enabled_self_collisions=enabled_self_collisions,
            )
            geometry_by_link[owner_name].extend(collision_geometries)
            for geometry in collision_geometries:
                self._geometries[geometry.geometry_id] = _GeometryBinding(geometry, spec.path, owner_name)
            self._accounted_colliders.add(str(prim.GetPath()))

        frame_descriptors: list[PlanningFrameDescriptor] = [
            PlanningFrameDescriptor(
                entity_frame_id,
                PlanningFrameKind.ENTITY,
                _WORLD_FRAME_ID,
                entity_id,
                None,
            )
        ]
        joint_bindings: list[_JointBinding] = []
        parent_name_by_child = {model[2]: model[1] for model in ordered_joint_models}
        joint_frame_by_child: dict[str, str] = {}
        joint_descriptors: list[PlanningJointDescriptor] = []
        for prim, parent_name, child_name, movable, joint_type in ordered_joint_models:
            joint_id = _stable_id("joint", spec.path.value, str(prim.GetPath()))
            joint_frame_id = _stable_id("frame.joint", spec.path.value, str(prim.GetPath()))
            joint_frame_by_child[child_name] = joint_frame_id
            local_pose = self._joint_local_pose(prim)
            descriptor = self._joint_descriptor(
                prim,
                joint_id,
                entity_id,
                link_id_by_name[parent_name],
                link_id_by_name[child_name],
                joint_frame_id,
                joint_type,
            )
            joint_descriptors.append(descriptor)
            joint_bindings.append(_JointBinding(descriptor, parent_name, child_name, local_pose, movable))
            frame_descriptors.append(
                PlanningFrameDescriptor(
                    joint_frame_id,
                    PlanningFrameKind.JOINT,
                    link_frame_by_name[parent_name],
                    entity_id,
                    link_id_by_name[child_name],
                )
            )
            self._frames[joint_frame_id] = _FrameBinding(
                joint_frame_id,
                spec.path,
                "joint",
                prim.GetName(),
                local_pose,
            )

        link_descriptors: list[PlanningLinkDescriptor] = []
        for name in sorted(link_id_by_name):
            parent_of_link = parent_name_by_child.get(name)
            frame_id = link_frame_by_name[name]
            frame_parent = entity_frame_id if parent_of_link is None else joint_frame_by_child[name]
            geometry_ids = tuple(sorted(item.geometry_id for item in geometry_by_link[name]))
            link_descriptors.append(
                PlanningLinkDescriptor(
                    link_id_by_name[name],
                    entity_id,
                    name,
                    frame_id,
                    None if parent_of_link is None else link_id_by_name[parent_of_link],
                    geometry_ids,
                )
            )
            frame_descriptors.append(
                PlanningFrameDescriptor(
                    frame_id,
                    PlanningFrameKind.LINK,
                    frame_parent,
                    entity_id,
                    link_id_by_name[name],
                )
            )
            self._frames[frame_id] = _FrameBinding(frame_id, spec.path, "link", name)
        self._frames[entity_frame_id] = _FrameBinding(entity_frame_id, spec.path, "entity_prim", None)

        self._declared_frames(
            spec,
            root,
            entity_id,
            entity_frame_id,
            link_id_by_name,
            link_frame_by_name,
            joint_bindings,
            frame_descriptors,
            root_name,
        )
        entity_geometries = tuple(
            sorted(
                (geometry for values in geometry_by_link.values() for geometry in values),
                key=lambda item: item.geometry_id,
            )
        )
        entity_frames = tuple(sorted(frame_descriptors, key=lambda item: item.frame_id))
        entity_links = tuple(sorted(link_descriptors, key=lambda item: item.link_id))
        kind = self._entity_kind(spec)
        entity = PlanningEntityDescriptor(
            entity_id,
            spec.path.value,
            kind,
            True,
            entity_frame_id,
            tuple(item.link_id for item in entity_links),
            tuple(item.frame_id for item in entity_frames),
            tuple(item.geometry_id for item in entity_geometries),
            tuple(item.joint_id for item in joint_descriptors),
        )
        return (
            entity,
            entity_links,
            tuple(joint_descriptors),
            entity_frames,
            entity_geometries,
            _EntityBinding(
                spec.path,
                entity_id,
                root_name,
                link_id_by_name[root_name],
                link_id_by_name,
                tuple(joint_bindings),
                tuple(
                    str(self._entity_root(spec, environment).GetPath())
                    for environment in range(self._world._spec.environments.count)
                ),
                root_name if str(root.GetPath()) in bodies else None,
            ),
        )

    def _tensor_body_binding(
        self,
        rows: tuple[dict[str, str], ...],
    ) -> _TensorBodyBinding:
        requested = tuple(path for row in rows for path in row.values())
        if not requested or len(set(requested)) != len(requested):
            raise NativePlanningError("topology_unsupported")
        view = self._world._usd_simulation_view().create_rigid_body_view(list(requested))
        actual = tuple(str(path) for path in view.prim_paths)
        if view.count != len(requested) or len(set(actual)) != len(actual) or set(actual) != set(requested):
            raise NativePlanningError("topology_unsupported")
        index_by_path = {path: index for index, path in enumerate(actual)}
        return _TensorBodyBinding(
            view,
            tuple({name: index_by_path[path] for name, path in row.items()} for row in rows),
        )

    def _composite_catalog(
        self,
        spec: Any,
    ) -> tuple[
        PlanningEntityDescriptor,
        tuple[PlanningLinkDescriptor, ...],
        tuple[PlanningJointDescriptor, ...],
        tuple[PlanningFrameDescriptor, ...],
        tuple[PlanningGeometryDescriptor, ...],
        _EntityBinding,
    ]:
        root = self._entity_root(spec, 0)
        root_path = str(root.GetPath())
        entity_id = _stable_id("entity", spec.path.value)
        entity_frame_id = _stable_id("frame.entity", spec.path.value)
        embedded = tuple(
            item
            for item in self._world._spec.entities
            if item.embedded_binding is not None and item.embedded_binding.container_path == spec.path
        )
        moving_relative_roots = tuple(
            sorted(item.embedded_binding.root_body_prim_path for item in embedded if item.embedded_binding is not None)
        )
        moving_absolute_roots = tuple(f"{root_path}/{relative}" for relative in moving_relative_roots)
        all_body_prims = tuple(prim for prim in self._walk(root) if prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI))
        bodies = {str(prim.GetPath()): prim for prim in all_body_prims}
        retained_bodies = {
            path: prim
            for path, prim in bodies.items()
            if not any(_path_is_at_or_under(path, moving) for moving in moving_absolute_roots)
        }
        body_name_by_path = {path: path.removeprefix(f"{root_path}/") for path in retained_bodies}
        if any(not name or name == path for path, name in body_name_by_path.items()):
            raise NativePlanningError("topology_unsupported")
        link_names = tuple(sorted(body_name_by_path.values()))
        link_id_by_name = {name: _stable_id("link", spec.path.value, name) for name in link_names}
        link_frame_by_name = {name: _stable_id("frame.link", spec.path.value, name) for name in link_names}
        geometry_by_link: dict[str | None, list[PlanningGeometryDescriptor]] = {
            None: [],
            **{name: [] for name in link_names},
        }
        for prim in self._walk(root):
            if prim.IsA(self._m.UsdPhysics.Joint):
                self._accounted_constraints.add(str(prim.GetPath()))
            if not prim.HasAPI(self._m.UsdPhysics.CollisionAPI):
                continue
            collision = self._m.UsdPhysics.CollisionAPI(prim)
            if collision.GetCollisionEnabledAttr().Get() is False:
                continue
            path = str(prim.GetPath())
            if any(_path_is_at_or_under(path, moving) for moving in moving_absolute_roots):
                continue
            owner_path = _nearest_body(path, bodies)
            if owner_path is None:
                owner_name = None
                owner_prim = root
                motion = PlanningGeometryMotionClass.STATIC
            else:
                if owner_path not in retained_bodies:
                    raise NativePlanningError("catalog_invalid")
                owner_name = body_name_by_path[owner_path]
                owner_prim = retained_bodies[owner_path]
                motion = _motion_class(self._m, owner_prim)
            collision_geometries = self._collision_geometry(
                spec,
                entity_id,
                prim,
                owner_prim,
                None if owner_name is None else link_id_by_name[owner_name],
                entity_frame_id if owner_name is None else link_frame_by_name[owner_name],
                motion=motion,
            )
            geometry_by_link[owner_name].extend(collision_geometries)
            for geometry in collision_geometries:
                self._geometries[geometry.geometry_id] = _GeometryBinding(geometry, spec.path, owner_name)
            self._accounted_colliders.add(path)
        self._accounted_bodies.update(retained_bodies)

        frame_descriptors = [
            PlanningFrameDescriptor(entity_frame_id, PlanningFrameKind.ENTITY, _WORLD_FRAME_ID, entity_id, None)
        ]
        link_descriptors: list[PlanningLinkDescriptor] = []
        for name in link_names:
            geometry_ids = tuple(sorted(item.geometry_id for item in geometry_by_link[name]))
            link_descriptors.append(
                PlanningLinkDescriptor(
                    link_id_by_name[name],
                    entity_id,
                    name,
                    link_frame_by_name[name],
                    None,
                    geometry_ids,
                )
            )
            frame_descriptors.append(
                PlanningFrameDescriptor(
                    link_frame_by_name[name],
                    PlanningFrameKind.LINK,
                    entity_frame_id,
                    entity_id,
                    link_id_by_name[name],
                )
            )
            self._frames[link_frame_by_name[name]] = _FrameBinding(link_frame_by_name[name], spec.path, "link", name)
        self._frames[entity_frame_id] = _FrameBinding(
            entity_frame_id,
            spec.path,
            "static_entity",
            _COMPOSITE_ENTITY_POSE,
        )
        if parse_planning_frame_declarations(spec.metadata.get("planning_frame_declarations")) is not None:
            raise NativePlanningError("frame_ambiguous")

        reference_names = tuple(sorted(body_name_by_path.values()))
        tensor_rows: list[dict[str, str]] = []
        static_poses: list[dict[str, PlanningPose]] = []
        cache = self._xform_cache
        for environment_index in range(self._world._spec.environments.count):
            environment_root = self._entity_root(spec, environment_index)
            environment_root_path = str(environment_root.GetPath())
            row = {
                str(prim.GetPath()).removeprefix(f"{environment_root_path}/"): str(prim.GetPath())
                for prim in self._walk(environment_root)
                if prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                and not any(
                    _path_is_at_or_under(
                        str(prim.GetPath()),
                        f"{environment_root_path}/{relative}",
                    )
                    for relative in moving_relative_roots
                )
            }
            if tuple(sorted(row)) != reference_names:
                raise NativePlanningError("topology_unsupported")
            tensor_rows.append(row)
            root_local, root_scale = _matrix_local_pose(self._m, cache.GetLocalToWorldTransform(environment_root))
            # Composite scale is already baked into each collision geometry by
            # ``_effective_collision_relative_transform``.  Planning entity
            # state carries pose only, so retaining the same scale here would
            # double-apply it.  Static composite roots may therefore expose an
            # arbitrary positive XYZ scale while their world pose remains the
            # scale-free transform below.
            del root_scale
            origin = self._world._origins_cpu[environment_index]
            static_poses.append(
                {
                    _COMPOSITE_ENTITY_POSE: PlanningPose(
                        _WORLD_FRAME_ID,
                        tuple(root_local.position_m[axis] - origin[axis] for axis in range(3)),  # type: ignore[arg-type]
                        root_local.orientation_xyzw,
                    )
                }
            )
        tensor_binding = self._tensor_body_binding(tuple(tensor_rows)) if reference_names else None
        entity_geometries = tuple(
            sorted(
                (geometry for values in geometry_by_link.values() for geometry in values),
                key=lambda item: item.geometry_id,
            )
        )
        entity_frames = tuple(sorted(frame_descriptors, key=lambda item: item.frame_id))
        entity_links = tuple(sorted(link_descriptors, key=lambda item: item.link_id))
        entity = PlanningEntityDescriptor(
            entity_id,
            spec.path.value,
            PlanningEntityKind(spec.metadata.get("planning_entity_kind", "other")),
            True,
            entity_frame_id,
            tuple(item.link_id for item in entity_links),
            tuple(item.frame_id for item in entity_frames),
            tuple(item.geometry_id for item in entity_geometries),
        )
        return (
            entity,
            entity_links,
            (),
            entity_frames,
            entity_geometries,
            _EntityBinding(
                spec.path,
                entity_id,
                _COMPOSITE_ENTITY_POSE,
                "",
                link_id_by_name,
                (),
                tuple(
                    str(self._entity_root(spec, environment).GetPath())
                    for environment in range(self._world._spec.environments.count)
                ),
                None,
                "tensor",
                tensor_binding,
                tuple(static_poses),
            ),
        )

    def _embedded_entity_catalog(
        self,
        spec: Any,
    ) -> tuple[
        PlanningEntityDescriptor,
        tuple[PlanningLinkDescriptor, ...],
        tuple[PlanningJointDescriptor, ...],
        tuple[PlanningFrameDescriptor, ...],
        tuple[PlanningGeometryDescriptor, ...],
        _EntityBinding,
    ]:
        binding = spec.embedded_binding
        if binding is None:
            raise NativePlanningError("catalog_invalid")
        container_spec = next(
            (
                item
                for item in self._world._spec.entities
                if item.path == binding.container_path and item.kind is EntityKind.COMPOSITE_SCENE
            ),
            None,
        )
        if container_spec is None:
            raise NativePlanningError("catalog_invalid")
        root = self._entity_root(container_spec, 0)
        root_path = str(root.GetPath())
        entity_id = _stable_id("entity", spec.path.value)
        entity_frame_id = _stable_id("frame.entity", spec.path.value)
        logical_by_path = {f"{root_path}/{item.relative_prim_path}": item.logical_name for item in binding.link_prims}
        bodies: dict[str, Any] = {}
        for path in logical_by_path:
            prim = self._stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI):
                raise NativePlanningError("topology_unsupported")
            bodies[path] = prim
        if len(bodies) != len(binding.link_prims):
            raise NativePlanningError("topology_unsupported")
        self._accounted_bodies.update(bodies)
        link_id_by_name = {
            item.logical_name: _stable_id("link", spec.path.value, item.relative_prim_path)
            for item in binding.link_prims
        }
        link_frame_by_name = {
            item.logical_name: _stable_id("frame.link", spec.path.value, item.relative_prim_path)
            for item in binding.link_prims
        }
        joint_models: list[tuple[Any, str, str, str | None, PlanningJointType]] = []
        for item in binding.joint_prims:
            path = f"{root_path}/{item.relative_prim_path}"
            prim = self._stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise NativePlanningError("topology_unsupported")
            joint_type = self._joint_type(prim)
            if joint_type is None:
                raise NativePlanningError("constraint_unsupported")
            joint = self._m.UsdPhysics.Joint(prim)
            body0 = _nearest_body(_relationship_target(joint.GetBody0Rel()), bodies)
            body1 = _nearest_body(_relationship_target(joint.GetBody1Rel()), bodies)
            if body0 is None or body1 is None or body0 == body1:
                raise NativePlanningError("constraint_unsupported")
            joint_models.append((prim, logical_by_path[body0], logical_by_path[body1], item.logical_name, joint_type))
            self._accounted_constraints.add(path)
        child_names = {item[2] for item in joint_models}
        root_names = tuple(sorted(set(link_id_by_name) - child_names))
        if len(root_names) != 1:
            raise NativePlanningError("topology_unsupported")
        root_name = root_names[0]
        ordered_joint_models = self._order_joints(root_name, joint_models)
        if spec.kind is EntityKind.ARTICULATION:
            if len(ordered_joint_models) != len(link_id_by_name) - 1:
                raise NativePlanningError("topology_unsupported")
            if tuple(model[3] for model in ordered_joint_models) != tuple(spec.joint_names):
                raise NativePlanningError("topology_unsupported")
        elif joint_models:
            raise NativePlanningError("topology_unsupported")

        moving_root = f"{root_path}/{binding.root_body_prim_path}"
        geometry_by_link: dict[str, list[PlanningGeometryDescriptor]] = {name: [] for name in link_id_by_name}
        for prim in self._walk(self._stage.GetPrimAtPath(moving_root)):
            if not prim.HasAPI(self._m.UsdPhysics.CollisionAPI):
                continue
            collision = self._m.UsdPhysics.CollisionAPI(prim)
            if collision.GetCollisionEnabledAttr().Get() is False:
                continue
            owner_path = _nearest_body(str(prim.GetPath()), bodies)
            if owner_path is None:
                raise NativePlanningError("catalog_invalid")
            owner_name = logical_by_path[owner_path]
            collision_geometries = self._collision_geometry(
                spec,
                entity_id,
                prim,
                bodies[owner_path],
                link_id_by_name[owner_name],
                link_frame_by_name[owner_name],
                source_spec=container_spec,
            )
            geometry_by_link[owner_name].extend(collision_geometries)
            for geometry in collision_geometries:
                self._geometries[geometry.geometry_id] = _GeometryBinding(geometry, spec.path, owner_name)
            self._accounted_colliders.add(str(prim.GetPath()))

        frame_descriptors: list[PlanningFrameDescriptor] = [
            PlanningFrameDescriptor(entity_frame_id, PlanningFrameKind.ENTITY, _WORLD_FRAME_ID, entity_id, None)
        ]
        joint_bindings: list[_JointBinding] = []
        parent_name_by_child = {model[2]: model[1] for model in ordered_joint_models}
        joint_frame_by_child: dict[str, str] = {}
        joint_descriptors: list[PlanningJointDescriptor] = []
        for prim, parent_name, child_name, movable, joint_type in ordered_joint_models:
            joint_id = _stable_id("joint", spec.path.value, str(prim.GetPath()).removeprefix(root_path))
            joint_frame_id = _stable_id("frame.joint", spec.path.value, str(prim.GetPath()).removeprefix(root_path))
            joint_frame_by_child[child_name] = joint_frame_id
            local_pose = self._joint_local_pose(prim)
            descriptor = self._joint_descriptor(
                prim,
                joint_id,
                entity_id,
                link_id_by_name[parent_name],
                link_id_by_name[child_name],
                joint_frame_id,
                joint_type,
                authored_name=movable,
            )
            joint_descriptors.append(descriptor)
            joint_bindings.append(_JointBinding(descriptor, parent_name, child_name, local_pose, movable))
            frame_descriptors.append(
                PlanningFrameDescriptor(
                    joint_frame_id,
                    PlanningFrameKind.JOINT,
                    link_frame_by_name[parent_name],
                    entity_id,
                    link_id_by_name[child_name],
                )
            )
            self._frames[joint_frame_id] = _FrameBinding(joint_frame_id, spec.path, "joint_id", joint_id, local_pose)
        link_descriptors: list[PlanningLinkDescriptor] = []
        for name in sorted(link_id_by_name):
            parent_of_link = parent_name_by_child.get(name)
            frame_id = link_frame_by_name[name]
            link_descriptors.append(
                PlanningLinkDescriptor(
                    link_id_by_name[name],
                    entity_id,
                    name,
                    frame_id,
                    None if parent_of_link is None else link_id_by_name[parent_of_link],
                    tuple(sorted(item.geometry_id for item in geometry_by_link[name])),
                )
            )
            frame_descriptors.append(
                PlanningFrameDescriptor(
                    frame_id,
                    PlanningFrameKind.LINK,
                    entity_frame_id if parent_of_link is None else joint_frame_by_child[name],
                    entity_id,
                    link_id_by_name[name],
                )
            )
            self._frames[frame_id] = _FrameBinding(frame_id, spec.path, "link", name)
        self._frames[entity_frame_id] = _FrameBinding(entity_frame_id, spec.path, "entity_prim", None)
        if parse_planning_frame_declarations(spec.metadata.get("planning_frame_declarations")) is not None:
            raise NativePlanningError("frame_ambiguous")

        tensor_rows = tuple(
            {
                item.logical_name: (
                    f"{self._entity_root(container_spec, environment_index).GetPath()}/{item.relative_prim_path}"
                )
                for item in binding.link_prims
            }
            for environment_index in range(self._world._spec.environments.count)
        )
        tensor_binding = self._tensor_body_binding(tensor_rows)
        entity_geometries = tuple(
            sorted(
                (geometry for values in geometry_by_link.values() for geometry in values),
                key=lambda item: item.geometry_id,
            )
        )
        entity_frames = tuple(sorted(frame_descriptors, key=lambda item: item.frame_id))
        entity_links = tuple(sorted(link_descriptors, key=lambda item: item.link_id))
        entity = PlanningEntityDescriptor(
            entity_id,
            spec.path.value,
            self._entity_kind(spec),
            True,
            entity_frame_id,
            tuple(item.link_id for item in entity_links),
            tuple(item.frame_id for item in entity_frames),
            tuple(item.geometry_id for item in entity_geometries),
            tuple(item.joint_id for item in joint_descriptors),
        )
        return (
            entity,
            entity_links,
            tuple(joint_descriptors),
            entity_frames,
            entity_geometries,
            _EntityBinding(
                spec.path,
                entity_id,
                root_name,
                link_id_by_name[root_name],
                link_id_by_name,
                tuple(joint_bindings),
                tuple(
                    self._world._entity_prim_path(spec.path, environment)
                    for environment in range(self._world._spec.environments.count)
                ),
                (
                    root_name
                    if self._world._entity_prim_path(spec.path, 0) == moving_root
                    else None
                ),
                "usd_articulation" if spec.kind is EntityKind.ARTICULATION else "tensor",
                tensor_binding,
            ),
        )

    def _entity_kind(self, spec: Any) -> PlanningEntityKind:
        locked = spec.metadata.get("planning_entity_kind")
        if locked is not None:
            return PlanningEntityKind(locked)
        if spec.kind is EntityKind.ARTICULATION:
            return PlanningEntityKind.ARTICULATION
        return PlanningEntityKind.RIGID_OBJECT

    def _joint_type(self, prim: Any) -> PlanningJointType | None:
        if prim.IsA(self._m.UsdPhysics.FixedJoint):
            return PlanningJointType.FIXED
        if prim.IsA(self._m.UsdPhysics.RevoluteJoint):
            return PlanningJointType.REVOLUTE
        if prim.IsA(self._m.UsdPhysics.PrismaticJoint):
            return PlanningJointType.PRISMATIC
        if prim.IsA(self._m.UsdPhysics.Joint):
            raise NativePlanningError("constraint_unsupported")
        return None

    @staticmethod
    def _order_joints(
        root_name: str,
        models: list[tuple[Any, str, str, str | None, PlanningJointType]],
    ) -> tuple[tuple[Any, str, str, str | None, PlanningJointType], ...]:
        children: dict[str, list[tuple[Any, str, str, str | None, PlanningJointType]]] = {}
        for model in models:
            children.setdefault(model[1], []).append(model)
        ordered = []
        queue = [root_name]
        while queue:
            parent = queue.pop(0)
            for model in sorted(children.get(parent, ()), key=lambda item: str(item[0].GetPath())):
                ordered.append(model)
                queue.append(model[2])
        return tuple(ordered)

    def _joint_local_pose(self, prim: Any) -> PlanningGeometryLocalPose:
        joint = self._m.UsdPhysics.Joint(prim)
        position = joint.GetLocalPos0Attr().Get() or (0.0, 0.0, 0.0)
        orientation = joint.GetLocalRot0Attr().Get()
        return PlanningGeometryLocalPose(
            _float_tuple(position, 3),  # type: ignore[arg-type]
            (0.0, 0.0, 0.0, 1.0) if orientation is None else _xyzw(orientation),
        )

    def _joint_descriptor(
        self,
        prim: Any,
        joint_id: str,
        entity_id: str,
        parent_link_id: str,
        child_link_id: str,
        frame_id: str,
        joint_type: PlanningJointType,
        *,
        authored_name: str | None = None,
    ) -> PlanningJointDescriptor:
        public_name = prim.GetName() if authored_name is None else authored_name
        if joint_type is PlanningJointType.FIXED:
            return PlanningJointDescriptor(
                joint_id,
                entity_id,
                public_name,
                parent_link_id,
                child_link_id,
                joint_type,
                frame_id,
                (1.0, 0.0, 0.0),
                "rad",
            )
        schema = (
            self._m.UsdPhysics.RevoluteJoint(prim)
            if joint_type is PlanningJointType.REVOLUTE
            else self._m.UsdPhysics.PrismaticJoint(prim)
        )
        axis_token = str(schema.GetAxisAttr().Get() or "X").upper()
        axes = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
        if axis_token not in axes:
            raise NativePlanningError("topology_unsupported")
        lower_value = schema.GetLowerLimitAttr().Get()
        upper_value = schema.GetUpperLimitAttr().Get()
        lower = None if lower_value is None else float(lower_value)
        upper = None if upper_value is None else float(upper_value)
        if lower is not None and upper is not None and (not math.isfinite(lower) or not math.isfinite(upper)):
            lower = upper = None
        if joint_type is PlanningJointType.REVOLUTE and lower is not None and upper is not None:
            lower = math.radians(lower)
            upper = math.radians(upper)
        return PlanningJointDescriptor(
            joint_id,
            entity_id,
            public_name,
            parent_link_id,
            child_link_id,
            joint_type,
            frame_id,
            axes[axis_token],
            "rad" if joint_type is PlanningJointType.REVOLUTE else "m",
            lower,
            upper,
        )

    def _validate_collision_common(self, prim: Any) -> None:
        collision = self._m.UsdPhysics.CollisionAPI(prim)
        if collision.GetCollisionEnabledAttr().Get() is False:
            raise NativePlanningError("catalog_invalid")
        applied = set(prim.GetAppliedSchemas())
        unexpected = {
            schema
            for schema in applied
            if ("Collision" in schema or "FilteredPairs" in schema) and schema not in _SUPPORTED_COLLISION_SCHEMAS
        }
        if unexpected:
            raise NativePlanningError("collision_geometry_unsupported")
        filtered = self._m.UsdPhysics.FilteredPairsAPI(prim)
        if (
            filtered
            and tuple(filtered.GetFilteredPairsRel().GetTargets())
            and str(prim.GetPath()) not in self._accounted_filtered_pair_sources
        ):
            raise NativePlanningError("collision_filter_unsupported")
        physx = self._m.PhysxSchema.PhysxCollisionAPI(prim)
        if physx:
            for attribute in (physx.GetContactOffsetAttr(), physx.GetRestOffsetAttr()):
                if attribute.HasAuthoredValueOpinion():
                    value = attribute.Get()
                    if value is None or not math.isfinite(float(value)) or float(value) != 0.0:
                        raise NativePlanningError("collision_geometry_unsupported")

    def _collision_geometry(
        self,
        spec: Any,
        entity_id: str,
        prim: Any,
        owner_body: Any,
        owner_link_id: str | None,
        owner_frame_id: str,
        *,
        motion: PlanningGeometryMotionClass | None = None,
        source_spec: Any | None = None,
        collision_group: int = _DEFAULT_COLLISION_GROUP,
        collision_mask: int = _DEFAULT_COLLISION_MASK,
        collision_filter_graph_sha256: str | None = None,
        collision_filter_distinct_pairs_sha256: str | None = None,
        collision_filter_pair_count: int = 0,
        collision_filter_self_owner_count: int = 0,
        enabled_self_collisions: bool = True,
    ) -> tuple[PlanningGeometryDescriptor, ...]:
        self._validate_collision_common(prim)
        cache = self._xform_cache
        matrix = _effective_collision_relative_transform(cache, prim, owner_body)
        mesh_capable = not any(
            prim.IsA(schema) for schema in (self._m.UsdGeom.Cube, self._m.UsdGeom.Sphere, self._m.UsdGeom.Cylinder)
        )
        local_pose, scale, reflection, baked_linear = _collision_pose_scale_bake(
            self._m,
            matrix,
            mesh_capable=mesh_capable,
        )
        geometry_id = _stable_id("geometry", spec.path.value, str(prim.GetPath()))
        motion = _motion_class(self._m, owner_body) if motion is None else motion
        source_spec = spec if source_spec is None else source_spec
        source_sha = self._source_sha256(source_spec)
        common = {
            "adapter": _PROVENANCE_PROFILE,
            "source_sha256": source_sha,
            "native_path": str(prim.GetPath()).removeprefix(f"/World/env_0/{_native_name(source_spec.path)}"),
            "applied_schemas": sorted(prim.GetAppliedSchemas()),
            "local_pose": [local_pose.position_m, local_pose.orientation_xyzw],
            "scale": scale,
            "baked_reflection": reflection,
            "baked_linear_transform": baked_linear,
            "collision_filter_profile": _COLLISION_FILTER_PROFILE,
            "collision_filter_graph_sha256": collision_filter_graph_sha256,
            "collision_filter_distinct_owner_pairs_sha256": collision_filter_distinct_pairs_sha256,
            "collision_filter_distinct_owner_pair_count": collision_filter_pair_count,
            "collision_filter_self_owner_count": collision_filter_self_owner_count,
            "collision_group": collision_group,
            "collision_mask": collision_mask,
            "enabled_self_collisions": enabled_self_collisions,
        }
        if prim.IsA(self._m.UsdGeom.Cube):
            size = float(self._m.UsdGeom.Cube(prim).GetSizeAttr().Get())
            if not math.isfinite(size) or size <= 0.0:
                raise NativePlanningError("collision_geometry_unsupported")
            inline = PlanningPrimitiveGeometry(PlanningGeometryRepresentation.BOX, (size, size, size))
            return (
                PlanningGeometryDescriptor(
                    geometry_id,
                    entity_id,
                    owner_link_id,
                    owner_frame_id,
                    PlanningGeometryPurpose.COLLISION,
                    PlanningGeometryRepresentation.BOX,
                    local_pose,
                    scale,
                    motion,
                    collision_group,
                    collision_mask,
                    _json_sha256({**common, "representation": "box", "dimensions": inline.dimensions_m}),
                    inline=inline,
                ),
            )
        if prim.IsA(self._m.UsdGeom.Sphere):
            radius = float(self._m.UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            if not math.isfinite(radius) or radius <= 0.0:
                raise NativePlanningError("collision_geometry_unsupported")
            inline = PlanningPrimitiveGeometry(PlanningGeometryRepresentation.SPHERE, (radius,))
            return (
                PlanningGeometryDescriptor(
                    geometry_id,
                    entity_id,
                    owner_link_id,
                    owner_frame_id,
                    PlanningGeometryPurpose.COLLISION,
                    PlanningGeometryRepresentation.SPHERE,
                    local_pose,
                    scale,
                    motion,
                    collision_group,
                    collision_mask,
                    _json_sha256({**common, "representation": "sphere", "dimensions": inline.dimensions_m}),
                    inline=inline,
                ),
            )
        if prim.IsA(self._m.UsdGeom.Cylinder):
            cylinder = self._m.UsdGeom.Cylinder(prim)
            radius = float(cylinder.GetRadiusAttr().Get())
            height = float(cylinder.GetHeightAttr().Get())
            axis = str(cylinder.GetAxisAttr().Get() or "Z").upper()
            if not math.isfinite(radius) or radius <= 0.0 or not math.isfinite(height) or height <= 0.0:
                raise NativePlanningError("collision_geometry_unsupported")
            cylinder_pose, cylinder_scale = _cylinder_pose_scale(local_pose, scale, axis)
            inline = PlanningPrimitiveGeometry(PlanningGeometryRepresentation.CYLINDER, (radius, height))
            return (
                PlanningGeometryDescriptor(
                    geometry_id,
                    entity_id,
                    owner_link_id,
                    owner_frame_id,
                    PlanningGeometryPurpose.COLLISION,
                    PlanningGeometryRepresentation.CYLINDER,
                    cylinder_pose,
                    cylinder_scale,
                    motion,
                    collision_group,
                    collision_mask,
                    _json_sha256(
                        {
                            **common,
                            "representation": "cylinder",
                            "dimensions": inline.dimensions_m,
                            "axis": axis,
                            "canonical_axis": "Z",
                        }
                    ),
                    inline=inline,
                ),
            )

        mesh_api = self._m.UsdPhysics.MeshCollisionAPI(prim)
        approximation = str(mesh_api.GetApproximationAttr().Get()) if mesh_api else ""
        if approximation not in {
            "boundingCube",
            "convexHull",
            "convexDecomposition",
            "meshSimplification",
            "sdf",
            "none",
        }:
            raise NativePlanningError("collision_geometry_unsupported")
        resolved_key = planning_cache_key(
            _PROVENANCE_PROFILE,
            _MESH_CANONICALIZATION,
            str(self._world._spec.build_resource_manifest_sha256),
            json.dumps(common, sort_keys=True, separators=(",", ":")),
            approximation,
        )
        mesh_source_sha = planning_cache_key("effective-mesh-v1", resolved_key)
        resolved_components = (
            None
            if approximation == "boundingCube"
            else self._persistent_mesh_cache.get(planning_cache_key("resolved-components-v1", resolved_key))
        )
        mesh_input: _MeshInput | None = None
        if resolved_components is None:
            mesh_input = _reflect_mesh_input(self._mesh_input(prim), reflection)
            if baked_linear is not None:
                mesh_input = _bake_mesh_linear_transform(mesh_input, baked_linear)
        if approximation == "boundingCube":
            assert mesh_input is not None
            lower = tuple(min(vertex[axis] for vertex in mesh_input.vertices) for axis in range(3))
            upper = tuple(max(vertex[axis] for vertex in mesh_input.vertices) for axis in range(3))
            dimensions = (upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2])
            if any(not math.isfinite(value) or value <= 0.0 for value in dimensions):
                raise NativePlanningError("collision_geometry_unsupported")
            center = (
                (lower[0] + upper[0]) * 0.5,
                (lower[1] + upper[1]) * 0.5,
                (lower[2] + upper[2]) * 0.5,
            )
            box_pose = _compose_scaled_local_pose(local_pose, scale, center)
            inline = PlanningPrimitiveGeometry(PlanningGeometryRepresentation.BOX, dimensions)
            return (
                PlanningGeometryDescriptor(
                    geometry_id,
                    entity_id,
                    owner_link_id,
                    owner_frame_id,
                    PlanningGeometryPurpose.COLLISION,
                    PlanningGeometryRepresentation.BOX,
                    box_pose,
                    scale,
                    motion,
                    collision_group,
                    collision_mask,
                    _json_sha256(
                        {
                            **common,
                            "representation": "box",
                            "approximation": approximation,
                            "mesh_source_sha256": mesh_source_sha,
                            "bounds": [lower, upper],
                            "center": center,
                        }
                    ),
                    inline=inline,
                ),
            )

        if approximation in {"convexHull", "convexDecomposition"}:
            components = (
                resolved_components
                if resolved_components is not None
                else self._cook_convex_components(mesh_input, approximation)  # type: ignore[arg-type]
            )
            if approximation == "convexHull" and len(components) != 1:
                raise NativePlanningError("collision_cooking_failed")
            representation = PlanningGeometryRepresentation.CONVEX_MESH
        else:
            components = (
                resolved_components
                if resolved_components is not None
                else (self._canonical_triangle_mesh(mesh_input),)  # type: ignore[arg-type]
            )
            representation = PlanningGeometryRepresentation.TRIANGLE_MESH
        if resolved_components is None:
            self._persistent_mesh_cache.put(planning_cache_key("resolved-components-v1", resolved_key), components)

        descriptors: list[PlanningGeometryDescriptor] = []
        for component_index, (content, vertex_count, triangle_count) in enumerate(components):
            if len(content) > _MAX_GEOMETRY_RESOURCE_BYTES:
                raise NativePlanningError("collision_cooking_failed")
            digest = hashlib.sha256(content).hexdigest()
            component_geometry_id = (
                geometry_id
                if len(components) == 1
                else _stable_id("geometry", spec.path.value, str(prim.GetPath()), f"component:{component_index}")
            )
            layout = PlanningGeometryResourceLayout(
                representation,
                PlanningGeometryContentProfile.MESH_TRIANGLES_RAW_LE_V1,
                PlanningGeometryDType.FLOAT32,
                (vertex_count, 3),
                PlanningGeometryDType.UINT32,
                (triangle_count, 3),
            )
            provenance = _json_sha256(
                {
                    **common,
                    "representation": representation.value,
                    "approximation": approximation,
                    "mesh_source_sha256": mesh_source_sha,
                    "component_index": component_index,
                    "component_count": len(components),
                    "content_sha256": digest,
                    "canonicalization": _MESH_CANONICALIZATION,
                }
            )
            descriptor = PlanningGeometryDescriptor(
                component_geometry_id,
                entity_id,
                owner_link_id,
                owner_frame_id,
                PlanningGeometryPurpose.COLLISION,
                representation,
                local_pose,
                scale,
                motion,
                collision_group,
                collision_mask,
                provenance,
                resource_id=_stable_id("resource", representation.value, digest),
                sha256=digest,
                content_profile=PlanningGeometryContentProfile.MESH_TRIANGLES_RAW_LE_V1,
                resource_layout=layout,
            )
            self._resources[component_geometry_id] = NativePlanningResource(
                component_geometry_id,
                representation,
                content,
                digest,
            )
            descriptors.append(descriptor)
        return tuple(descriptors)

    def _mesh_input(self, carrier: Any) -> _MeshInput:
        mesh_prim = single_exact_convex_mesh(self._m, carrier, self._walk(carrier))
        mesh = self._m.UsdGeom.Mesh(mesh_prim)
        points = tuple(mesh.GetPointsAttr().Get() or ())
        counts = tuple(int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or ()))
        indices = tuple(int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or ()))
        if not points:
            raise NativePlanningError("collision_cooking_failed")
        matrix, resets = self._xform_cache.ComputeRelativeTransform(mesh_prim, carrier)
        if resets:
            raise NativePlanningError("collision_cooking_failed")
        transformed = tuple(matrix.Transform(point) for point in points)
        vertices = tuple(
            (float(value[0]), float(value[1]), float(value[2]))
            for value in (_float_tuple(point, 3) for point in transformed)
        )
        orientation = str(mesh.GetOrientationAttr().Get() or "rightHanded")
        hole_faces = tuple(int(value) for value in (mesh.GetHoleIndicesAttr().Get() or ()))
        triangles = _triangulate_faces(
            counts,
            indices,
            vertex_count=len(vertices),
            orientation=orientation,
            hole_faces=frozenset(hole_faces),
        )
        subdivision = str(mesh.GetSubdivisionSchemeAttr().Get() or "none")
        input_sha = _mesh_source_sha256(
            vertices,
            counts,
            indices,
            hole_faces,
            subdivision,
            orientation,
        )
        return _MeshInput(
            vertices,
            counts,
            indices,
            triangles,
            hole_faces,
            subdivision,
            orientation,
            input_sha,
        )

    def _canonical_triangle_mesh(self, mesh_input: _MeshInput) -> tuple[bytes, int, int]:
        cached = self._triangle_cache.get(mesh_input.source_sha256)
        if cached is not None:
            return cached
        persistent_key = planning_cache_key(
            _PROVENANCE_PROFILE, _MESH_CANONICALIZATION, "triangle", mesh_input.source_sha256
        )
        persistent = self._persistent_mesh_cache.get(persistent_key)
        if persistent is not None and len(persistent) == 1:
            self._triangle_cache[mesh_input.source_sha256] = persistent[0]
            return persistent[0]
        vertex_bytes = b"".join(struct.pack("<fff", *vertex) for vertex in mesh_input.vertices)
        index_bytes = b"".join(struct.pack("<III", *triangle) for triangle in mesh_input.triangles)
        result = (vertex_bytes + index_bytes, len(mesh_input.vertices), len(mesh_input.triangles))
        self._triangle_cache[mesh_input.source_sha256] = result
        self._persistent_mesh_cache.put(persistent_key, (result,))
        return result

    def _cook_convex_components(
        self,
        mesh_input: _MeshInput,
        approximation: str,
    ) -> tuple[tuple[bytes, int, int], ...]:
        cache_key = (approximation, mesh_input.source_sha256)
        cached = self._convex_cache.get(cache_key)
        if cached is not None:
            return cached

        persistent_key = planning_cache_key(
            _PROVENANCE_PROFILE,
            _MESH_CANONICALIZATION,
            "physx-convex-v1",
            approximation,
            mesh_input.source_sha256,
        )
        persistent = self._persistent_mesh_cache.get(persistent_key)
        if persistent is not None:
            self._convex_cache[cache_key] = persistent
            return persistent

        import omni.physx  # type: ignore[import-not-found]
        from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt  # type: ignore[import-not-found]

        temporary = Usd.Stage.CreateInMemory()
        clone = UsdGeom.Mesh.Define(temporary, "/collision")
        clone.GetPointsAttr().Set(Vt.Vec3fArray(mesh_input.vertices))
        clone.GetFaceVertexCountsAttr().Set(Vt.IntArray(mesh_input.face_counts))
        clone.GetFaceVertexIndicesAttr().Set(Vt.IntArray(mesh_input.face_indices))
        if mesh_input.hole_faces:
            clone.GetHoleIndicesAttr().Set(Vt.IntArray(mesh_input.hole_faces))
        clone.GetSubdivisionSchemeAttr().Set("none")
        clone.GetOrientationAttr().Set(mesh_input.orientation)
        UsdPhysics.CollisionAPI.Apply(clone.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(clone.GetPrim())
        mesh_collision.GetApproximationAttr().Set(approximation)
        cache = UsdUtils.StageCache.Get()
        stage_id_object = cache.Insert(temporary)
        result_values: list[object] = []
        convex_values: list[object] = []

        def callback(result: object, convexes: list[object]) -> None:
            result_values.append(result)
            convex_values.extend(convexes)

        try:
            omni.physx.get_physx_cooking_interface().request_convex_collision_representation(
                stage_id_object.ToLongInt(),
                PhysicsSchemaTools.sdfPathToInt("/collision"),
                False,
                callback,
            )
        finally:
            cache.Erase(stage_id_object)
        if len(result_values) != 1 or "RESULT_VALID" not in repr(result_values[0]) or not convex_values:
            raise NativePlanningError("collision_cooking_failed")
        components: list[tuple[bytes, int, int]] = []
        for cooked_value in convex_values:
            cooked: Any = cooked_value
            cooked_vertices = tuple(_float_tuple(value, 3) for value in cooked.vertices)
            cooked_indices = tuple(int(value) for value in cooked.indices)
            triangles: list[tuple[int, int, int]] = []
            for polygon in cooked.polygons:
                start = int(polygon.index_base)
                count = int(polygon.num_vertices)
                polygon_indices = cooked_indices[start : start + count]
                if count < 3 or len(polygon_indices) != count:
                    raise NativePlanningError("collision_cooking_failed")
                for offset in range(1, count - 1):
                    triangles.append((polygon_indices[0], polygon_indices[offset], polygon_indices[offset + 1]))
            if (
                not cooked_vertices
                or not triangles
                or any(index < 0 or index >= len(cooked_vertices) for triangle in triangles for index in triangle)
            ):
                raise NativePlanningError("collision_cooking_failed")
            vertex_bytes = b"".join(struct.pack("<fff", *vertex) for vertex in cooked_vertices)
            index_bytes = b"".join(struct.pack("<III", *triangle) for triangle in triangles)
            components.append((vertex_bytes + index_bytes, len(cooked_vertices), len(triangles)))
        result = tuple(sorted(components, key=lambda item: (hashlib.sha256(item[0]).digest(), item[0])))
        self._convex_cache[cache_key] = result
        self._persistent_mesh_cache.put(persistent_key, result)
        return result

    def _declared_frames(
        self,
        spec: Any,
        root: Any,
        entity_id: str,
        entity_frame_id: str,
        link_id_by_name: dict[str, str],
        link_frame_by_name: dict[str, str],
        joint_bindings: list[_JointBinding],
        descriptors: list[PlanningFrameDescriptor],
        root_link_name: str,
    ) -> None:
        declarations = parse_planning_frame_declarations(spec.metadata.get("planning_frame_declarations"))
        if declarations is None:
            return
        if declarations.component_sha256 != self._source_sha256(spec):
            raise NativePlanningError("frame_ambiguous")
        joint_by_name = {binding.descriptor.authored_name: binding for binding in joint_bindings}
        prims_by_name: dict[str, list[Any]] = {}
        for prim in self._walk(root):
            prims_by_name.setdefault(prim.GetName(), []).append(prim)
        for declaration in declarations.entries:
            frame_id = _stable_id("frame.named", spec.path.value, declaration.name)
            source = declaration.source
            if source.kind is PlanningFrameSourceKind.LINK:
                if source.name not in link_id_by_name:
                    raise NativePlanningError("frame_missing")
                if declaration.owner_link_name not in {None, source.name}:
                    raise NativePlanningError("frame_ambiguous")
                owner_name = source.name
                binding = _FrameBinding(frame_id, spec.path, "link", source.name)
            elif source.kind is PlanningFrameSourceKind.JOINT:
                joint = joint_by_name.get(source.name)
                if joint is None:
                    raise NativePlanningError("frame_missing")
                owner_name = joint.child_name
                if declaration.owner_link_name not in {None, owner_name}:
                    raise NativePlanningError("frame_ambiguous")
                binding = _FrameBinding(frame_id, spec.path, "joint", source.name, joint.local_pose)
            else:
                matches = prims_by_name.get(source.name, [])
                if len(matches) != 1:
                    raise NativePlanningError("frame_missing" if not matches else "frame_ambiguous")
                native_owner_name = declaration.owner_link_name
                if native_owner_name is None:
                    parent_frame = entity_frame_id
                    owner_link_id = None
                    owner_prim = next(
                        prim
                        for prim in self._walk(root)
                        if prim.GetName() == root_link_name and prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                    )
                else:
                    if native_owner_name not in link_id_by_name:
                        raise NativePlanningError("frame_missing")
                    parent_frame = link_frame_by_name[native_owner_name]
                    owner_link_id = link_id_by_name[native_owner_name]
                    owner_prim = next(
                        prim
                        for prim in self._walk(root)
                        if prim.GetName() == native_owner_name and prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                    )
                matrix, resets = self._xform_cache.ComputeRelativeTransform(matches[0], owner_prim)
                if resets:
                    raise NativePlanningError("frame_ambiguous")
                local_pose, scale = _matrix_local_pose(self._m, matrix)
                if scale != (1.0, 1.0, 1.0):
                    raise NativePlanningError("frame_ambiguous")
                descriptors.append(
                    PlanningFrameDescriptor(
                        frame_id,
                        PlanningFrameKind.NAMED,
                        parent_frame,
                        entity_id,
                        owner_link_id,
                        declaration.name,
                    )
                )
                self._frames[frame_id] = _FrameBinding(
                    frame_id,
                    spec.path,
                    "native",
                    native_owner_name,
                    local_pose,
                )
                continue
            descriptors.append(
                PlanningFrameDescriptor(
                    frame_id,
                    PlanningFrameKind.NAMED,
                    link_frame_by_name[owner_name],
                    entity_id,
                    link_id_by_name[owner_name],
                    declaration.name,
                )
            )
            self._frames[frame_id] = binding

    @staticmethod
    def _normalized_clone_text(value: object, root_path: str) -> str:
        environment_path = root_path.rsplit("/", 1)[0]
        return repr(value).replace(root_path, "$ENTITY").replace(environment_path, "$ENV")

    def _prim_signature(self, prim: Any, root_path: str) -> tuple[object, ...]:
        try:
            attributes = []
            for attribute in prim.GetAttributes():
                samples = tuple(float(value) for value in attribute.GetTimeSamples())
                values = (
                    tuple(self._normalized_clone_text(attribute.Get(sample), root_path) for sample in samples)
                    if samples
                    else (self._normalized_clone_text(attribute.Get(), root_path),)
                )
                attributes.append((attribute.GetName(), str(attribute.GetTypeName()), samples, values))
            relationships = tuple(
                sorted(
                    (
                        relationship.GetName(),
                        tuple(self._normalized_clone_text(target, root_path) for target in relationship.GetTargets()),
                    )
                    for relationship in prim.GetRelationships()
                )
            )
            prototype = str(prim.GetPrimInPrototype().GetPath()) if prim.IsInstanceProxy() else None
            return (
                prim.GetTypeName(),
                tuple(sorted(prim.GetAppliedSchemas())),
                prototype,
                tuple(sorted(attributes)),
                relationships,
            )
        except BaseException as error:
            try:
                error.__traceback__ = None
                error.__cause__ = None
                error.__context__ = None
            except BaseException:
                pass
            raise NativePlanningError("native_failure") from None

    @staticmethod
    def _relative_prim_map(root: Any, prims: tuple[Any, ...]) -> dict[str, Any]:
        root_path = str(root.GetPath())
        result: dict[str, Any] = {}
        for prim in prims:
            path = str(prim.GetPath())
            if path != root_path and not path.startswith(root_path + "/"):
                raise NativePlanningError("catalog_invalid")
            relative = path.removeprefix(root_path)
            if relative in result:
                raise NativePlanningError("catalog_invalid")
            result[relative] = prim
        return result

    def _collision_clone_signature(
        self,
        prim: Any,
        owner_body: Any,
        root_path: str,
        *,
        motion: PlanningGeometryMotionClass | None = None,
    ) -> tuple[object, ...]:
        self._validate_collision_common(prim)
        matrix = _effective_collision_relative_transform(
            self._xform_cache,
            prim,
            owner_body,
        )
        mesh_capable = not any(
            prim.IsA(schema) for schema in (self._m.UsdGeom.Cube, self._m.UsdGeom.Sphere, self._m.UsdGeom.Cylinder)
        )
        local_pose, scale, reflection, baked_linear = _collision_pose_scale_bake(
            self._m,
            matrix,
            mesh_capable=mesh_capable,
        )
        common: tuple[object, ...] = (
            self._prim_signature(prim, root_path),
            local_pose,
            scale,
            reflection,
            baked_linear,
            _motion_class(self._m, owner_body) if motion is None else motion,
        )
        if prim.IsA(self._m.UsdGeom.Cube):
            return (*common, "box", float(self._m.UsdGeom.Cube(prim).GetSizeAttr().Get()))
        if prim.IsA(self._m.UsdGeom.Sphere):
            return (*common, "sphere", float(self._m.UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
        if prim.IsA(self._m.UsdGeom.Cylinder):
            cylinder = self._m.UsdGeom.Cylinder(prim)
            return (
                *common,
                "cylinder",
                float(cylinder.GetRadiusAttr().Get()),
                float(cylinder.GetHeightAttr().Get()),
                str(cylinder.GetAxisAttr().Get() or "Z").upper(),
            )
        mesh_api = self._m.UsdPhysics.MeshCollisionAPI(prim)
        approximation = str(mesh_api.GetApproximationAttr().Get()) if mesh_api else ""
        if approximation not in {
            "boundingCube",
            "convexHull",
            "convexDecomposition",
            "meshSimplification",
            "sdf",
            "none",
        }:
            raise NativePlanningError("collision_geometry_unsupported")
        mesh_input = _reflect_mesh_input(self._mesh_input(prim), reflection)
        if baked_linear is not None:
            mesh_input = _bake_mesh_linear_transform(mesh_input, baked_linear)
        return (*common, approximation, mesh_input.source_sha256)

    def _verify_declared_clone_frames(self, spec: Any, reference_root: Any, clone_root: Any) -> None:
        declarations = parse_planning_frame_declarations(spec.metadata.get("planning_frame_declarations"))
        if declarations is None:
            return
        reference_prims = self._walk(reference_root)
        clone_prims = self._walk(clone_root)
        for declaration in declarations.entries:
            if declaration.source.kind is not PlanningFrameSourceKind.NATIVE_NAMED:
                continue
            reference_matches = tuple(prim for prim in reference_prims if prim.GetName() == declaration.source.name)
            clone_matches = tuple(prim for prim in clone_prims if prim.GetName() == declaration.source.name)
            if len(reference_matches) != 1 or len(clone_matches) != 1:
                raise NativePlanningError("frame_missing")
            if declaration.owner_link_name is None:
                root_link_name = self._entities[spec.path].root_link_name
                reference_owners = tuple(
                    prim
                    for prim in reference_prims
                    if prim.GetName() == root_link_name and prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                )
                clone_owners = tuple(
                    prim
                    for prim in clone_prims
                    if prim.GetName() == root_link_name and prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                )
                if len(reference_owners) != 1 or len(clone_owners) != 1:
                    raise NativePlanningError("frame_missing")
                reference_owner = reference_owners[0]
                clone_owner = clone_owners[0]
            else:
                reference_owners = tuple(
                    prim
                    for prim in reference_prims
                    if prim.GetName() == declaration.owner_link_name and prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                )
                clone_owners = tuple(
                    prim
                    for prim in clone_prims
                    if prim.GetName() == declaration.owner_link_name and prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
                )
                if len(reference_owners) != 1 or len(clone_owners) != 1:
                    raise NativePlanningError("frame_missing")
                reference_owner = reference_owners[0]
                clone_owner = clone_owners[0]
            reference_matrix, reference_resets = self._xform_cache.ComputeRelativeTransform(
                reference_matches[0], reference_owner
            )
            clone_matrix, clone_resets = self._xform_cache.ComputeRelativeTransform(
                clone_matches[0], clone_owner
            )
            if reference_resets or clone_resets:
                raise NativePlanningError("frame_ambiguous")
            if _matrix_local_pose(self._m, reference_matrix) != _matrix_local_pose(self._m, clone_matrix):
                raise NativePlanningError("frame_ambiguous")
            if self._prim_signature(reference_matches[0], str(reference_root.GetPath())) != self._prim_signature(
                clone_matches[0], str(clone_root.GetPath())
            ):
                raise NativePlanningError("frame_ambiguous")

    def _verify_cloned_environments(self) -> None:
        if self._world._spec.environments.count == 1:
            return
        for spec in self._world._spec.entities:
            if spec.kind is EntityKind.CAMERA_SENSOR or spec.embedded_binding is not None:
                continue
            reference_root = self._entity_root(spec, 0)
            reference_prims = self._walk(reference_root)
            reference_bodies = self._relative_prim_map(
                reference_root,
                tuple(prim for prim in reference_prims if prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)),
            )
            reference_colliders = self._relative_prim_map(
                reference_root,
                tuple(prim for prim in reference_prims if prim.HasAPI(self._m.UsdPhysics.CollisionAPI)),
            )
            reference_joints = self._relative_prim_map(
                reference_root,
                tuple(prim for prim in reference_prims if prim.IsA(self._m.UsdPhysics.Joint)),
            )
            reference_body_paths = {str(prim.GetPath()): relative for relative, prim in reference_bodies.items()}
            for environment_index in range(1, self._world._spec.environments.count):
                clone_root = self._entity_root(spec, environment_index)
                clone_prims = self._walk(clone_root)
                clone_bodies = self._relative_prim_map(
                    clone_root,
                    tuple(prim for prim in clone_prims if prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)),
                )
                clone_colliders = self._relative_prim_map(
                    clone_root,
                    tuple(prim for prim in clone_prims if prim.HasAPI(self._m.UsdPhysics.CollisionAPI)),
                )
                clone_joints = self._relative_prim_map(
                    clone_root,
                    tuple(prim for prim in clone_prims if prim.IsA(self._m.UsdPhysics.Joint)),
                )
                if (
                    clone_bodies.keys() != reference_bodies.keys()
                    or clone_colliders.keys() != reference_colliders.keys()
                    or clone_joints.keys() != reference_joints.keys()
                ):
                    raise NativePlanningError("topology_unsupported")
                clone_body_paths = {str(prim.GetPath()): relative for relative, prim in clone_bodies.items()}
                for relative, reference in reference_bodies.items():
                    clone = clone_bodies[relative]
                    if self._prim_signature(reference, str(reference_root.GetPath())) != self._prim_signature(
                        clone, str(clone_root.GetPath())
                    ):
                        raise NativePlanningError("topology_unsupported")
                    self._accounted_bodies.add(str(clone.GetPath()))
                    reference_path = str(reference.GetPath())
                    if reference_path in self._accounted_filtered_pair_sources:
                        self._accounted_filtered_pair_sources.add(str(clone.GetPath()))
                for relative, reference in reference_colliders.items():
                    clone = clone_colliders[relative]
                    reference_enabled = (
                        self._m.UsdPhysics.CollisionAPI(reference).GetCollisionEnabledAttr().Get() is not False
                    )
                    clone_enabled = self._m.UsdPhysics.CollisionAPI(clone).GetCollisionEnabledAttr().Get() is not False
                    if reference_enabled != clone_enabled:
                        raise NativePlanningError("catalog_invalid")
                    if not reference_enabled:
                        continue
                    reference_owner_path = _nearest_body(str(reference.GetPath()), reference_body_paths)
                    clone_owner_path = _nearest_body(str(clone.GetPath()), clone_body_paths)
                    if (reference_owner_path is None) != (clone_owner_path is None):
                        raise NativePlanningError("topology_unsupported")
                    if reference_owner_path is None:
                        if spec.kind is not EntityKind.COMPOSITE_SCENE:
                            raise NativePlanningError("catalog_invalid")
                        reference_owner = reference_root
                        clone_owner = clone_root
                        motion = PlanningGeometryMotionClass.STATIC
                    else:
                        assert clone_owner_path is not None
                        if reference_body_paths[reference_owner_path] != clone_body_paths[clone_owner_path]:
                            raise NativePlanningError("topology_unsupported")
                        reference_owner = reference_bodies[reference_body_paths[reference_owner_path]]
                        clone_owner = clone_bodies[clone_body_paths[clone_owner_path]]
                        motion = None
                    if self._collision_clone_signature(
                        reference,
                        reference_owner,
                        str(reference_root.GetPath()),
                        motion=motion,
                    ) != self._collision_clone_signature(
                        clone,
                        clone_owner,
                        str(clone_root.GetPath()),
                        motion=motion,
                    ):
                        raise NativePlanningError("collision_geometry_unsupported")
                    self._accounted_colliders.add(str(clone.GetPath()))
                for relative, reference in reference_joints.items():
                    clone = clone_joints[relative]
                    type_matches = (
                        reference.GetTypeName() == clone.GetTypeName()
                        if spec.kind is EntityKind.COMPOSITE_SCENE
                        else self._joint_type(reference) is self._joint_type(clone)
                    )
                    if not type_matches or self._prim_signature(
                        reference, str(reference_root.GetPath())
                    ) != self._prim_signature(clone, str(clone_root.GetPath())):
                        raise NativePlanningError("constraint_unsupported")
                    self._accounted_constraints.add(str(clone.GetPath()))
                self._verify_declared_clone_frames(spec, reference_root, clone_root)

    def _verify_complete_native_world(self) -> None:
        self._verify_cloned_environments()
        all_prims = self._walk(self._stage.GetPseudoRoot())
        collision_groups = tuple(prim for prim in all_prims if prim.IsA(self._m.UsdPhysics.CollisionGroup))
        if collision_groups:
            raise NativePlanningError("collision_filter_unsupported")
        rigid_body_paths = frozenset(
            str(prim.GetPath()) for prim in all_prims if prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI)
        )
        live_bodies: set[str] = set()
        live_colliders: set[str] = set()
        for prim in all_prims:
            filtered = self._m.UsdPhysics.FilteredPairsAPI(prim)
            if filtered:
                targets = tuple(filtered.GetFilteredPairsRel().GetTargets())
                if targets:
                    target_prims = tuple(self._stage.GetPrimAtPath(target) for target in targets)
                    target_paths = tuple(str(target.GetPath()) for target in target_prims if target.IsValid())
                    if (
                        str(prim.GetPath()) not in self._accounted_filtered_pair_sources
                        or str(prim.GetPath()) not in rigid_body_paths
                        or len(target_paths) != len(targets)
                        or any(path not in rigid_body_paths for path in target_paths)
                    ):
                        raise NativePlanningError("collision_filter_unsupported")
            if prim.HasAPI(self._m.UsdPhysics.RigidBodyAPI):
                live_bodies.add(str(prim.GetPath()))
            if prim.HasAPI(self._m.UsdPhysics.CollisionAPI):
                if self._m.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                    live_colliders.add(str(prim.GetPath()))
            if prim.IsA(self._m.UsdPhysics.Joint):
                path = str(prim.GetPath())
                if path not in self._accounted_constraints:
                    raise NativePlanningError("constraint_unsupported")
            type_name = prim.GetTypeName()
            if "Attachment" in type_name or "Constraint" in type_name:
                raise NativePlanningError("constraint_unsupported")
        if live_bodies != self._accounted_bodies or live_colliders != self._accounted_colliders:
            raise NativePlanningError("catalog_invalid")

    def state(self, environment_index: int) -> NativePlanningState:
        if type(environment_index) is not int or not 0 <= environment_index < self._world._spec.environments.count:
            raise NativePlanningError("generation_stale")
        poses: dict[tuple[EntityPath, str], PlanningPose] = {}
        twists: dict[tuple[EntityPath, str], PlanningTwist] = {}
        articulations: list[PlanningArticulationState] = []
        entity_states: list[PlanningEntityState] = []
        if _SYSTEM_FRAME_ID in self._frames:
            entity_states.append(PlanningEntityState(PLANNING_SYSTEM_ENTITY_ID, _IDENTITY_POSE, _ZERO_TWIST))
        link_states: list[PlanningLinkState] = []
        entity_pose_by_path: dict[EntityPath, PlanningPose] = {}
        xform_cache = self._m.UsdGeom.XformCache()

        for path, binding in self._entities.items():
            origin = self._world._origins_cpu[environment_index]
            asset = None
            if binding.state_source == "asset":
                asset = self._world._articulations.get(path) or self._world._rigids.get(path)
                if asset is None:
                    raise NativePlanningError("native_failure")
                body_names = tuple(asset.body_names)
                if set(body_names) != set(binding.link_id_by_name):
                    raise NativePlanningError("topology_unsupported")
                body_poses = asset.data.body_link_pose_w.torch[environment_index].detach().cpu().tolist()
                body_velocities = asset.data.body_link_vel_w.torch[environment_index].detach().cpu().tolist()
                body_rows = {name: (body_poses[index], body_velocities[index]) for index, name in enumerate(body_names)}
            else:
                tensor_bodies = binding.tensor_bodies
                if tensor_bodies is None and binding.link_id_by_name:
                    raise NativePlanningError("native_failure")
                if tensor_bodies is None:
                    body_rows = {}
                else:
                    transforms = tensor_bodies.view.get_transforms().detach().cpu().tolist()
                    velocities = tensor_bodies.view.get_velocities().detach().cpu().tolist()
                    body_rows = {
                        name: (transforms[index], velocities[index])
                        for name, index in tensor_bodies.row_by_environment_and_name[environment_index].items()
                    }
                if set(body_rows) != set(binding.link_id_by_name):
                    raise NativePlanningError("topology_unsupported")
            for name, (row, velocity) in body_rows.items():
                pose = PlanningPose(
                    _WORLD_FRAME_ID,
                    tuple(float(row[axis]) - origin[axis] for axis in range(3)),  # type: ignore[arg-type]
                    _xyzw(row[3:7]),
                )
                twist = PlanningTwist(
                    _WORLD_FRAME_ID,
                    _float_tuple(velocity, 3),  # type: ignore[arg-type]
                    _float_tuple(velocity[3:], 3),  # type: ignore[arg-type]
                )
                poses[path, name] = pose
                twists[path, name] = twist
                link_states.append(PlanningLinkState(binding.link_id_by_name[name], pose, twist))
            if binding.entity_prim_link_name is not None:
                entity_pose = poses[path, binding.entity_prim_link_name]
                entity_twist = twists[path, binding.entity_prim_link_name]
            else:
                prim = self._stage.GetPrimAtPath(binding.entity_prim_paths[environment_index])
                if not prim or not prim.IsValid() or not self._m.UsdGeom.Xformable(prim):
                    raise NativePlanningError("native_failure")
                prim_pose, _prim_scale = _matrix_local_pose(
                    self._m,
                    xform_cache.GetLocalToWorldTransform(prim),
                )
                entity_pose = PlanningPose(
                    _WORLD_FRAME_ID,
                    tuple(prim_pose.position_m[axis] - origin[axis] for axis in range(3)),  # type: ignore[arg-type]
                    prim_pose.orientation_xyzw,
                )
                entity_twist = _ZERO_TWIST
            if binding.static_poses:
                poses[path, _COMPOSITE_ENTITY_POSE] = entity_pose
                twists[path, _COMPOSITE_ENTITY_POSE] = entity_twist
            entity_pose_by_path[path] = entity_pose
            entity_states.append(PlanningEntityState(binding.entity_id, entity_pose, entity_twist))
            if binding.state_source == "asset" and path in self._world._articulations:
                assert asset is not None
                joint_names = tuple(asset.joint_names)
                positions = asset.data.joint_pos.torch[environment_index].detach().cpu().tolist()
                velocities = asset.data.joint_vel.torch[environment_index].detach().cpu().tolist()
                by_name = {
                    name: (float(positions[index]), float(velocities[index])) for index, name in enumerate(joint_names)
                }
                joint_positions = []
                joint_velocities = []
                units = []
                for joint in binding.joint_bindings:
                    if joint.movable_name is None:
                        joint_positions.append(0.0)
                        joint_velocities.append(0.0)
                    else:
                        try:
                            position, velocity = by_name[joint.movable_name]
                        except KeyError:
                            raise NativePlanningError("topology_unsupported") from None
                        joint_positions.append(position)
                        joint_velocities.append(velocity)
                    units.append(joint.descriptor.position_unit)
                articulations.append(
                    PlanningArticulationState(
                        binding.entity_id,
                        tuple(joint.descriptor.joint_id for joint in binding.joint_bindings),
                        tuple(joint_positions),
                        tuple(joint_velocities),
                        tuple(units),
                    )
                )
            elif binding.state_source == "usd_articulation":
                view = self._world._usd_articulation_views.get(path)
                joint_map = self._world._joint_maps.get(path)
                if view is None or joint_map is None or len(joint_map) != len(binding.joint_bindings):
                    raise NativePlanningError("topology_unsupported")
                native_positions = view.get_dof_positions()[environment_index].detach().cpu().tolist()
                native_velocities = view.get_dof_velocities()[environment_index].detach().cpu().tolist()
                articulations.append(
                    PlanningArticulationState(
                        binding.entity_id,
                        tuple(joint.descriptor.joint_id for joint in binding.joint_bindings),
                        tuple(float(native_positions[index]) for index in joint_map),
                        tuple(float(native_velocities[index]) for index in joint_map),
                        tuple(joint.descriptor.position_unit for joint in binding.joint_bindings),
                    )
                )

        frame_poses: dict[str, PlanningPose] = {_WORLD_FRAME_ID: _IDENTITY_POSE}
        if _SYSTEM_FRAME_ID in self._frames:
            frame_poses[_SYSTEM_FRAME_ID] = _IDENTITY_POSE
        for frame_id, frame in self._frames.items():
            if frame.entity_path is None:
                continue
            binding = self._entities[frame.entity_path]
            if frame.source == "entity_prim":
                frame_poses[frame_id] = entity_pose_by_path[frame.entity_path]
            elif frame.source in {"link", "static_entity"}:
                assert frame.source_name is not None
                frame_poses[frame_id] = poses[frame.entity_path, frame.source_name]
            elif frame.source == "joint":
                joint = next(
                    item for item in binding.joint_bindings if item.descriptor.authored_name == frame.source_name
                )
                parent_pose = poses[frame.entity_path, joint.parent_name]
                frame_poses[frame_id] = _compose_pose(parent_pose, joint.local_pose)
            elif frame.source == "joint_id":
                joint = next(item for item in binding.joint_bindings if item.descriptor.joint_id == frame.source_name)
                parent_pose = poses[frame.entity_path, joint.parent_name]
                frame_poses[frame_id] = _compose_pose(parent_pose, joint.local_pose)
            elif frame.source == "native":
                assert frame.local_pose is not None
                parent_pose = (
                    poses[frame.entity_path, frame.source_name]
                    if frame.source_name is not None
                    else poses[frame.entity_path, binding.root_link_name]
                )
                frame_poses[frame_id] = _compose_pose(parent_pose, frame.local_pose)
        frame_states = tuple(PlanningFrameState(frame_id, frame_poses[frame_id]) for frame_id in sorted(frame_poses))
        geometry_transforms = []
        for geometry_id, geometry_binding in sorted(self._geometries.items()):
            if geometry_binding.entity_path is None:
                parent_pose = _IDENTITY_POSE
            elif geometry_binding.owner_link_name is None:
                parent_pose = entity_pose_by_path[geometry_binding.entity_path]
            else:
                parent_pose = poses[geometry_binding.entity_path, geometry_binding.owner_link_name]
            geometry_transforms.append(
                PlanningGeometryTransform(
                    geometry_id,
                    _compose_pose(parent_pose, geometry_binding.descriptor.parent_frame_T_geometry),
                )
            )
        attachments: list[PlanningAttachment] = []
        for (attachment_environment, _attachment_id), attachment in sorted(
            getattr(self._world, "_runtime_attachments", {}).items()
        ):
            if attachment_environment != environment_index:
                continue
            parent = self._entities.get(attachment.parent_path)
            child = self._entities.get(attachment.child_path)
            if parent is None or child is None:
                raise NativePlanningError("attachment_entity_missing")
            parent_link_name = attachment.parent_link_name or parent.root_link_name
            child_link_name = attachment.child_link_name or child.root_link_name
            parent_link_id = parent.link_id_by_name.get(parent_link_name)
            child_link_id = child.link_id_by_name.get(child_link_name)
            if parent_link_id is None or child_link_id is None:
                raise NativePlanningError("attachment_link_missing")

            def link_frame_id(link_id: str) -> str:
                matches = tuple(
                    link.frame_id
                    for link in self._catalog.links
                    if link.link_id == link_id
                )
                if len(matches) != 1:
                    raise NativePlanningError("attachment_frame_missing")
                return matches[0]

            parent_frame_id = link_frame_id(parent_link_id)
            child_frame_id = link_frame_id(child_link_id)
            geometry_ids = tuple(
                geometry.descriptor.geometry_id
                for geometry in self._geometries.values()
                if geometry.entity_path == attachment.child_path
                and geometry.descriptor.owner_link_id == child_link_id
            )
            if not geometry_ids:
                raise NativePlanningError("attachment_geometry_missing")
            try:
                current_relative = _relative_pose(
                    poses[attachment.parent_path, parent_link_name],
                    poses[attachment.child_path, child_link_name],
                    parent_frame_id,
                )
            except KeyError:
                raise NativePlanningError("attachment_pose_missing") from None
            except NativePlanningError:
                raise
            except Exception:
                raise NativePlanningError("attachment_pose_invalid") from None
            try:
                attachments.append(
                    PlanningAttachment(
                        attachment.attachment_id,
                        parent.entity_id,
                        child.entity_id,
                        parent_frame_id,
                        child_frame_id,
                        current_relative,
                        tuple(sorted(geometry_ids)),
                        parent_link_id,
                        child_link_id,
                    )
                )
            except NativePlanningError:
                raise
            except Exception:
                raise NativePlanningError("attachment_value_invalid") from None
        return NativePlanningState(
            self._world._step_index,
            tuple(sorted(entity_states, key=lambda item: item.entity_id)),
            tuple(sorted(link_states, key=lambda item: item.link_id)),
            frame_states,
            tuple(sorted(articulations, key=lambda item: item.entity_id)),
            tuple(geometry_transforms),
            tuple(sorted(attachments, key=lambda item: item.attachment_id)),
        )


class IsaacLabNativePlanningWorld(IsaacLabNativeWorld):
    """Native World subtype instantiated only for an explicit planning demand."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._planning_admission: _PlanningAdmission | None = None
        super().__init__(*args, **kwargs)

    def _build(self) -> None:
        super()._build()
        self._planning_admission = _PlanningAdmission(self)

    def _planning(self) -> _PlanningAdmission:
        if self._planning_admission is None:
            raise NativePlanningError("native_failure")
        return self._planning_admission

    def planning_catalog(self, environment_index: int = 0) -> NativePlanningCatalog:
        if type(environment_index) is not int or not 0 <= environment_index < self._spec.environments.count:
            raise NativePlanningError("generation_stale")
        return self._planning().catalog

    def planning_state(self, environment_index: int = 0) -> NativePlanningState:
        return self._planning().state(environment_index)

    def planning_resource(self, geometry_id: str, environment_index: int = 0) -> NativePlanningResource:
        if type(environment_index) is not int or not 0 <= environment_index < self._spec.environments.count:
            raise NativePlanningError("generation_stale")
        if type(geometry_id) is not str or not geometry_id:
            raise NativePlanningError("resource_missing")
        return self._planning().resource(geometry_id)

__all__ = ["IsaacLabNativePlanningWorld"]
