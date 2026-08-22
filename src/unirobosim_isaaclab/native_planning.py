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
_PROVENANCE_PROFILE = "isaaclab-3.0-physx-convex-canonical-v1"
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
class _EntityBinding:
    path: EntityPath
    entity_id: str
    root_link_name: str
    root_link_id: str
    link_id_by_name: dict[str, str]
    joint_bindings: tuple[_JointBinding, ...]


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


def _matrix_local_pose(modules: Any, matrix: Any) -> tuple[PlanningGeometryLocalPose, tuple[float, float, float]]:
    transform = modules.Gf.Transform(matrix)
    translation = _float_tuple(transform.GetTranslation(), 3)
    raw_scale = _float_tuple(transform.GetScale(), 3)
    scale = (raw_scale[0], raw_scale[1], raw_scale[2])
    if any(item <= 0.0 for item in scale):
        raise NativePlanningError("collision_geometry_unsupported")
    scale_orientation = _xyzw(transform.GetPivotOrientation().GetQuat())
    pivot = _float_tuple(transform.GetPivotPosition(), 3)
    if scale_orientation != (0.0, 0.0, 0.0, 1.0) or any(abs(item) > 1.0e-10 for item in pivot):
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
    lengths = tuple(math.sqrt(sum(component * component for component in vector)) for vector in basis)
    tolerance = 1.0e-7 * max(1.0, *scale)
    if any(abs(origin[index] - translation[index]) > tolerance for index in range(3)) or any(
        abs(lengths[index] - scale[index]) > tolerance for index in range(3)
    ):
        raise NativePlanningError("collision_geometry_unsupported")
    if any(
        abs(sum(basis[left][axis] * basis[right][axis] for axis in range(3))) > tolerance
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise NativePlanningError("collision_geometry_unsupported")
    orientation = _xyzw(transform.GetRotation().GetQuat())
    return PlanningGeometryLocalPose(translation, orientation), scale  # type: ignore[arg-type]


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
        self._accounted_bodies: set[str] = set()
        self._accounted_colliders: set[str] = set()
        self._accounted_constraints: set[str] = set()
        self._catalog = self._build_catalog()

    @property
    def catalog(self) -> NativePlanningCatalog:
        return self._catalog

    def resource(self, geometry_id: str) -> NativePlanningResource:
        try:
            return self._resources[geometry_id]
        except KeyError:
            raise NativePlanningError("resource_missing") from None

    def _walk(self, root: Any) -> tuple[Any, ...]:
        return tuple(self._m.Usd.PrimRange(root, self._m.Usd.TraverseInstanceProxies()))

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

    def _build_catalog(self) -> NativePlanningCatalog:
        frames: list[PlanningFrameDescriptor] = [
            PlanningFrameDescriptor(_WORLD_FRAME_ID, PlanningFrameKind.WORLD, None, None, None)
        ]
        entities: list[PlanningEntityDescriptor] = []
        links: list[PlanningLinkDescriptor] = []
        joints: list[PlanningJointDescriptor] = []
        geometries: list[PlanningGeometryDescriptor] = []

        system_geometry = self._ground_geometry()
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
        self._frames[_SYSTEM_FRAME_ID] = _FrameBinding(_SYSTEM_FRAME_ID, None, "system")

        for spec in self._world._spec.entities:
            if spec.kind is EntityKind.CAMERA_SENSOR:
                self._verify_nonphysical_entity(spec)
                continue
            if spec.kind not in {EntityKind.ARTICULATION, EntityKind.RIGID_BODY}:
                raise NativePlanningError("soft_matter_unsupported")
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

    def _ground_geometry(self) -> PlanningGeometryDescriptor:
        root = self._stage.GetPrimAtPath("/World/unirobosimGround")
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
            self._m.UsdGeom.XformCache().GetLocalToWorldTransform(plane),
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
            if set(movable_names) != set(asset_names) or set(spec.joint_names) != set(asset_names):
                raise NativePlanningError("topology_unsupported")
        elif movable_names:
            raise NativePlanningError("topology_unsupported")

        geometry_by_link: dict[str, list[PlanningGeometryDescriptor]] = {name: [] for name in link_id_by_name}
        for prim in self._walk(root):
            if not prim.HasAPI(self._m.UsdPhysics.CollisionAPI):
                continue
            collision = self._m.UsdPhysics.CollisionAPI(prim)
            if collision.GetCollisionEnabledAttr().Get() is False:
                continue
            owner_path = _nearest_body(str(prim.GetPath()), bodies)
            if owner_path is None:
                raise NativePlanningError("catalog_invalid")
            owner_name = body_name_by_path[owner_path]
            geometry = self._collision_geometry(
                spec,
                entity_id,
                prim,
                bodies[owner_path],
                link_id_by_name[owner_name],
                link_frame_by_name[owner_name],
            )
            geometry_by_link[owner_name].append(geometry)
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
            parent_name = parent_name_by_child.get(name)
            frame_id = link_frame_by_name[name]
            frame_parent = entity_frame_id if parent_name is None else joint_frame_by_child[name]
            geometry_ids = tuple(sorted(item.geometry_id for item in geometry_by_link[name]))
            link_descriptors.append(
                PlanningLinkDescriptor(
                    link_id_by_name[name],
                    entity_id,
                    name,
                    frame_id,
                    None if parent_name is None else link_id_by_name[parent_name],
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
        self._frames[entity_frame_id] = _FrameBinding(entity_frame_id, spec.path, "entity", root_name)

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
            ),
        )

    def _entity_kind(self, spec: Any) -> PlanningEntityKind:
        locked = spec.metadata.get("planning_entity_kind")
        if locked is not None:
            return PlanningEntityKind(locked)
        if spec.kind is EntityKind.ARTICULATION:
            return (
                PlanningEntityKind.ROBOT if spec.path.value.startswith("/robots/") else PlanningEntityKind.ARTICULATION
            )
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
    ) -> PlanningJointDescriptor:
        if joint_type is PlanningJointType.FIXED:
            return PlanningJointDescriptor(
                joint_id,
                entity_id,
                prim.GetName(),
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
            prim.GetName(),
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
        allowed = {
            "PhysicsCollisionAPI",
            "PhysicsMeshCollisionAPI",
            "PhysxContactReportAPI",
        }
        unexpected = {
            schema
            for schema in applied
            if ("Collision" in schema or "FilteredPairs" in schema) and schema not in allowed
        }
        if unexpected:
            raise NativePlanningError("collision_geometry_unsupported")
        filtered = self._m.UsdPhysics.FilteredPairsAPI(prim)
        if filtered and tuple(filtered.GetFilteredPairsRel().GetTargets()):
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
        owner_link_id: str,
        owner_frame_id: str,
    ) -> PlanningGeometryDescriptor:
        self._validate_collision_common(prim)
        cache = self._m.UsdGeom.XformCache()
        matrix, resets = cache.ComputeRelativeTransform(prim, owner_body)
        if resets:
            raise NativePlanningError("collision_geometry_unsupported")
        local_pose, scale = _matrix_local_pose(self._m, matrix)
        geometry_id = _stable_id("geometry", spec.path.value, str(prim.GetPath()))
        motion = _motion_class(self._m, owner_body)
        source_sha = self._source_sha256(spec)
        common = {
            "adapter": _PROVENANCE_PROFILE,
            "source_sha256": source_sha,
            "native_path": str(prim.GetPath()).removeprefix(f"/World/env_0/{_native_name(spec.path)}"),
            "applied_schemas": sorted(prim.GetAppliedSchemas()),
            "local_pose": [local_pose.position_m, local_pose.orientation_xyzw],
            "scale": scale,
        }
        if prim.IsA(self._m.UsdGeom.Cube):
            size = float(self._m.UsdGeom.Cube(prim).GetSizeAttr().Get())
            if not math.isfinite(size) or size <= 0.0:
                raise NativePlanningError("collision_geometry_unsupported")
            inline = PlanningPrimitiveGeometry(PlanningGeometryRepresentation.BOX, (size, size, size))
            return PlanningGeometryDescriptor(
                geometry_id,
                entity_id,
                owner_link_id,
                owner_frame_id,
                PlanningGeometryPurpose.COLLISION,
                PlanningGeometryRepresentation.BOX,
                local_pose,
                scale,
                motion,
                _DEFAULT_COLLISION_GROUP,
                _DEFAULT_COLLISION_MASK,
                _json_sha256({**common, "representation": "box", "dimensions": inline.dimensions_m}),
                inline=inline,
            )
        if prim.IsA(self._m.UsdGeom.Sphere):
            radius = float(self._m.UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            if not math.isfinite(radius) or radius <= 0.0:
                raise NativePlanningError("collision_geometry_unsupported")
            inline = PlanningPrimitiveGeometry(PlanningGeometryRepresentation.SPHERE, (radius,))
            return PlanningGeometryDescriptor(
                geometry_id,
                entity_id,
                owner_link_id,
                owner_frame_id,
                PlanningGeometryPurpose.COLLISION,
                PlanningGeometryRepresentation.SPHERE,
                local_pose,
                scale,
                motion,
                _DEFAULT_COLLISION_GROUP,
                _DEFAULT_COLLISION_MASK,
                _json_sha256({**common, "representation": "sphere", "dimensions": inline.dimensions_m}),
                inline=inline,
            )

        mesh_api = self._m.UsdPhysics.MeshCollisionAPI(prim)
        approximation = str(mesh_api.GetApproximationAttr().Get()) if mesh_api else ""
        if approximation != "convexHull":
            raise NativePlanningError("collision_geometry_unsupported")
        content, vertex_count, triangle_count, input_sha = self._cook_exact_convex(prim)
        if len(content) > _MAX_GEOMETRY_RESOURCE_BYTES:
            raise NativePlanningError("collision_cooking_failed")
        digest = hashlib.sha256(content).hexdigest()
        layout = PlanningGeometryResourceLayout(
            PlanningGeometryRepresentation.CONVEX_MESH,
            PlanningGeometryContentProfile.MESH_TRIANGLES_RAW_LE_V1,
            PlanningGeometryDType.FLOAT32,
            (vertex_count, 3),
            PlanningGeometryDType.UINT32,
            (triangle_count, 3),
        )
        provenance = _json_sha256(
            {
                **common,
                "representation": "convex_mesh",
                "approximation": approximation,
                "cooked_sha256": digest,
                "cooking_input_sha256": input_sha,
                "canonicalization": "float32-le-vertices-then-uint32-le-triangles-v1",
            }
        )
        descriptor = PlanningGeometryDescriptor(
            geometry_id,
            entity_id,
            owner_link_id,
            owner_frame_id,
            PlanningGeometryPurpose.COLLISION,
            PlanningGeometryRepresentation.CONVEX_MESH,
            local_pose,
            scale,
            motion,
            _DEFAULT_COLLISION_GROUP,
            _DEFAULT_COLLISION_MASK,
            provenance,
            resource_id=_stable_id("resource", spec.path.value, str(prim.GetPath()), digest),
            sha256=digest,
            content_profile=PlanningGeometryContentProfile.MESH_TRIANGLES_RAW_LE_V1,
            resource_layout=layout,
        )
        self._resources[geometry_id] = NativePlanningResource(
            geometry_id,
            PlanningGeometryRepresentation.CONVEX_MESH,
            content,
            digest,
        )
        return descriptor

    def _cook_exact_convex(self, carrier: Any) -> tuple[bytes, int, int, str]:
        descendants = tuple(prim for prim in self._walk(carrier) if prim != carrier and prim.IsA(self._m.UsdGeom.Mesh))
        nested_colliders = tuple(
            prim for prim in self._walk(carrier) if prim != carrier and prim.HasAPI(self._m.UsdPhysics.CollisionAPI)
        )
        if len(descendants) != 1 or nested_colliders:
            raise NativePlanningError("collision_cooking_failed")
        mesh_prim = descendants[0]
        mesh = self._m.UsdGeom.Mesh(mesh_prim)
        points = tuple(mesh.GetPointsAttr().Get() or ())
        counts = tuple(int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or ()))
        indices = tuple(int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or ()))
        if (
            not points
            or not counts
            or sum(counts) != len(indices)
            or not any(count >= 3 for count in counts)
            or any(count in {1, 2} for count in counts)
        ):
            raise NativePlanningError("collision_cooking_failed")
        matrix, resets = self._m.UsdGeom.XformCache().ComputeRelativeTransform(mesh_prim, carrier)
        if resets:
            raise NativePlanningError("collision_cooking_failed")
        transformed = tuple(matrix.Transform(point) for point in points)
        vertices = tuple(_float_tuple(point, 3) for point in transformed)
        if any(index < 0 or index >= len(vertices) for index in indices):
            raise NativePlanningError("collision_cooking_failed")
        input_sha = _json_sha256(
            {
                "vertices": vertices,
                "face_counts": counts,
                "face_indices": indices,
                "subdivision": str(mesh.GetSubdivisionSchemeAttr().Get() or "none"),
                "orientation": str(mesh.GetOrientationAttr().Get() or "rightHanded"),
            }
        )

        import omni.physx  # type: ignore[import-not-found]
        from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt  # type: ignore[import-not-found]

        temporary = Usd.Stage.CreateInMemory()
        clone = UsdGeom.Mesh.Define(temporary, "/collision")
        clone.GetPointsAttr().Set(Vt.Vec3fArray(vertices))
        clone.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
        clone.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
        clone.GetSubdivisionSchemeAttr().Set("none")
        clone.GetOrientationAttr().Set(str(mesh.GetOrientationAttr().Get() or "rightHanded"))
        UsdPhysics.CollisionAPI.Apply(clone.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(clone.GetPrim())
        mesh_collision.GetApproximationAttr().Set("convexHull")
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
        if len(result_values) != 1 or "RESULT_VALID" not in repr(result_values[0]) or len(convex_values) != 1:
            raise NativePlanningError("collision_cooking_failed")
        cooked: Any = convex_values[0]
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
        return vertex_bytes + index_bytes, len(cooked_vertices), len(triangles), input_sha

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
                matrix, resets = self._m.UsdGeom.XformCache().ComputeRelativeTransform(matches[0], owner_prim)
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

    def _collision_clone_signature(self, prim: Any, owner_body: Any, root_path: str) -> tuple[object, ...]:
        self._validate_collision_common(prim)
        matrix, resets = self._m.UsdGeom.XformCache().ComputeRelativeTransform(prim, owner_body)
        if resets:
            raise NativePlanningError("collision_geometry_unsupported")
        local_pose, scale = _matrix_local_pose(self._m, matrix)
        common: tuple[object, ...] = (
            self._prim_signature(prim, root_path),
            local_pose,
            scale,
            _motion_class(self._m, owner_body),
        )
        if prim.IsA(self._m.UsdGeom.Cube):
            return (*common, "box", float(self._m.UsdGeom.Cube(prim).GetSizeAttr().Get()))
        if prim.IsA(self._m.UsdGeom.Sphere):
            return (*common, "sphere", float(self._m.UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
        mesh_api = self._m.UsdPhysics.MeshCollisionAPI(prim)
        approximation = str(mesh_api.GetApproximationAttr().Get()) if mesh_api else ""
        if approximation != "convexHull":
            raise NativePlanningError("collision_geometry_unsupported")
        descendants = tuple(item for item in self._walk(prim) if item != prim and item.IsA(self._m.UsdGeom.Mesh))
        nested_colliders = tuple(
            item for item in self._walk(prim) if item != prim and item.HasAPI(self._m.UsdPhysics.CollisionAPI)
        )
        if len(descendants) != 1 or nested_colliders:
            raise NativePlanningError("collision_cooking_failed")
        mesh = descendants[0]
        child_matrix, child_resets = self._m.UsdGeom.XformCache().ComputeRelativeTransform(mesh, prim)
        if child_resets:
            raise NativePlanningError("collision_cooking_failed")
        child_pose, child_scale = _matrix_local_pose(self._m, child_matrix)
        mesh_source: object
        if mesh.IsInstanceProxy():
            mesh_source = ("prototype", str(mesh.GetPrimInPrototype().GetPath()))
        else:
            mesh_source = ("authored", self._prim_signature(mesh, root_path))
        return (*common, "convexHull", child_pose, child_scale, mesh_source)

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
            reference_matrix, reference_resets = self._m.UsdGeom.XformCache().ComputeRelativeTransform(
                reference_matches[0], reference_owner
            )
            clone_matrix, clone_resets = self._m.UsdGeom.XformCache().ComputeRelativeTransform(
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
        for spec in self._world._spec.entities:
            if spec.kind is EntityKind.CAMERA_SENSOR:
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
                    if reference_owner_path is None or clone_owner_path is None:
                        raise NativePlanningError("catalog_invalid")
                    if reference_body_paths[reference_owner_path] != clone_body_paths[clone_owner_path]:
                        raise NativePlanningError("topology_unsupported")
                    if self._collision_clone_signature(
                        reference,
                        reference_bodies[reference_body_paths[reference_owner_path]],
                        str(reference_root.GetPath()),
                    ) != self._collision_clone_signature(
                        clone,
                        clone_bodies[clone_body_paths[clone_owner_path]],
                        str(clone_root.GetPath()),
                    ):
                        raise NativePlanningError("collision_geometry_unsupported")
                    self._accounted_colliders.add(str(clone.GetPath()))
                for relative, reference in reference_joints.items():
                    clone = clone_joints[relative]
                    reference_type = self._joint_type(reference)
                    clone_type = self._joint_type(clone)
                    if reference_type is not clone_type or self._prim_signature(
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
        live_bodies: set[str] = set()
        live_colliders: set[str] = set()
        for prim in all_prims:
            filtered = self._m.UsdPhysics.FilteredPairsAPI(prim)
            if filtered and tuple(filtered.GetFilteredPairsRel().GetTargets()):
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
        entity_states: list[PlanningEntityState] = [
            PlanningEntityState(PLANNING_SYSTEM_ENTITY_ID, _IDENTITY_POSE, _ZERO_TWIST)
        ]
        link_states: list[PlanningLinkState] = []

        for path, binding in self._entities.items():
            asset = self._world._articulations.get(path) or self._world._rigids.get(path)
            if asset is None:
                raise NativePlanningError("native_failure")
            body_names = tuple(asset.body_names)
            if set(body_names) != set(binding.link_id_by_name):
                raise NativePlanningError("topology_unsupported")
            body_poses = asset.data.body_link_pose_w.torch[environment_index].detach().cpu().tolist()
            body_velocities = asset.data.body_link_vel_w.torch[environment_index].detach().cpu().tolist()
            origin = self._world._origins_cpu[environment_index]
            for index, name in enumerate(body_names):
                row = body_poses[index]
                pose = PlanningPose(
                    _WORLD_FRAME_ID,
                    tuple(float(row[axis]) - origin[axis] for axis in range(3)),  # type: ignore[arg-type]
                    _xyzw(row[3:7]),
                )
                velocity = body_velocities[index]
                twist = PlanningTwist(
                    _WORLD_FRAME_ID,
                    _float_tuple(velocity, 3),  # type: ignore[arg-type]
                    _float_tuple(velocity[3:], 3),  # type: ignore[arg-type]
                )
                poses[path, name] = pose
                twists[path, name] = twist
                link_states.append(PlanningLinkState(binding.link_id_by_name[name], pose, twist))
            root_pose = poses[path, binding.root_link_name]
            root_twist = twists[path, binding.root_link_name]
            entity_states.append(PlanningEntityState(binding.entity_id, root_pose, root_twist))
            if path in self._world._articulations:
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

        frame_poses: dict[str, PlanningPose] = {
            _WORLD_FRAME_ID: _IDENTITY_POSE,
            _SYSTEM_FRAME_ID: _IDENTITY_POSE,
        }
        for frame_id, frame in self._frames.items():
            if frame.entity_path is None:
                continue
            binding = self._entities[frame.entity_path]
            if frame.source in {"entity", "link"}:
                assert frame.source_name is not None
                frame_poses[frame_id] = poses[frame.entity_path, frame.source_name]
            elif frame.source == "joint":
                joint = next(
                    item for item in binding.joint_bindings if item.descriptor.authored_name == frame.source_name
                )
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
            else:
                assert geometry_binding.owner_link_name is not None
                parent_pose = poses[geometry_binding.entity_path, geometry_binding.owner_link_name]
            geometry_transforms.append(
                PlanningGeometryTransform(
                    geometry_id,
                    _compose_pose(parent_pose, geometry_binding.descriptor.parent_frame_T_geometry),
                )
            )
        return NativePlanningState(
            self._world._step_index,
            tuple(sorted(entity_states, key=lambda item: item.entity_id)),
            tuple(sorted(link_states, key=lambda item: item.link_id)),
            frame_states,
            tuple(sorted(articulations, key=lambda item: item.entity_id)),
            tuple(geometry_transforms),
            (),
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
