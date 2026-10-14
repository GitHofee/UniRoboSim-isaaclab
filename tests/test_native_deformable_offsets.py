from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from unirobosim import ArrayValue, DeformableBodySpec, DeformableTopology, EntityKind, EntityPath, EntitySpec

from unirobosim_isaaclab.native import IsaacLabNativeWorld


class FakeAttribute:
    def __init__(self, prim: FakePrim, name: str) -> None:
        self.prim = prim
        self.name = name

    def Set(self, value: object) -> bool:
        if self.prim.stage.rejected_attribute == self.name:
            return False
        self.prim.attributes[self.name] = value
        self.prim.stage.events.append(("attribute", self.prim.path, self.name))
        return True


class FakePrim:
    def __init__(self, stage: FakeStage, path: str, type_name: str) -> None:
        self.stage = stage
        self.path = path
        self.type_name = type_name
        self.apis: set[str] = set()
        self.attributes: dict[str, object] = {}

    def HasAPI(self, schema: str) -> bool:
        return schema in self.apis

    def IsA(self, schema: Any) -> bool:
        return self.type_name == schema.type_name

    def GetPath(self) -> str:
        return self.path


class FakeStage:
    def __init__(self) -> None:
        self.prims: dict[str, FakePrim] = {}
        self.events: list[tuple[str, ...]] = []
        self.rejected_attribute: str | None = None

    def define(self, path: str, type_name: str) -> FakePrim:
        prim = FakePrim(self, path, type_name)
        self.prims[path] = prim
        return prim

    def GetPrimAtPath(self, path: str) -> FakePrim | None:
        return self.prims.get(path)

    def RemovePrim(self, path: str) -> None:
        self.prims = {key: value for key, value in self.prims.items() if key != path and not key.startswith(path + "/")}


class FakeMesh:
    type_name = "Mesh"

    def __init__(self, prim: FakePrim) -> None:
        self.prim = prim

    @classmethod
    def Define(cls, stage: FakeStage, path: str) -> FakeMesh:
        return cls(stage.define(path, cls.type_name))

    def GetPointsAttr(self) -> FakeAttribute:
        return FakeAttribute(self.prim, "points")

    def GetFaceVertexIndicesAttr(self) -> FakeAttribute:
        return FakeAttribute(self.prim, "faceVertexIndices")

    def GetFaceVertexCountsAttr(self) -> FakeAttribute:
        return FakeAttribute(self.prim, "faceVertexCounts")


class FakeTetMesh(FakeMesh):
    type_name = "TetMesh"

    def GetTetVertexIndicesAttr(self) -> FakeAttribute:
        return FakeAttribute(self.prim, "tetVertexIndices")


class FakeCollisionAPI:
    def __init__(self, prim: FakePrim) -> None:
        self.prim = prim

    def CreateContactOffsetAttr(self) -> FakeAttribute:
        return FakeAttribute(self.prim, "physxCollision:contactOffset")

    def CreateRestOffsetAttr(self) -> FakeAttribute:
        return FakeAttribute(self.prim, "physxCollision:restOffset")


def native_fixture(
    kind: EntityKind,
    *,
    environments: int = 1,
    contact: float | None = 0.0005,
    rest: float = 0.0,
    failure: str | None = None,
    surface_collision: str = "sim_mesh",
    existing_offsets: tuple[float, float] | None = None,
) -> tuple[IsaacLabNativeWorld, EntitySpec, FakeStage, list[dict[str, Any]], list[str]]:
    stage = FakeStage()
    body_configs: list[dict[str, Any]] = []
    collision_applications: list[str] = []
    points = ArrayValue.from_rows(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    surface = kind is EntityKind.SURFACE_DEFORMABLE
    body = DeformableBodySpec(
        DeformableTopology.SURFACE if surface else DeformableTopology.VOLUME,
        points,
        surface_triangles=ArrayValue.from_rows(((0, 1, 2),), dtype="int64") if surface else None,
        tetrahedra=None if surface else ArrayValue.from_rows(((0, 1, 2, 3),), dtype="int64"),
        node_mass_kg=0.2,
        linear_damping_per_s=0.7,
        self_collision=True,
        contact_offset_m=contact,
        rest_offset_m=rest,
    )
    entity = EntitySpec(EntityPath("/soft"), kind, deformable=body)

    def create_prim(path: str, type_name: str, **kwargs: Any) -> FakePrim:
        del kwargs
        return stage.define(path, type_name)

    def body_properties(**kwargs: Any) -> SimpleNamespace:
        body_configs.append(kwargs)
        return SimpleNamespace(**kwargs)

    def define_body(root: str, properties: SimpleNamespace, *, deformable_type: str) -> None:
        stage.events.append(("body_define", root))
        # Model the SDK boundary regression: body cfg collision fields, if
        # supplied, are written on the passed body root rather than the mesh.
        for name in ("contact_offset", "rest_offset"):
            if getattr(properties, name, None) is not None:
                stage.GetPrimAtPath(root).attributes[name] = getattr(properties, name)
        if failure == "missing":
            return
        path = root + ("/" + surface_collision if deformable_type == "surface" else "/sim_mesh")
        if path not in stage.prims:
            stage.define(path, "Mesh")
        collider = stage.GetPrimAtPath(path)
        collider.apis.add("PhysicsCollisionAPI")
        if existing_offsets is not None:
            collider.attributes.update(
                zip(
                    ("physxCollision:contactOffset", "physxCollision:restOffset"),
                    existing_offsets,
                    strict=True,
                )
            )
        if failure == "ambiguous":
            stage.define(root + "/other_collision", collider.type_name).apis.add("PhysicsCollisionAPI")
        elif failure == "topology":
            collider.type_name = "TetMesh" if deformable_type == "surface" else "Mesh"

    def apply_collision(prim: FakePrim) -> FakeCollisionAPI | None:
        collision_applications.append(prim.path)
        stage.events.append(("apply_collision", prim.path))
        if failure == "schema":
            return None
        prim.apis.add("PhysxCollisionAPI")
        return FakeCollisionAPI(prim)

    def prim_range(prim: FakePrim, _predicate: object) -> tuple[FakePrim, ...]:
        return tuple(p for path, p in stage.prims.items() if path == prim.path or path.startswith(prim.path + "/"))

    world = object.__new__(IsaacLabNativeWorld)
    world._config = SimpleNamespace(deformable_mode="simulation")
    world._spec = SimpleNamespace(environments=SimpleNamespace(count=environments))
    world._deformables = {}
    world._m = SimpleNamespace(
        sim_utils=SimpleNamespace(get_current_stage=lambda: stage, create_prim=create_prim),
        UsdGeom=SimpleNamespace(Mesh=FakeMesh, TetMesh=FakeTetMesh),
        UsdPhysics=SimpleNamespace(CollisionAPI="PhysicsCollisionAPI"),
        Usd=SimpleNamespace(PrimRange=prim_range, TraverseInstanceProxies=lambda: None),
        Vt=SimpleNamespace(Vec3fArray=tuple, Vec4iArray=tuple, IntArray=tuple),
        PhysxSchema=SimpleNamespace(PhysxCollisionAPI=SimpleNamespace(Apply=apply_collision)),
        PhysxDeformableBodyPropertiesCfg=body_properties,
        define_deformable_body_properties=define_body,
        DeformableObjectCfg=lambda **kwargs: SimpleNamespace(**kwargs),
        DeformableObject=lambda cfg: SimpleNamespace(cfg=cfg),
    )
    return world, entity, stage, body_configs, collision_applications


@pytest.mark.parametrize("kind", [EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE])
@pytest.mark.parametrize("environments", [1, 3])
@pytest.mark.parametrize(("contact", "rest"), [(0.0005, 0.0), (0.004, 0.0015)])
def test_explicit_offsets_target_actual_collision_mesh_in_every_environment(
    kind: EntityKind,
    environments: int,
    contact: float,
    rest: float,
) -> None:
    world, entity, stage, configs, applications = native_fixture(
        kind,
        environments=environments,
        contact=contact,
        rest=rest,
    )
    world._author_deformable(entity)
    assert len(applications) == environments
    for path in applications:
        assert path.endswith("/sim_mesh")
        prim = stage.prims[path]
        assert prim.attributes["physxCollision:contactOffset"] == contact
        assert prim.attributes["physxCollision:restOffset"] == rest
        assert "PhysxCollisionAPI" in prim.apis
        root = path.rsplit("/", 1)[0]
        assert not any("offset" in name.lower() for name in stage.prims[root].attributes)
        assert stage.events.index(("body_define", root)) < stage.events.index(("apply_collision", path))
        if kind is EntityKind.VOLUME_DEFORMABLE:
            assert not any("physxCollision:" in name for name in stage.prims[root + "/vis_mesh"].attributes)
    assert all(cfg["mass"] == pytest.approx(0.8) and cfg["linear_damping"] == 0.7 for cfg in configs)
    assert all(cfg["self_collision"] is True and cfg["solver_position_iteration_count"] == 16 for cfg in configs)
    assert all("contact_offset" not in cfg and "rest_offset" not in cfg for cfg in configs)
    assert entity.path in world._deformables


def test_surface_offsets_follow_collision_api_on_visual_mesh_when_sdk_uses_it() -> None:
    world, entity, stage, _, applications = native_fixture(EntityKind.SURFACE_DEFORMABLE, surface_collision="vis_mesh")
    world._author_deformable(entity)
    assert applications[0].endswith("/vis_mesh")
    assert stage.prims[applications[0]].attributes["physxCollision:contactOffset"] == 0.0005


@pytest.mark.parametrize("kind", [EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE])
@pytest.mark.parametrize("existing", [None, (0.03, 0.002)])
def test_none_offset_preserves_sdk_collision_defaults_without_applying_schema(
    kind: EntityKind,
    existing: tuple[float, float] | None,
) -> None:
    world, entity, stage, configs, applications = native_fixture(kind, contact=None, existing_offsets=existing)
    world._author_deformable(entity)
    assert applications == []
    collider = next(p for p in stage.prims.values() if p.HasAPI("PhysicsCollisionAPI"))
    assert "PhysxCollisionAPI" not in collider.apis
    expected = (
        {}
        if existing is None
        else dict(
            zip(
                ("physxCollision:contactOffset", "physxCollision:restOffset"),
                existing,
                strict=True,
            )
        )
    )
    assert {k: v for k, v in collider.attributes.items() if k.startswith("physxCollision:")} == expected
    assert "contact_offset" not in configs[0] and "rest_offset" not in configs[0]


@pytest.mark.parametrize("kind", [EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE])
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "exactly one collision mesh; found 0"),
        ("ambiguous", "exactly one collision mesh; found 2"),
        ("topology", "does not match its surface/volume topology"),
    ],
)
def test_invalid_collision_geometry_rejects_before_authoring_offsets(
    kind: EntityKind,
    failure: str,
    message: str,
) -> None:
    world, entity, _, _, applications = native_fixture(kind, failure=failure)
    with pytest.raises(ValueError, match=message):
        world._author_deformable(entity)
    assert applications == []
    assert world._deformables == {}


@pytest.mark.parametrize("attribute", ["physxCollision:contactOffset", "physxCollision:restOffset"])
def test_offset_attribute_write_failure_is_explicit(attribute: str) -> None:
    world, entity, stage, _, _ = native_fixture(EntityKind.VOLUME_DEFORMABLE)
    stage.rejected_attribute = attribute
    with pytest.raises(RuntimeError, match="failed to author deformable"):
        world._author_deformable(entity)
    assert world._deformables == {}


def test_unavailable_collision_schema_is_explicit() -> None:
    world, entity, _, _, _ = native_fixture(EntityKind.VOLUME_DEFORMABLE, failure="schema")
    with pytest.raises(RuntimeError, match="failed to apply deformable collision offsets"):
        world._author_deformable(entity)
    assert world._deformables == {}


def test_render_only_profile_never_defines_body_or_collision_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    world, entity, stage, configs, applications = native_fixture(EntityKind.VOLUME_DEFORMABLE)
    world._config.deformable_mode = "render_only"
    rendered: list[EntitySpec] = []
    monkeypatch.setattr(world, "_author_render_deformable", rendered.append)
    world._author_deformable(entity)
    assert rendered == [entity]
    assert configs == applications == []
    assert stage.prims == {}
