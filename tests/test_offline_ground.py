from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from unirobosim_isaaclab.native import IsaacLabNativeWorld


class _Attribute:
    def __init__(self, values: dict[str, object], name: str) -> None:
        self._values = values
        self._name = name

    def Set(self, value: object) -> None:
        self._values[self._name] = value


class _Plane:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def CreateAxisAttr(self) -> _Attribute:
        return _Attribute(self._values, "axis")

    def CreateWidthAttr(self) -> _Attribute:
        return _Attribute(self._values, "width")

    def CreateLengthAttr(self) -> _Attribute:
        return _Attribute(self._values, "length")

    def CreateDoubleSidedAttr(self) -> _Attribute:
        return _Attribute(self._values, "double_sided")

    def CreateDisplayColorPrimvar(self, interpolation: object) -> _Attribute:
        self._values["color_interpolation"] = interpolation
        return _Attribute(self._values, "color")

    def GetPrim(self) -> _Plane:
        return self


def test_implicit_ground_is_authored_procedurally_without_nucleus_asset() -> None:
    values: dict[str, object] = {}
    plane = _Plane(values)

    class Xform:
        @staticmethod
        def Define(stage: object, path: str) -> None:
            values["xform"] = (stage, path)

    class Plane:
        @staticmethod
        def Define(stage: object, path: str) -> _Plane:
            values["plane"] = (stage, path)
            return plane

    class Collision:
        def CreateCollisionEnabledAttr(self) -> _Attribute:
            return _Attribute(values, "collision_enabled")

    class CollisionAPI:
        @staticmethod
        def Apply(prim: object) -> Collision:
            values["collision_prim"] = prim
            return Collision()

    stage = object()
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(
        sim_utils=SimpleNamespace(get_current_stage=lambda: stage),
        UsdGeom=SimpleNamespace(Xform=Xform, Plane=Plane, Tokens=SimpleNamespace(constant="constant")),
        UsdPhysics=SimpleNamespace(CollisionAPI=CollisionAPI),
        Vt=SimpleNamespace(Vec3fArray=lambda value: tuple(value)),
        Gf=SimpleNamespace(Vec3f=lambda *value: tuple(value)),
    )

    world._author_procedural_ground()

    assert values == {
        "xform": (stage, "/World/unirobosimGround"),
        "plane": (stage, "/World/unirobosimGround/plane"),
        "axis": "Z",
        "width": 100.0,
        "length": 100.0,
        "double_sided": True,
        "color_interpolation": "constant",
        "color": ((0.2, 0.23, 0.28),),
        "collision_prim": plane,
        "collision_enabled": True,
    }

    source = (Path(__file__).resolve().parents[1] / "src/unirobosim_isaaclab/native.py").read_text(encoding="utf-8")
    assert "GroundPlaneCfg" not in source
    assert "ISAAC_NUCLEUS_DIR" not in source
