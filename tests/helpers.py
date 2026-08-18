from __future__ import annotations

from pathlib import Path
from typing import Any

from unirobosim import (
    ArrayValue,
    DeformableBodySpec,
    DeformableTopology,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    FrozenMap,
    ProbeReport,
    ProviderDescriptor,
    WorldSpec,
)

from unirobosim_isaaclab.native_protocols import Matrix, PointBatch


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

    def reset(self, environment_indices: tuple[int, ...]) -> None:
        if self.reset_error is not None:
            raise self.reset_error
        self.calls.append(("reset", environment_indices))

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
