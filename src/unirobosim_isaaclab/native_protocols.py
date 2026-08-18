"""Narrow SDK-independent seam between public runtime and Isaac-native implementation."""

from __future__ import annotations

from typing import Protocol

from unirobosim import CommandMode, EntityPath, WorldSpec

Matrix = tuple[tuple[float, ...], ...]
Vector3 = tuple[float, float, float]
PointBatch = tuple[tuple[Vector3, ...], ...]


class NativeWorldDriver(Protocol):
    def reset(self, environment_indices: tuple[int, ...]) -> None: ...

    def apply_articulation(
        self,
        path: EntityPath,
        mode: CommandMode,
        targets: Matrix,
        environment_indices: tuple[int, ...],
        degree_of_freedom_indices: tuple[int, ...],
    ) -> None: ...

    def read_articulation(self, path: EntityPath) -> tuple[Matrix, Matrix]: ...

    def apply_rigid_body_wrench(
        self,
        path: EntityPath,
        forces_n: Matrix,
        torques_n_m: Matrix,
        environment_indices: tuple[int, ...],
    ) -> None: ...

    def read_rigid_body(self, path: EntityPath) -> tuple[Matrix, Matrix, Matrix, Matrix]: ...

    def read_contact(self, path: EntityPath) -> Matrix: ...

    def apply_deformable_position(
        self,
        path: EntityPath,
        targets: PointBatch,
        environment_indices: tuple[int, ...],
        point_indices: tuple[int, ...],
    ) -> None: ...

    def read_deformable(self, path: EntityPath) -> tuple[PointBatch, PointBatch]: ...

    def step(self, count: int) -> None: ...

    def close(self) -> None: ...


class NativeRuntime(Protocol):
    def build_world(self, spec: WorldSpec) -> NativeWorldDriver: ...

    def close(self) -> None: ...
