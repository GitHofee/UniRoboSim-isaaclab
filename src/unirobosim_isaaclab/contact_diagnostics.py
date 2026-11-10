"""Demand-created contact views and bounded per-substep evidence."""

from __future__ import annotations

from collections import deque
from typing import Any, cast

import numpy as np
from unirobosim import (
    ContactDiagnosticsRequest,
    ContactDiagnosticsSample,
    ContactForcePoint,
    ContactPairEvidence,
)


def _cpu(value: Any) -> np.ndarray:
    return cast(np.ndarray, value.numpy().copy())


def _partner_pose(body: Any) -> np.ndarray:
    if type(body) is tuple:
        return np.asarray(body, dtype=float).copy()
    return cast(np.ndarray, _cpu(body.get_transforms())[0])


def _points(forces: np.ndarray, positions: np.ndarray, origin: np.ndarray) -> tuple[ContactForcePoint, ...]:
    return tuple(
        ContactForcePoint(tuple((p - origin).tolist()), tuple(f))
        for f, p in zip(forces.tolist(), positions, strict=True)
    )


def _range(counts: np.ndarray, starts: np.ndarray, capacity: int, index: int = 0, pairs: int = 1) -> slice:
    if counts.shape != (1, pairs) or starts.shape != (1, pairs):
        raise ValueError("contact view did not preserve the exact selected pair")
    count, start = int(counts[0, index]), int(starts[0, index])
    if count == 0:
        return slice(0, 0)
    if count < 0 or start < 0 or count >= capacity or start + count >= capacity:
        raise OverflowError("contact detail buffer is full or invalid; evidence is incomplete")
    return slice(start, start + count)


class NativeContactDiagnostics:
    """The native driver alone owns the lifecycle; no background polling."""

    def __init__(self, request: ContactDiagnosticsRequest, bodies: list[Any], contact: Any, origin: Any) -> None:
        self.request = request
        self.bodies = bodies
        self.contact = contact
        self.mass_kg = float(_cpu(bodies[0].get_masses()).reshape(-1)[0])
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("contact diagnostics requires a finite positive native mass")
        self.origin = np.asarray(origin, dtype=float)
        self.samples: deque[ContactDiagnosticsSample] = deque(maxlen=request.capacity_substeps)
        self.sequence = 0
        self._retained_points = 0
        self._start: tuple[Any, ...] | None = None

    def _body_state(self, body: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = _cpu(body.get_transforms())[0]
        local = _cpu(body.get_coms())[0, :3]
        q, w = pose[3:6], pose[6]
        uv = np.cross(q, local)
        com = pose[:3] + 2 * (w * uv + np.cross(q, uv)) + local - self.origin
        pose[:3] -= self.origin
        return pose, com, _cpu(body.get_velocities())[0]

    def begin_step(self) -> None:
        if self._start is not None:
            raise RuntimeError("contact diagnostics has an unfinished physics step")
        subject, com, velocity = self._body_state(self.bodies[0])
        partners = []
        for body in self.bodies[1:]:
            pose = _partner_pose(body)
            pose[:3] -= self.origin
            partners.append(pose)
        self._start = subject, com, velocity, partners

    def sample(self, world_step: int, substep: int, substeps: int, dt: float) -> None:
        if self._start is None:
            raise RuntimeError("contact diagnostics requires the pre-integration body state")
        start_pose, start_com, start_velocity, start_partners = self._start
        self._start = None
        subject, com, velocity = self._body_state(self.bodies[0])
        pairs = []
        # One sensor view owns all partner filters; only selected-pair forces are read.
        raw = tuple(_cpu(x) for x in self.contact.get_contact_data(dt))
        raw_f = tuple(_cpu(x) for x in self.contact.get_friction_data(dt))
        matrix = _cpu(self.contact.get_contact_force_matrix(dt))
        count = len(self.request.partners)
        if matrix.shape != (1, count, 3):
            raise ValueError("contact aggregate shape does not match selected partners")
        for index, (partner, body) in enumerate(zip(self.request.partners, self.bodies[1:], strict=True)):
            section = _range(raw[4], raw[5], self.request.max_contact_points, index, count)
            normal = _points(raw[0][section] * raw[2][section], raw[1][section], self.origin)
            section_f = _range(raw_f[2], raw_f[3], self.request.max_contact_points, index, count)
            friction = _points(raw_f[0][section_f], raw_f[1][section_f], self.origin)
            nf = np.sum([p.force_n for p in normal], axis=0) if normal else np.zeros(3)
            ff = np.sum([p.force_n for p in friction], axis=0) if friction else np.zeros(3)
            if not np.allclose(nf, matrix[0, index], rtol=1e-4, atol=1e-3):
                raise ValueError("contact point forces do not reproduce the selected pair aggregate")
            torque = sum(
                (np.cross(np.asarray(p.position_m) - start_com, p.force_n) for p in (*normal, *friction)), np.zeros(3)
            )
            pose = _partner_pose(body)
            pose[:3] -= self.origin
            pairs.append(
                ContactPairEvidence(
                    partner.target_id,
                    tuple(pose.tolist()),
                    normal,
                    friction,
                    tuple(nf.tolist()),
                    tuple(ff.tolist()),
                    tuple(torque.tolist()),
                    tuple(start_partners[index].tolist()),
                )
            )
        self.sequence += 1
        if len(self.samples) == self.samples.maxlen:
            self._retained_points -= sum(len(p.normal)+len(p.friction) for p in self.samples[0].pairs)
        self.samples.append(
            ContactDiagnosticsSample(
                self.sequence,
                world_step,
                substep,
                substeps,
                dt,
                tuple(subject.tolist()),
                tuple(com.tolist()),
                tuple(velocity.tolist()),
                tuple(pairs),
                self.mass_kg,
                tuple(start_pose.tolist()),
                tuple(start_com.tolist()),
                tuple(start_velocity.tolist()),
            )
        )
        self._retained_points += sum(len(p.normal)+len(p.friction) for p in pairs)
        # Bound object-memory as well as sample count. Stale readers detect any eviction.
        while self._retained_points > 262144:
            removed = self.samples.popleft()
            self._retained_points -= sum(len(p.normal)+len(p.friction) for p in removed.pairs)

    def read(self, after_sequence: int) -> tuple[ContactDiagnosticsSample, ...]:
        if type(after_sequence) is not int or not 0 <= after_sequence <= self.sequence:
            raise ValueError("invalid contact evidence cursor")
        if self.samples and after_sequence < self.samples[0].sequence - 1:
            raise OverflowError("contact evidence cursor was overwritten; records were lost")
        return tuple(s for s in self.samples if s.sequence > after_sequence)
