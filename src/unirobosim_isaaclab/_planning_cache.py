"""Small, process-safe persistent cache for canonical planning mesh payloads."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
from collections.abc import Iterable
from pathlib import Path

MeshComponent = tuple[bytes, int, int]
_MAGIC = b"URSPMC01"
_HEADER = struct.Struct("<8sI")
_COMPONENT = struct.Struct("<III")
_MAX_COMPONENTS = 4096
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _default_path() -> Path:
    configured = os.environ.get("UNIROBOSIM_ISAACLAB_PLANNING_CACHE")
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "unirobosim" / "isaaclab" / "planning-mesh-v1.sqlite3"


def cache_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _encode(components: Iterable[MeshComponent]) -> bytes:
    values = tuple(components)
    if not values or len(values) > _MAX_COMPONENTS:
        raise ValueError("invalid planning mesh component count")
    output = bytearray(_HEADER.pack(_MAGIC, len(values)))
    for payload, vertices, triangles in values:
        if (
            type(payload) is not bytes
            or type(vertices) is not int
            or type(triangles) is not int
            or vertices <= 0
            or triangles <= 0
            or len(payload) != (vertices + triangles) * 12
            or len(payload) > _MAX_PAYLOAD_BYTES
        ):
            raise ValueError("invalid planning mesh component")
        output.extend(_COMPONENT.pack(vertices, triangles, len(payload)))
        output.extend(payload)
    return bytes(output)


def _decode(value: bytes) -> tuple[MeshComponent, ...]:
    if len(value) < _HEADER.size:
        raise ValueError("truncated planning cache payload")
    magic, count = _HEADER.unpack_from(value)
    if magic != _MAGIC or not 0 < count <= _MAX_COMPONENTS:
        raise ValueError("invalid planning cache header")
    cursor = _HEADER.size
    result: list[MeshComponent] = []
    for _ in range(count):
        if cursor + _COMPONENT.size > len(value):
            raise ValueError("truncated planning cache component")
        vertices, triangles, size = _COMPONENT.unpack_from(value, cursor)
        cursor += _COMPONENT.size
        if size > _MAX_PAYLOAD_BYTES or size != (vertices + triangles) * 12 or cursor + size > len(value):
            raise ValueError("invalid planning cache component layout")
        result.append((value[cursor : cursor + size], vertices, triangles))
        cursor += size
    if cursor != len(value):
        raise ValueError("planning cache payload has trailing bytes")
    return tuple(result)


class PlanningMeshCache:
    """Best-effort cache; any storage failure falls back to native cooking."""

    def __init__(self, path: Path | None = None) -> None:
        self._connection: sqlite3.Connection | None = None
        target = _default_path() if path is None else path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(target, timeout=5.0)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS mesh (key TEXT PRIMARY KEY, payload BLOB NOT NULL) WITHOUT ROWID"
            )
            self._connection = connection
        except (OSError, sqlite3.Error):
            self.close()

    def get(self, key: str) -> tuple[MeshComponent, ...] | None:
        if self._connection is None:
            return None
        try:
            row = self._connection.execute("SELECT payload FROM mesh WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            try:
                return _decode(bytes(row[0]))
            except (TypeError, ValueError):
                self._connection.execute("DELETE FROM mesh WHERE key = ?", (key,))
                return None
        except sqlite3.Error:
            return None

    def put(self, key: str, components: tuple[MeshComponent, ...]) -> None:
        if self._connection is None:
            return
        try:
            payload = _encode(components)
            self._connection.execute("INSERT OR REPLACE INTO mesh(key, payload) VALUES (?, ?)", (key, payload))
        except (sqlite3.Error, TypeError, ValueError):
            return

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.commit()
                connection.close()
            except sqlite3.Error:
                pass
