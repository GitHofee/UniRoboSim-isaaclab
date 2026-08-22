"""Setuptools hooks with deterministic wheel and source-distribution metadata."""

from __future__ import annotations

import gzip
import os
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast


def _backend() -> Any:
    from setuptools import build_meta  # type: ignore[import-untyped]

    return build_meta


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    return cast(list[str], _backend().get_requires_for_build_wheel(config_settings))


def get_requires_for_build_sdist(config_settings: dict[str, Any] | None = None) -> list[str]:
    return cast(list[str], _backend().get_requires_for_build_sdist(config_settings))


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None) -> list[str]:
    return cast(list[str], _backend().get_requires_for_build_editable(config_settings))


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return cast(str, _backend().prepare_metadata_for_build_wheel(metadata_directory, config_settings))


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return cast(str, _backend().prepare_metadata_for_build_editable(metadata_directory, config_settings))


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    filename = cast(str, _backend().build_wheel(wheel_directory, config_settings, metadata_directory))
    _rewrite_wheel(Path(wheel_directory, filename))
    return filename


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return cast(str, _backend().build_editable(wheel_directory, config_settings, metadata_directory))


def _source_date_epoch() -> int:
    raw = os.environ.get("SOURCE_DATE_EPOCH", "0")
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise RuntimeError("SOURCE_DATE_EPOCH must be a base-10 integer") from exc
    if not 0 <= value <= 0xFFFFFFFF:
        raise RuntimeError("SOURCE_DATE_EPOCH is outside the gzip timestamp range")
    return value


def _normalized_permissions(mode: int, *, is_directory: bool, is_link: bool = False) -> int:
    if is_link:
        return 0o777
    if is_directory:
        return 0o755
    return 0o755 if mode & 0o111 else 0o644


def _rewrite_wheel(path: Path) -> None:
    members: list[tuple[zipfile.ZipInfo, bytes]] = []
    with zipfile.ZipFile(path, "r") as source:
        archive_comment = source.comment
        for member in source.infolist():
            members.append((member, source.read(member)))

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.comment = archive_comment
            for member, payload in members:
                original_mode = member.external_attr >> 16
                file_type = stat.S_IFMT(original_mode)
                is_directory = member.is_dir()
                is_link = file_type == stat.S_IFLNK
                if is_directory:
                    file_type = stat.S_IFDIR
                elif not is_link:
                    file_type = stat.S_IFREG
                permissions = _normalized_permissions(
                    original_mode,
                    is_directory=is_directory,
                    is_link=is_link,
                )
                member.create_system = 3
                member.external_attr = (file_type | permissions) << 16
                if is_directory:
                    member.external_attr |= 0x10
                archive.writestr(member, payload, compress_type=member.compress_type, compresslevel=9)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rewrite_sdist(path: Path, epoch: int) -> None:
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(path, "r:gz") as source:
        for member in source.getmembers():
            stream = source.extractfile(member) if member.isfile() else None
            payload = None if stream is None else stream.read()
            members.append((member, payload))

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=epoch, compresslevel=9) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for member, payload in sorted(members, key=lambda item: item[0].name):
                        member.uid = 0
                        member.gid = 0
                        member.uname = ""
                        member.gname = ""
                        member.mtime = epoch
                        member.pax_headers = {}
                        member.mode = _normalized_permissions(
                            member.mode,
                            is_directory=member.isdir(),
                            is_link=member.issym() or member.islnk(),
                        )
                        archive.addfile(member, None if payload is None else _BytesReader(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _BytesReader:
    """Minimal file object consumed synchronously by :meth:`TarFile.addfile`."""

    __slots__ = ("_offset", "_payload")

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        start = self._offset
        self._offset = min(len(self._payload), start + size)
        return self._payload[start : self._offset]


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    filename = cast(str, _backend().build_sdist(sdist_directory, config_settings))
    _rewrite_sdist(Path(sdist_directory, filename), _source_date_epoch())
    return filename
