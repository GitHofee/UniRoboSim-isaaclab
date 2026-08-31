from __future__ import annotations

import gzip
import io
import stat
import tarfile
import tomllib
import zipfile
from pathlib import Path

import build_backend


def _write_nondeterministic_sdist(path: Path, *, timestamp: int) -> None:
    with path.open("wb") as output:
        with gzip.GzipFile(filename=path.name, mode="wb", fileobj=output, mtime=timestamp) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                root = tarfile.TarInfo("unirobosim_isaaclab-0.10.0")
                root.type = tarfile.DIRTYPE
                root.mode = 0o755
                root.mtime = timestamp
                root.uid = 1000
                root.gid = 1000
                root.uname = "builder"
                root.gname = "builder"
                archive.addfile(root)
                payload = b"portable source payload\n"
                member = tarfile.TarInfo("unirobosim_isaaclab-0.10.0/example.txt")
                member.mode = 0o644
                member.mtime = timestamp + 1
                member.uid = 1000
                member.gid = 1000
                member.uname = "builder"
                member.gname = "builder"
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))


def _write_wheel_with_mode(path: Path, mode: int) -> None:
    payload = b"portable wheel payload\n"
    with zipfile.ZipFile(path, "w") as archive:
        member = zipfile.ZipInfo("unirobosim_isaaclab/example.py")
        member.create_system = 3
        member.external_attr = (stat.S_IFREG | mode) << 16
        archive.writestr(member, payload)


def test_sdist_rewriter_normalizes_gzip_and_tar_metadata(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_nondeterministic_sdist(first, timestamp=100)
    _write_nondeterministic_sdist(second, timestamp=200)
    assert first.read_bytes() != second.read_bytes()

    epoch = 1_787_414_947
    build_backend._rewrite_sdist(first, epoch)
    build_backend._rewrite_sdist(second, epoch)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
        assert all(member.mtime == epoch for member in members)
        assert all((member.uid, member.gid, member.uname, member.gname) == (0, 0, "", "") for member in members)
        assert [member.mode for member in members] == [0o755, 0o644]

    first_wheel = tmp_path / "first.whl"
    second_wheel = tmp_path / "second.whl"
    _write_wheel_with_mode(first_wheel, 0o664)
    _write_wheel_with_mode(second_wheel, 0o644)
    assert first_wheel.read_bytes() != second_wheel.read_bytes()
    build_backend._rewrite_wheel(first_wheel)
    build_backend._rewrite_wheel(second_wheel)
    assert first_wheel.read_bytes() == second_wheel.read_bytes()
    with zipfile.ZipFile(first_wheel) as archive:
        assert archive.getinfo("unirobosim_isaaclab/example.py").external_attr >> 16 == stat.S_IFREG | 0o644


def test_source_distribution_manifest_includes_complete_test_support() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = (project_root / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include build_backend.py" in manifest
    assert "recursive-include patches *.patch" in manifest
    assert "recursive-include tests *.py" in manifest
    profile_patch = project_root / "patches" / "isaaclab-6.1.17-runtime-profile.patch"
    patch_text = profile_patch.read_text(encoding="utf-8")
    assert '"torch==2.11.0"' in patch_text
    assert '"torchaudio==2.11.0"' in patch_text
    assert '"torchvision==0.26.0"' in patch_text
    assert '-    "coverage==7.6.1"' in patch_text
    assert (project_root / "tests" / "__init__.py").is_file()
    assert (project_root / "tests" / "helpers.py").is_file()


def test_release_metadata_requires_the_matching_core_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as stream:
        project_file = tomllib.load(stream)
    assert project_file["project"]["version"] == "0.10.9"
    assert project_file["project"]["dependencies"] == ["unirobosim>=0.10.1,<0.11"]
    assert project_file["project"]["optional-dependencies"]["dev"] == [
        "mypy==1.20.2",
        "pytest==8.4.2",
        "ruff==0.16.3",
    ]
    assert project_file["build-system"]["build-backend"] == "build_backend"
    assert project_file["build-system"]["backend-path"] == ["."]
