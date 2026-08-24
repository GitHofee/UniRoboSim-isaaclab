#!/usr/bin/env python3
"""Public UniRoboSim acceptance for one embedded scene005 door mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from unirobosim import (
    COMPOSITE_WORLD_SCHEMA_VERSION,
    ArrayValue,
    ArticulationCommand,
    BuildInput,
    BuildResourceEntry,
    BuildResourceManifest,
    BuildSourceEntry,
    CapabilityId,
    CapabilityRequirement,
    CommandMode,
    EmbeddedEntityBinding,
    EmbeddedPrimBinding,
    EntityKind,
    EntityPath,
    EntitySpec,
    LocalSourceIdentity,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, IsaacLabProvider

_CONTAINER = EntityPath("/scene")
_DOOR = EntityPath("/scene/bathroom-door")
_FRAME_PRIM = "door/livingDiningRoom1_bathroom1_door_cabinet_006/body"
_DOOR_PRIM = "door/livingDiningRoom1_bathroom1_door_cabinet_006/door_1"
_JOINT_PRIM = f"{_DOOR_PRIM}/RevoluteJoint"


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri)
    raise ValueError(f"inventory resource is not local: {uri}")


def _build_input(inventory: dict[str, Any]) -> tuple[BuildInput, Path]:
    bundle_root = Path(inventory["bundle_root"])
    resources = tuple(inventory["resources"])
    entries: list[BuildResourceEntry] = []
    sources: list[BuildSourceEntry] = []
    selected_asset: Path | None = None
    for resource in sorted(resources, key=lambda item: item["name"]):
        resource_id = str(resource["name"])
        uri = str(resource["uri"])
        path = _local_path(uri)
        stat_result = path.stat()
        selected = bool(resource.get("selected_simulation_input", False))
        if selected:
            if selected_asset is not None:
                raise ValueError("inventory contains more than one selected simulation input")
            selected_asset = path
        entries.append(
            BuildResourceEntry(
                entity_id="scene005",
                component_id="scene005.composite",
                resource_id=resource_id,
                role=str(resource["role"]),
                media_type=str(resource["format"]),
                requested_uri=uri,
                resolved_uri=uri,
                canonical_source_identity=f"sha256:{resource['sha256']}",
                byte_size=int(resource["byte_size"]),
                sha256=str(resource["sha256"]),
                selected_simulation_input=selected,
                purposes=("collision", "simulation", "visual"),
                relative_bundle_path=str(resource["relative_bundle_path"]),
                dependencies=tuple(resource.get("dependencies", ())),
            )
        )
        relative_source_path = path.relative_to(bundle_root).as_posix()
        sources.append(
            BuildSourceEntry(
                resource_id=resource_id,
                source_kind="local-file",
                source_root=str(bundle_root),
                relative_source_path=relative_source_path,
                expected_identity=LocalSourceIdentity(
                    stat_result.st_dev,
                    stat_result.st_ino,
                    stat_result.st_mode,
                    stat_result.st_size,
                    stat_result.st_mtime_ns,
                    stat_result.st_ctime_ns,
                ),
                expected_sha256=str(resource["sha256"]),
            )
        )
    if selected_asset is None:
        raise ValueError("inventory has no selected simulation input")
    manifest = BuildResourceManifest(tuple(entries))
    return BuildInput(manifest=manifest, sources=tuple(sources)), selected_asset


def _world(asset: Path, build_input: BuildInput, *, planning: bool = False) -> WorldSpec:
    scene = EntitySpec(_CONTAINER, EntityKind.COMPOSITE_SCENE, asset_uri=asset.as_uri())
    door = EntitySpec(
        _DOOR,
        EntityKind.ARTICULATION,
        joint_names=("door_hinge",),
        initial_joint_positions=(0.0,),
        joint_position_units=("rad",),
        embedded_binding=EmbeddedEntityBinding(
            container_path=_CONTAINER,
            root_body_prim_path=_DOOR_PRIM,
            link_prims=(
                EmbeddedPrimBinding("frame", _FRAME_PRIM),
                EmbeddedPrimBinding("door", _DOOR_PRIM),
            ),
            joint_prims=(EmbeddedPrimBinding("door_hinge", _JOINT_PRIM),),
        ),
    )
    return WorldSpec(
        "scene005-composite-door-acceptance",
        (scene, door),
        requirements=(CapabilityRequirement(CapabilityId("planning.scene@2")),) if planning else (),
        schema_version=COMPOSITE_WORLD_SCHEMA_VERSION,
        build_resource_manifest_sha256=build_input.manifest.sha256,
    )


def run(inventory_path: Path, *, steps: int, target_rad: float, planning: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    build_input, asset = _build_input(inventory)
    manifest_ready = time.monotonic()
    provider = IsaacLabProvider(
        IsaacLabAdapterConfig(
            headless=True,
            enable_cameras=False,
            render=False,
            render_on_step=False,
            device="cuda:0",
        )
    )
    with provider.open() as session:
        worker_ready = time.monotonic()
        with session.build(_world(asset, build_input, planning=planning), build_input=build_input) as world:
            world_ready = time.monotonic()
            planning_result: dict[str, Any] | None = None
            if planning:
                catalog = world.planning_scene_catalog()
                state = world.planning_scene_state()
                resource_geometries = tuple(item for item in catalog.geometries if item.resource_id is not None)
                if not resource_geometries:
                    raise RuntimeError("planning catalog did not publish any resource-backed collision geometry")
                sample = resource_geometries[0]
                lease = world.resolve_planning_geometry(sample.geometry_id)
                try:
                    sample_size = lease.descriptor.byte_size
                    sample_prefix_sha256 = hashlib.sha256(lease.read(0, min(sample_size, 4096))).hexdigest()
                finally:
                    lease.close()
                planning_result = {
                    "catalog": {
                        "entities": len(catalog.entities),
                        "links": len(catalog.links),
                        "joints": len(catalog.joints),
                        "frames": len(catalog.frames),
                        "geometries": len(catalog.geometries),
                        "representations": dict(
                            sorted(Counter(item.representation.value for item in catalog.geometries).items())
                        ),
                        "resource_geometries": len(resource_geometries),
                        "content_sha256": catalog.content_sha256,
                    },
                    "state": {
                        "entities": len(state.entities),
                        "links": len(state.links),
                        "frames": len(state.frames),
                        "articulations": len(state.articulations),
                        "geometry_transforms": len(state.geometry_transforms),
                        "world_revision": state.world_revision,
                        "transform_revision": state.transform_revision,
                        "geometry_revision": state.geometry_revision,
                    },
                    "sample_resource": {
                        "geometry_id": sample.geometry_id,
                        "representation": sample.representation.value,
                        "byte_size": sample_size,
                        "prefix_sha256": sample_prefix_sha256,
                    },
                }
            handle = world.resolve(_DOOR)
            initial = float(world.read_articulation(handle).joint_positions.rows()[0][0])
            world.apply_articulation_command(
                ArticulationCommand(
                    handle,
                    CommandMode.POSITION,
                    ArrayValue.from_rows(((target_rad,),)),
                    target_units=("rad",),
                )
            )
            world.step(steps)
            moved = float(world.read_articulation(handle).joint_positions.rows()[0][0])
            world.reset()
            reset_once = float(world.read_articulation(handle).joint_positions.rows()[0][0])
            world.step(5)
            world.reset()
            reset_twice = float(world.read_articulation(handle).joint_positions.rows()[0][0])
        world_closed = time.monotonic()
    provider_closed = time.monotonic()
    if abs(moved - initial) < 0.05:
        raise RuntimeError(f"embedded door did not move enough: initial={initial}, moved={moved}")
    if not math.isclose(reset_once, initial, rel_tol=0.0, abs_tol=1.0e-5):
        raise RuntimeError(f"first reset did not restore the door: initial={initial}, reset={reset_once}")
    if not math.isclose(reset_twice, initial, rel_tol=0.0, abs_tol=1.0e-5):
        raise RuntimeError(f"second reset did not restore the door: initial={initial}, reset={reset_twice}")
    return {
        "schema": "unirobosim-isaaclab-composite-acceptance/1",
        "asset": str(asset),
        "manifest_sha256": build_input.manifest.sha256,
        "resource_count": len(build_input.manifest.entries),
        "joint": {
            "initial_rad": initial,
            "target_rad": target_rad,
            "moved_rad": moved,
            "reset_once_rad": reset_once,
            "reset_twice_rad": reset_twice,
        },
        "planning": planning_result,
        "timing_seconds": {
            "manifest": manifest_ready - started,
            "worker_start": worker_ready - manifest_ready,
            "world_build": world_ready - worker_ready,
            "world_close": world_closed - world_ready,
            "provider_close": provider_closed - world_closed,
            "total": provider_closed - started,
        },
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--target-rad", type=float, default=0.55)
    parser.add_argument("--planning", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.inventory, steps=args.steps, target_rad=args.target_rad, planning=args.planning)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n", encoding="utf-8")
    print(payload, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
