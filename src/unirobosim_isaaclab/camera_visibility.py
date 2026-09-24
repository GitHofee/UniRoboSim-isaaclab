"""Camera-local mesh exclusion using simultaneous RTX scene partitions.

Only explicitly selected visual meshes are duplicated. Physics APIs never enter
the visual copies; the source collision mesh, transforms and visibility remain
unchanged. No camera is rendered against another camera's temporary visibility.
"""

from __future__ import annotations

import hashlib

_PRIM_PARTITION = "primvars:omni:scenePartition"
_CAMERA_PARTITION = "omni:scenePartition"
_DEFAULT = "unirobosim_camera_default"


def _ancestors(prim):
    while prim:
        yield prim
        prim = prim.GetParent()


def partition_groups(camera_exclusions):
    groups = {(): _DEFAULT}
    cameras = {}
    for camera, paths in camera_exclusions.items():
        key = tuple(sorted(set(paths)))
        if key not in groups:
            groups[key] = "unirobosim_camera_" + hashlib.sha256("\n".join(key).encode()).hexdigest()[:16]
        cameras[camera] = groups[key]
    return groups, cameras


def configure_camera_exclusions(stage, camera_exclusions, *, usd, usd_geom, sdf, settings):
    """Resolve exact mesh paths, then author render-only partition references.

    Existing cameras without exclusions are assigned the default view, which sees
    every source mesh. This runs once after all WorldSpec cameras are authored.
    """
    selected = sorted({path for paths in camera_exclusions.values() for path in paths})
    if not selected:
        return ()
    if len(selected) > 256 or len(camera_exclusions) > 128:
        raise ValueError("camera exclusions exceed bounded render partition limits")
    # Validate the complete request before changing the stage.
    for path in selected:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(usd_geom.Mesh):
            raise ValueError(f"camera render exclusion must select an exact Mesh: {path}")
        for ancestor in _ancestors(prim):
            attr = ancestor.GetAttribute(_PRIM_PARTITION)
            if attr and attr.Get():
                raise ValueError("camera exclusion cannot override an existing scene partition")
        if any(child.IsA(usd_geom.Mesh) for child in usd.PrimRange(prim) if child != prim):
            raise ValueError("camera render exclusion cannot select nested mesh hierarchies")
    cameras = [prim for prim in stage.Traverse() if prim.IsA(usd_geom.Camera)]
    known = {str(prim.GetPath()) for prim in cameras}
    if set(camera_exclusions) - known:
        raise ValueError("camera exclusion references an unauthored camera")
    for camera in cameras:
        attr = camera.GetAttribute(_CAMERA_PARTITION)
        if attr and attr.Get():
            raise ValueError("camera exclusion cannot override an existing camera partition")
    groups, camera_groups = partition_groups(camera_exclusions)
    settings.set("/renderer/scenePartitioning/enabled", True)
    for camera in cameras:
        camera.CreateAttribute(_CAMERA_PARTITION, sdf.ValueTypeNames.Token).Set(
            camera_groups.get(str(camera.GetPath()), _DEFAULT)
        )
    clones = []
    for path in selected:
        source = stage.GetPrimAtPath(path)
        # Imported USD meshes are often instance proxies. De-instance only the
        # enclosing instance needed to author this exact visual mesh partition.
        while source.IsInstanceProxy():
            ancestor = source.GetParent()
            while ancestor and not ancestor.IsInstance():
                ancestor = ancestor.GetParent()
            if not ancestor:
                raise RuntimeError("cannot resolve camera exclusion instance proxy")
            ancestor.SetInstanceable(False)
            source = stage.GetPrimAtPath(path)
        allowed = [token for exclusions, token in groups.items() if path not in exclusions]
        source.CreateAttribute(_PRIM_PARTITION, sdf.ValueTypeNames.Token).Set(allowed[0])
        for token in allowed[1:]:
            clone_path = source.GetPath().GetParentPath().AppendChild(
                "__unirobosim_camera_visual_" + hashlib.sha256((path + token).encode()).hexdigest()[:20]
            )
            if stage.GetPrimAtPath(clone_path):
                raise RuntimeError("camera visual reference path already exists")
            clone = stage.DefinePrim(clone_path)
            clone.GetReferences().AddInternalReference(source.GetPath())
            clone.SetInstanceable(False)
            clone.CreateAttribute(_PRIM_PARTITION, sdf.ValueTypeNames.Token).Set(token)
            for child in usd.PrimRange(clone):
                for schema in tuple(child.GetAppliedSchemas()):
                    if schema.startswith(("Physics", "Physx")):
                        child.RemoveAppliedSchema(schema)
            clones.append(str(clone_path))
    return tuple(clones)
