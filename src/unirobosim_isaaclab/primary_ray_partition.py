"""World-owned static primary visibility using official RTX scene partitions.

Original visual meshes remain the sole secondary-ray contributors. Small sibling
references provide primary appearance to allowed cameras. No per-frame mutation.
"""

from __future__ import annotations

import hashlib
from typing import Any

PARTITION_SETTING = "/renderer/scenePartitioning/enabled"


def concrete_camera_path(prim_path: str, environment: int) -> str:
    """Expand the actual authored CameraCfg path, including mounted body suffix."""
    if not isinstance(prim_path, str) or prim_path.count("/env_.*/") != 1:
        raise ValueError("camera path must contain one authored environment expression")
    if isinstance(environment, bool) or not isinstance(environment, int) or environment < 0:
        raise ValueError("camera environment must be a nonnegative integer")
    return prim_path.replace("/env_.*/", f"/env_{environment}/")


class PrimaryRayPartitionScope:
    def __init__(
        self,
        stage: Any,
        cameras: dict[str, tuple[str, ...]],
        *,
        usd: Any,
        sdf: Any,
        usd_geom: Any,
        usd_physics: Any,
        usd_shade: Any,
        settings: Any,
    ) -> None:
        self.stage = stage
        self.layer = None
        self.closed = False
        if settings.get(PARTITION_SETTING) is not True:
            raise RuntimeError("primary partitions must be enabled before renderer startup")
        if not cameras or len(cameras) > 128:
            raise ValueError("primary partitions require 1..128 managed cameras")
        selected = sorted({path for paths in cameras.values() for path in paths})
        if not selected or len(selected) > 256:
            raise ValueError("primary partitions require 1..256 exact visual meshes")
        for path in cameras:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(usd_geom.Camera):
                raise ValueError(f"primary partition camera does not exist: {path}")
        for path in selected:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(usd_geom.Mesh):
                raise ValueError(f"primary partition requires exact Mesh: {path}")
            if prim.HasAPI(usd_physics.CollisionAPI) or prim.HasAPI(usd_physics.RigidBodyAPI):
                raise ValueError(f"primary partition mesh must be visual only: {path}")
            if list(prim.GetChildren()):
                raise ValueError(f"primary partition requires leaf visual Mesh: {path}")
            if usd_geom.Xformable(prim).GetResetXformStack():
                raise ValueError(f"primary partition does not support reset-xform Mesh: {path}")
        # Existing assignments would alter exact membership semantics silently.
        preexisting = [
            str(p.GetPath())
            for p in usd.PrimRange.Stage(stage, usd.TraverseInstanceProxies())
            if p.GetAttribute("primvars:omni:scenePartition").Get() or p.GetAttribute("omni:scenePartition").Get()
        ]
        if preexisting:
            raise ValueError("scene already has partition assignments: " + repr(preexisting))
        copies_count = sum(path not in hidden for hidden in cameras.values() for path in selected)
        if copies_count > 4096:
            raise ValueError("primary partition reference budget exceeds 4096")
        self.receipt = {
            "policy": "static-primary-ray-partitions",
            "original_meshes": selected,
            "cameras": cameras,
            "copies": [],
            "per_frame_policy_updates": 0,
            "source_secondary_flags_modified": False,
        }
        self.layer = sdf.Layer.CreateAnonymous("primary-ray-partitions.usda")
        session = stage.GetSessionLayer()
        session.subLayerPaths = [self.layer.identifier, *session.subLayerPaths]
        try:
            with usd.EditContext(stage, self.layer):
                for path in selected:
                    prim = stage.GetPrimAtPath(path)
                    while prim.IsInstanceProxy():
                        ancestor = prim.GetParent()
                        while ancestor and not ancestor.IsInstance():
                            ancestor = ancestor.GetParent()
                        if not ancestor:
                            raise RuntimeError("cannot resolve mesh instance proxy: " + path)
                        ancestor.SetInstanceable(False)
                        prim = stage.GetPrimAtPath(path)
                    pv = usd_geom.PrimvarsAPI(prim)
                    if not pv.CreatePrimvar("hideForCamera", sdf.ValueTypeNames.Bool).Set(True):
                        raise RuntimeError("cannot hide original primary mesh: " + path)
                    if prim.GetAttribute("primvars:hideForCamera").Get() is not True:
                        raise RuntimeError("stronger opinion blocks primary hide: " + path)
                for camera, hidden in sorted(cameras.items()):
                    token = "fs_" + hashlib.sha256(camera.encode()).hexdigest()[:16]
                    cam = stage.GetPrimAtPath(camera)
                    cam.CreateAttribute("omni:scenePartition", sdf.ValueTypeNames.Token, custom=True).Set(token)
                    if cam.GetAttribute("omni:scenePartition").Get() != token:
                        raise RuntimeError("stronger opinion blocks camera partition")
                    for path in selected:
                        if path in hidden:
                            continue
                        original = stage.GetPrimAtPath(path)
                        name = "__fs_primary_" + hashlib.sha256((camera + "|" + path).encode()).hexdigest()[:20]
                        target = original.GetPath().GetParentPath().AppendChild(name)
                        if stage.GetPrimAtPath(target):
                            raise ValueError("primary reference target already exists: " + str(target))
                        copy = stage.DefinePrim(target, "Mesh")
                        copy.GetReferences().AddInternalReference(original.GetPath())
                        pv = usd_geom.PrimvarsAPI(copy)
                        for flag, value in [
                            ("hideForCamera", False),
                            ("invisibleToSecondaryRays", True),
                            ("doNotCastShadows", True),
                        ]:
                            pv.CreatePrimvar(flag, sdf.ValueTypeNames.Bool).Set(value)
                            if copy.GetAttribute("primvars:" + flag).Get() is not value:
                                raise RuntimeError("stronger opinion blocks reference ray flag")
                        pv.CreatePrimvar("omni:scenePartition", sdf.ValueTypeNames.Token).Set(token)
                        # References to a leaf do not retain bindings outside its
                        # subtree. Explicitly bind the same material in this stage.
                        mat, _ = usd_shade.MaterialBindingAPI(original).ComputeBoundMaterial()
                        if mat:
                            usd_shade.MaterialBindingAPI.Apply(copy).Bind(mat)
                        if copy.HasAPI(usd_physics.CollisionAPI) or copy.HasAPI(usd_physics.RigidBodyAPI):
                            raise RuntimeError("primary copy unexpectedly carries physics")
                        self.receipt["copies"].append(
                            {
                                "source": path,
                                "copy": str(target),
                                "camera": camera,
                                "material": str(mat.GetPath()) if mat else None,
                            }
                        )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.layer is not None:
            session = self.stage.GetSessionLayer()
            session.subLayerPaths = [x for x in session.subLayerPaths if x != self.layer.identifier]
