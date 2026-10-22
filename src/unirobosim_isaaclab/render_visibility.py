"""Transactional world-owned USD visibility opinions; never edit source assets."""

from unirobosim import EntityKind, RenderVisibilityOverride, RenderVisibilityReceipt


class RenderVisibilityScope:
    def __init__(self, stage, overrides, roots, *, layer=None):
        from pxr import Usd, UsdGeom, Sdf

        if type(overrides) is not tuple or not 1 <= len(overrides) <= 32:
            raise ValueError("visibility overrides must contain 1..32 entries")
        self.stage = stage
        self.layer = None
        selected = []
        keys = set()
        for request in overrides:
            if type(request) is not RenderVisibilityOverride:
                raise TypeError("visibility requires typed overrides")
            root_path = roots.get((request.entity_path.value, request.environment_index))
            if root_path is None:
                raise ValueError("visibility target is not an available environment")
            path = root_path + "/" + request.relative_prim_path
            if path in keys:
                raise ValueError("duplicate visibility target")
            if any(path.startswith(k + "/") or k.startswith(path + "/") for k in keys):
                raise ValueError("overlapping visibility targets")
            keys.add(path)
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive() or not UsdGeom.Imageable(prim) or prim.IsInstanceProxy():
                raise ValueError("visibility target must be an active non-proxy Imageable")
            imageable = UsdGeom.Imageable(prim)
            attr = imageable.GetVisibilityAttr()
            if (
                attr.GetNumTimeSamples()
                or attr.Get() != request.expected_visibility
                or imageable.ComputeVisibility() != request.expected_visibility
            ):
                raise ValueError("visibility source authored/effective precondition differs")
            ancestor = prim.GetParent()
            while ancestor and not ancestor.IsPseudoRoot():
                if UsdGeom.Imageable(ancestor) and UsdGeom.Imageable(ancestor).GetVisibilityAttr().Get() == "invisible":
                    raise ValueError("visibility ancestor remains hidden")
                ancestor = ancestor.GetParent()
            selected.append((request, path, str(attr.Get()), str(imageable.ComputeVisibility())))
        self.layer = layer if layer is not None else Sdf.Layer.CreateAnonymous("replay-render-visibility.usda")
        session = stage.GetSessionLayer()
        attached = self.layer.identifier in session.subLayerPaths
        if not attached:
            session.subLayerPaths = [self.layer.identifier, *session.subLayerPaths]
        receipts = []
        try:
            with Usd.EditContext(stage, self.layer):
                for request, path, before, effective in selected:
                    imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
                    if not imageable.GetVisibilityAttr().Set(request.visibility):
                        raise RuntimeError("visibility write failed")
                    receipts.append(
                        RenderVisibilityReceipt(
                            request,
                            path,
                            before,
                            effective,
                            str(imageable.GetVisibilityAttr().Get()),
                            str(imageable.ComputeVisibility()),
                        )
                    )
            self.receipts = tuple(receipts)
        except BaseException:
            self.close()
            if not attached:
                # No render has observed a failed transaction.
                session.subLayerPaths = [p for p in session.subLayerPaths if p != self.layer.identifier]
            raise

    def close(self):
        if self.layer is not None:
            # RTX retains native weak references to rendered USD layers. Restore
            # opinions without detaching or releasing the world-owned layer.
            self.layer.Clear()


def apply(world, overrides):
    from .native import _native_name

    if getattr(world, "_render_visibility_scope", None) is not None:
        raise ValueError("render visibility was already applied; reset before a new batch")
    roots = {
        (e.path.value, i): f"/World/env_{i}/{_native_name(e.path)}"
        for e in world._spec.entities
        if e.kind in (EntityKind.STATIC_SCENE, EntityKind.COMPOSITE_SCENE)
        for i in range(world._spec.environments.count)
    }
    from pxr import Sdf

    layer = getattr(world, "_render_visibility_layer", None)
    if layer is None:
        layer = Sdf.Layer.CreateAnonymous("replay-render-visibility.usda")
        world._render_visibility_layer = layer
        # World teardown precedes SimulationApp teardown in the worker.
        world._runtime._retained_visibility_layers.append(layer)
    scope = RenderVisibilityScope(world._m.sim_utils.get_current_stage(), overrides, roots, layer=layer)
    world._render_visibility_scope = scope
    world._invalidate_render()
    return scope.receipts


def clear(world):
    scope = getattr(world, "_render_visibility_scope", None)
    if scope is not None:
        scope.close()
        world._render_visibility_scope = None
