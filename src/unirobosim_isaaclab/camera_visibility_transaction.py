"""Synchronous per-view visibility transactions for existing render products."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager


@contextmanager
def temporary_mesh_visibility(stage, hidden: tuple[str, ...], *, usd, layer, selected: tuple[str, ...]):
    """Use a world-owned session layer; leave all original opinions unchanged."""
    if layer.identifier not in stage.GetSessionLayer().subLayerPaths:
        raise RuntimeError("visibility layer must remain attached for the world lifetime")
    layer.Clear()
    prims = {path: stage.GetPrimAtPath(path) for path in selected}
    for path, prim in prims.items():
        if not prim or prim.GetTypeName() != "Mesh":
            raise ValueError(f"camera visibility exclusion is not an exact Mesh: {path}")
    original = {path: prim.GetAttribute("visibility").Get() for path, prim in prims.items()}
    try:
        with usd.EditContext(stage, layer):
            # Explicitly author both visible and hidden meshes on every group.
            # Kit 110.1 may retain stale RTX visibility after an opinion removal.
            for path, prim in prims.items():
                value = "invisible" if path in hidden else original[path]
                attribute = prim.GetAttribute("visibility")
                if not attribute.Set(value) or attribute.Get() != value:
                    raise RuntimeError(f"a stronger USD opinion blocks camera exclusion: {path}")
        yield
    finally:
        layer.Clear()


def render_camera_groups(
    exclusions: Mapping[str, tuple[str, ...]],
    products: Mapping[str, Sequence[object]],
    viewports: Sequence[object],
    render: Callable[[], None],
    visibility_scope: Callable[[tuple[str, ...]], AbstractContextManager],
) -> None:
    """Render each enabled product only in its group; always restore update gates.

    The caller freezes simulation state for this synchronous transaction. All
    externally visible viewports update only in the unfiltered group. Transient
    visibility never changes the original authored USD properties or collision.
    """
    if set(exclusions) != set(products):
        raise ValueError("every managed camera requires a render-product binding")
    groups: dict[tuple[str, ...], list[str]] = {(): []}
    for camera, paths in exclusions.items():
        groups.setdefault(tuple(sorted(set(paths))), []).append(camera)
    old_products = {
        camera: tuple(product.get_updates_enabled() for product in bindings) for camera, bindings in products.items()
    }
    old_viewports = tuple(viewport.updates_enabled for viewport in viewports)
    try:
        # Finish with the unfiltered group so even a subsequent external viewport
        # update sees restored visual state on SDKs with delayed opinion removal.
        ordered_groups = [item for item in groups.items() if item[0]] + [((), groups[()])]
        for hidden, cameras in ordered_groups:
            if not cameras and (hidden or not any(old_viewports)):
                continue
            active = set(cameras)
            for camera, bindings in products.items():
                for product, enabled in zip(bindings, old_products[camera], strict=True):
                    product.set_updates_enabled(enabled and camera in active)
            for viewport, enabled in zip(viewports, old_viewports, strict=True):
                viewport.updates_enabled = enabled and not hidden
            with visibility_scope(hidden):
                render()
    finally:
        active_error = sys.exc_info()[1]
        errors = []
        for camera, bindings in products.items():
            for product, enabled in zip(bindings, old_products[camera], strict=True):
                try:
                    product.set_updates_enabled(enabled)
                except Exception as error:
                    errors.append(error)
        for viewport, enabled in zip(viewports, old_viewports, strict=True):
            try:
                viewport.updates_enabled = enabled
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError(f"failed to restore {len(errors)} camera update gates") from (active_error or errors[0])
