"""Publish render-only body poses through Isaac Lab's public FrameView API."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


def _inherited_world_scales(prims: Any) -> tuple[tuple[float, float, float], ...]:
    """Read scale before FrameView lazily replaces existing Fabric matrices.

    USD local scale is insufficient: physical bodies commonly inherit a scale
    from an asset container. Match Fabric's row-vector matrix decomposition.
    This bounded traversal runs once per view, never once per recorded frame.
    """
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    result: list[tuple[float, float, float]] = []
    for prim in prims:
        matrix = cache.GetLocalToWorldTransform(prim)
        scale = tuple(math.sqrt(sum(float(matrix[row][col]) ** 2 for col in range(3))) for row in range(3))
        if any(not math.isfinite(value) or value <= 0 for value in scale):
            raise ValueError('render body has an invalid inherited world scale')
        result.append((scale[0], scale[1], scale[2]))
    return tuple(result)


def _path_groups(paths: tuple[str, ...]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if len(set(paths)) != len(paths) or any(not path.startswith('/') for path in paths):
        raise ValueError('render body paths must be unique absolute USD paths')
    depths: dict[int, list[int]] = defaultdict(list)
    for index, path in enumerate(paths):
        depths[len(path.split('/'))].append(index)
    result = []
    for depth, indices in sorted(depths.items()):
        segments = [paths[index].split('/')[1:] for index in indices]
        pattern = '/' + '/'.join(
            '(?:' + '|'.join(re.escape(value) for value in sorted({parts[axis] for parts in segments})) + ')'
            for axis in range(depth - 1)
        )
        result.append((pattern, tuple(indices)))
    return tuple(result)


class RenderTransformPublisher:
    """Cache views per path depth; preserve exact physics-to-USD identity mapping.

    Lab's path matcher operates one path segment at a time, so different-depth
    body paths require separate views. Extra regex matches are never written.
    All gathers and pose writes stay on the native device.
    """

    def __init__(self, paths: tuple[str, ...], *, device: Any, view_factory: Any, torch: Any, warp: Any) -> None:
        self._warp = warp
        self._groups = []
        for pattern, source_indices in _path_groups(paths):
            view = view_factory(pattern, device=str(device), validate_xform_ops=False)
            actual = tuple(view.prim_paths)
            if len(set(actual)) != len(actual):
                raise RuntimeError('render FrameView returned duplicate prim paths')
            lookup = {path: index for index, path in enumerate(actual)}
            if any(paths[index] not in lookup for index in source_indices):
                raise RuntimeError('render FrameView omitted a requested physical body')
            target_indices = tuple(lookup[paths[index]] for index in source_indices)
            # FabricFrameView initializes from USD local scale, dropping parent
            # scale from the existing world matrices in a paused world. With no
            # physics step, that lost scale never repairs itself. Initialize and
            # restore all matched prim scales once, including regex extra matches;
            # actual pose writes below remain limited to requested body paths.
            # USD fallback set_scales has *local* semantics and must not receive
            # these world scales.
            if getattr(view, '_use_fabric', False):
                scales = _inherited_world_scales(view.prims)
                view.set_world_poses()
                view.set_scales(warp.array(scales, dtype=warp.float32, device=str(device)))
            self._groups.append((
                view,
                torch.tensor(source_indices, dtype=torch.long, device=device),
                warp.array(target_indices, dtype=warp.uint32, device=str(device)),
            ))

    def publish(self, poses: Any) -> None:
        for view, source_indices, target_indices in self._groups:
            selected = poses.index_select(0, source_indices)
            positions = self._warp.from_torch(selected[:, :3].contiguous(), dtype=self._warp.vec3f)
            orientations = self._warp.from_torch(selected[:, 3:7].contiguous(), dtype=self._warp.vec4f)
            view.set_world_poses(positions, orientations, target_indices)
