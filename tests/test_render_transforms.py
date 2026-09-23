from types import SimpleNamespace

import pytest

from unirobosim_isaaclab.render_transforms import RenderTransformPublisher, _path_groups


def test_depth_patterns_keep_parent_before_child_and_reject_aliases():
    paths = ('/World/env_1/robot/base/arm', '/World/env_0/robot/base', '/World/env_1/robot/base')
    groups = _path_groups(paths)
    assert groups[0][1] == (1, 2)
    assert groups[1][1] == (0,)
    with pytest.raises(ValueError):
        _path_groups(('/x', '/x'))


def test_publisher_maps_actual_view_order_and_does_not_write_extra_matches():
    import torch

    paths = ('/World/env_1/body', '/World/env_0/body')
    calls = []

    def factory(pattern, **kwargs):
        assert kwargs['validate_xform_ops'] is False
        return SimpleNamespace(
            prim_paths=('/World/env_0/body', '/World/env_0/unrelated', '/World/env_1/body'),
            set_world_poses=lambda pos, quat, indices: calls.append((pos.clone(), quat.clone(), indices)),
        )

    warp = SimpleNamespace(
        uint32='uint32', vec3f='vec3f', vec4f='vec4f',
        array=lambda values, **kwargs: tuple(values),
        from_torch=lambda tensor, **kwargs: tensor,
    )
    publisher = RenderTransformPublisher(paths, device='cpu', view_factory=factory, torch=torch, warp=warp)
    poses = torch.tensor([[1., 2., 3., 0., 0., 0., 1.], [4., 5., 6., 0., 0., 1., 0.]])
    publisher.publish(poses)
    publisher.publish(poses + 1)
    assert calls[0][2] == (2, 0)
    assert torch.equal(calls[0][0], poses[:, :3])
    assert torch.equal(calls[0][1], poses[:, 3:])
    assert torch.equal(calls[1][0], poses[:, :3] + 1)


def test_missing_physical_body_is_rejected_before_any_pose_write():
    import torch

    warp = SimpleNamespace()
    with pytest.raises(RuntimeError, match='omitted'):
        RenderTransformPublisher(
            ('/World/body',), device='cpu', torch=torch, warp=warp,
            view_factory=lambda *args, **kwargs: SimpleNamespace(prim_paths=('/World/other',)),
        )


def test_inherited_world_scales_include_rotated_parent_and_nonuniform_scale():
    from pxr import Gf, Usd, UsdGeom
    from unirobosim_isaaclab.render_transforms import _inherited_world_scales

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, '/asset')
    parent.AddRotateZOp().Set(30.)
    parent.AddScaleOp().Set(Gf.Vec3f(1.5, 2., 3.))
    body = UsdGeom.Xform.Define(stage, '/asset/body')
    body.AddScaleOp().Set(Gf.Vec3f(2., 3., 4.))
    assert _inherited_world_scales((body.GetPrim(),))[0] == pytest.approx((3., 6., 12.))


def test_fabric_initialization_restores_world_scales_once_including_extra_matches(monkeypatch):
    import torch
    import unirobosim_isaaclab.render_transforms as module

    paths = ('/asset/body', '/asset/body/child')
    views = []
    reads = []
    scales = {'/asset/body': (1.5, 2., 3.), '/asset/extra': (4., 5., 6.), '/asset/body/child': (3., 4., 6.)}

    class View:
        _use_fabric = True

        def __init__(self, actual):
            self.prim_paths = actual
            self.prims = actual
            self.scales = [scales[p] for p in actual]
            self.initializations = 0
            self.restores = 0
            self.writes = []

        def set_world_poses(self, positions=None, orientations=None, indices=None):
            if positions is None:
                self.initializations += 1
                self.scales = [(1., 1., 1.) for _ in self.prims]
            else:
                self.writes.append((positions.clone(), orientations.clone(), indices))

        def set_scales(self, values):
            self.restores += 1
            self.scales = list(values)

    def factory(pattern, **kwargs):
        actual = ('/asset/extra', '/asset/body') if not views else ('/asset/body/child',)
        view = View(actual)
        views.append(view)
        return view

    def read(prims):
        reads.append(tuple(prims))
        return tuple(scales[p] for p in prims)

    monkeypatch.setattr(module, '_inherited_world_scales', read)
    warp = SimpleNamespace(
        uint32='uint32', float32='float32', vec3f='vec3f', vec4f='vec4f',
        array=lambda values, **kwargs: tuple(values), from_torch=lambda tensor, **kwargs: tensor,
    )
    publisher = RenderTransformPublisher(paths, device='cpu', view_factory=factory, torch=torch, warp=warp)
    poses = torch.tensor([[1., 2., 3., 0., 0., 0., 1.], [4., 5., 6., 0., 0., 1., 0.]])
    for _ in range(4):
        publisher.publish(poses)
    assert len(reads) == 2
    assert all(v.initializations == v.restores == 1 for v in views)
    assert all(v.scales == [scales[p] for p in v.prims] for v in views)
    assert all(call[2] == (1,) for call in views[0].writes)  # unrelated pose is untouched
    assert all(call[2] == (0,) for call in views[1].writes)
    assert torch.equal(views[0].writes[0][0], poses[:1, :3])
    assert torch.equal(views[1].writes[0][1], poses[1:, 3:])


def test_usd_fallback_never_receives_world_scale(monkeypatch):
    import torch
    import unirobosim_isaaclab.render_transforms as module

    def unexpected(*args):
        pytest.fail('USD fallback must not restore inherited scale as local scale')

    monkeypatch.setattr(module, '_inherited_world_scales', unexpected)
    view = SimpleNamespace(
        _use_fabric=False, prim_paths=('/body',), set_scales=unexpected,
        set_world_poses=lambda *args: None,
    )
    warp = SimpleNamespace(
        uint32='uint32', vec3f='vec3f', vec4f='vec4f',
        array=lambda values, **kwargs: tuple(values), from_torch=lambda tensor, **kwargs: tensor,
    )
    publisher = RenderTransformPublisher(('/body',), device='cpu', view_factory=lambda *args, **kwargs: view,
                                         torch=torch, warp=warp)
    publisher.publish(torch.tensor([[1., 2., 3., 0., 0., 0., 1.]]))
