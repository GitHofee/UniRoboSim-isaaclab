from types import SimpleNamespace

import pytest

from unirobosim_isaaclab.render_transforms import RenderTransformPublisher, _path_groups


def test_depth_patterns_keep_parent_before_child_and_reject_aliases():
    paths = ("/World/env_1/robot/base/arm", "/World/env_0/robot/base", "/World/env_1/robot/base")
    groups = _path_groups(paths)
    assert groups[0][1] == (1, 2)
    assert groups[1][1] == (0,)
    with pytest.raises(ValueError):
        _path_groups(("/x", "/x"))


def test_publisher_maps_actual_view_order_and_does_not_write_extra_matches():
    import torch

    paths = ("/World/env_1/body", "/World/env_0/body")
    calls = []

    def factory(pattern, **kwargs):
        assert kwargs["validate_xform_ops"] is False
        return SimpleNamespace(
            prim_paths=("/World/env_0/body", "/World/env_0/unrelated", "/World/env_1/body"),
            set_world_poses=lambda pos, quat, indices: calls.append((pos.clone(), quat.clone(), indices)),
        )

    warp = SimpleNamespace(
        uint32="uint32",
        vec3f="vec3f",
        vec4f="vec4f",
        array=lambda values, **kwargs: tuple(values),
        from_torch=lambda tensor, **kwargs: tensor,
    )
    publisher = RenderTransformPublisher(paths, device="cpu", view_factory=factory, torch=torch, warp=warp)
    poses = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0]])
    publisher.publish(poses)
    publisher.publish(poses + 1)
    assert calls[0][2] == (2, 0)
    assert torch.equal(calls[0][0], poses[:, :3])
    assert torch.equal(calls[0][1], poses[:, 3:])
    assert torch.equal(calls[1][0], poses[:, :3] + 1)


def test_missing_physical_body_is_rejected_before_any_pose_write():
    import torch

    warp = SimpleNamespace()
    with pytest.raises(RuntimeError, match="omitted"):
        RenderTransformPublisher(
            ("/World/body",),
            device="cpu",
            torch=torch,
            warp=warp,
            view_factory=lambda *args, **kwargs: SimpleNamespace(prim_paths=("/World/other",)),
        )


def test_inherited_world_scales_include_rotated_parent_and_nonuniform_scale():
    from pxr import Gf, Usd, UsdGeom

    from unirobosim_isaaclab.render_transforms import _inherited_world_scales

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/asset")
    parent.AddRotateZOp().Set(30.0)
    parent.AddScaleOp().Set(Gf.Vec3f(1.5, 2.0, 3.0))
    body = UsdGeom.Xform.Define(stage, "/asset/body")
    body.AddScaleOp().Set(Gf.Vec3f(2.0, 3.0, 4.0))
    assert _inherited_world_scales((body.GetPrim(),))[0] == pytest.approx((3.0, 6.0, 12.0))


def test_fabric_initialization_restores_world_scales_once_including_extra_matches(monkeypatch):
    import torch

    import unirobosim_isaaclab.render_transforms as module

    paths = ("/asset/body", "/asset/body/child")
    views = []
    reads = []
    scales = {"/asset/body": (1.5, 2.0, 3.0), "/asset/extra": (4.0, 5.0, 6.0), "/asset/body/child": (3.0, 4.0, 6.0)}

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
                self.scales = [(1.0, 1.0, 1.0) for _ in self.prims]
            else:
                self.writes.append((positions.clone(), orientations.clone(), indices))

        def set_scales(self, values):
            self.restores += 1
            self.scales = list(values)

    def factory(pattern, **kwargs):
        actual = ("/asset/extra", "/asset/body") if not views else ("/asset/body/child",)
        view = View(actual)
        views.append(view)
        return view

    def read(prims):
        reads.append(tuple(prims))
        return tuple(scales[p] for p in prims)

    monkeypatch.setattr(module, "_inherited_world_scales", read)
    warp = SimpleNamespace(
        uint32="uint32",
        float32="float32",
        vec3f="vec3f",
        vec4f="vec4f",
        array=lambda values, **kwargs: tuple(values),
        from_torch=lambda tensor, **kwargs: tensor,
    )
    publisher = RenderTransformPublisher(paths, device="cpu", view_factory=factory, torch=torch, warp=warp)
    poses = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0]])
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
        pytest.fail("USD fallback must not restore inherited scale as local scale")

    monkeypatch.setattr(module, "_inherited_world_scales", unexpected)
    view = SimpleNamespace(
        _use_fabric=False,
        prim_paths=("/body",),
        set_scales=unexpected,
        set_world_poses=lambda *args: None,
    )
    warp = SimpleNamespace(
        uint32="uint32",
        vec3f="vec3f",
        vec4f="vec4f",
        array=lambda values, **kwargs: tuple(values),
        from_torch=lambda tensor, **kwargs: tensor,
    )
    publisher = RenderTransformPublisher(
        ("/body",), device="cpu", view_factory=lambda *args, **kwargs: view, torch=torch, warp=warp
    )
    publisher.publish(torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]))


def test_native_frame_batches_entities_and_rebuilds_when_body_layout_changes(monkeypatch):
    import sys

    import torch
    from unirobosim import EntityPath

    import unirobosim_isaaclab.render_transforms as module
    from unirobosim_isaaclab.native import IsaacLabNativeWorld

    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "isaaclab.sim.views", SimpleNamespace(FrameView=object()))
    publishers = []

    class Publisher:
        def __init__(self, paths, **kwargs):
            self.paths = paths
            self.frames = []
            publishers.append(self)

        def publish(self, poses):
            self.frames.append(poses.clone())

    monkeypatch.setattr(module, "RenderTransformPublisher", Publisher)
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(torch=torch)
    world._render_transform_publishers = {}
    robot, box = EntityPath("/robot"), EntityPath("/box")
    robot_poses = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0]])
    box_poses = torch.tensor([[7.0, 8.0, 9.0, 0.0, 1.0, 0.0, 0.0]])
    world._articulations = {
        robot: SimpleNamespace(
            root_view=SimpleNamespace(link_paths=(("/W/base", "/W/base/arm"),)),
            data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=robot_poses)),
        )
    }
    world._rigids = {
        box: SimpleNamespace(
            root_view=SimpleNamespace(prim_paths=("/W/box",)),
            data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=box_poses)),
        )
    }
    world._publish_render_body_transforms((robot,), (box,))
    world._publish_render_body_transforms((robot,), (box,))
    assert len(publishers) == 1
    assert publishers[0].paths == ("/W/base", "/W/base/arm", "/W/box")
    assert len(publishers[0].frames) == 2
    assert torch.equal(publishers[0].frames[0], torch.cat((robot_poses, box_poses)))
    # A different selected subset is a different mapping, not an accidental reuse.
    world._publish_render_body_transforms((), (box,))
    assert len(publishers) == 2 and publishers[-1].paths == ("/W/box",)
    # The same entity can expose a changed prim layout after a scene rebuild.
    world._rigids[box].root_view.prim_paths = ("/W/box_new",)
    world._publish_render_body_transforms((), (box,))
    assert len(publishers) == 3 and publishers[-1].paths == ("/W/box_new",)
    assert len(world._render_transform_publishers) == 1


def test_fabric_batch_prepares_all_views_before_writes_then_propagates_once(monkeypatch):
    import sys

    import torch

    events = []
    kernel = object()
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils.warp.fabric",
        SimpleNamespace(
            compose_fabric_transformation_matrix_from_warp_arrays=kernel,
        ),
    )
    stage = SimpleNamespace(
        GetFabricId=lambda: SimpleNamespace(id=1),
        GetStageIdAsStageId=lambda: SimpleNamespace(id=2),
    )
    hierarchy = SimpleNamespace(update_world_xforms=lambda: events.append("hierarchy"))
    views = []
    for name in ("parent", "child"):
        views.append(
            SimpleNamespace(
                _use_fabric=True,
                _fabric_initialized=True,
                _fabric_world_matrices=name,
                _fabric_dummy_buffer=object(),
                _view_to_fabric=object(),
                _fabric_device="cpu",
                _fabric_stage=stage,
                _fabric_hierarchy=hierarchy,
                _fabric_usd_sync_done=False,
                _prepare_for_reuse=lambda name=name: events.append("prepare:" + name),
            )
        )
    publisher = object.__new__(RenderTransformPublisher)
    publisher._groups = [(view, torch.tensor([index]), torch.tensor([0])) for index, view in enumerate(views)]
    publisher._warp = SimpleNamespace(
        float32="float32",
        from_torch=lambda value, **kw: value,
        launch=lambda **kw: events.append("write:" + kw["inputs"][0]),
        synchronize=lambda: events.append("sync"),
    )
    poses = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0]])
    assert publisher._publish_fabric_batch(poses)
    assert events == ["prepare:parent", "prepare:child", "write:parent", "write:child", "sync", "hierarchy"]
    assert all(view._fabric_usd_sync_done for view in views)
    events.clear()
    views[1]._fabric_device = "cuda:1"
    assert not publisher._publish_fabric_batch(poses)
    assert events == []  # Unsupported mixed contexts are rejected before any write.
