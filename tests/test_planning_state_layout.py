from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from unirobosim import EntityPath, PlanningGeometryLocalPose

from unirobosim_isaaclab import native_planning as native


class _Rows:
    def __init__(self, values):
        self.values = values
        self.reads = 0

    def detach(self):
        self.reads += 1
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [row[:] for row in self.values]


def _admission():
    admission = object.__new__(native._PlanningAdmission)
    path = EntityPath("/robot")
    pose_rows = _Rows([[0., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 0., 0., 0.]])
    velocity_rows = _Rows([[0.] * 6, [0.] * 6])
    asset = NS(body_names=("base", "tip"), data=NS(
        body_link_pose_w=NS(torch=[pose_rows, pose_rows]),
        body_link_vel_w=NS(torch=[velocity_rows, velocity_rows]),
    ))
    modules = NS(UsdGeom=NS(XformCache=object), sim_utils=NS(get_current_stage=object))
    admission._world = NS(_m=modules, _spec=NS(environments=NS(count=2)), _step_index=0,
                          _origins_cpu=[(0., 0., 0.), (2., 0., 0.)],
                          _articulations={}, _rigids={path: asset})
    admission._m = modules
    # Duplicate authored names retain the previous first-match behavior;
    # the second joint remains independently selectable through its stable ID.
    joints = tuple(native._JointBinding(
        NS(authored_name="joint", joint_id=f"joint.{i}"), parent, "tip",
        PlanningGeometryLocalPose((offset, 0., 0.)), None,
    ) for i, (parent, offset) in enumerate((("base", .25), ("tip", .75))))
    admission._entities = {path: native._EntityBinding(
        path, "entity.robot", "base", "link.base", {"base": "link.base", "tip": "link.tip"},
        joints, ("/robot", "/robot"), entity_prim_link_name="base")}
    admission._frames = {
        "frame.by_name": native._FrameBinding("frame.by_name", path, "joint", "joint"),
        "frame.by_id": native._FrameBinding("frame.by_id", path, "joint_id", "joint.1"),
    }
    admission._geometries = {
        name: native._GeometryBinding(NS(parent_frame_T_geometry=PlanningGeometryLocalPose((0., 0., 0.))),
                                     path, parent)
        for name, parent in (("geometry.z", "tip"), ("geometry.a", "base"))
    }
    admission._geometry_transform_caches = {}
    admission._prepare_state_reads()
    return admission, path, pose_rows, velocity_rows


def test_layout_keeps_joint_name_first_match_id_selection_and_geometry_order():
    admission, _, _, _ = _admission()
    state = admission.state(0)
    frames = {frame.frame_id: frame.world_pose.position_m[0] for frame in state.frames}
    assert frames["frame.by_name"] == .25
    assert frames["frame.by_id"] == 1.75
    assert tuple(item.geometry_id for item in state.geometry_transforms) == ("geometry.a", "geometry.z")


def test_layout_reads_current_poses_on_same_tick_reset_and_other_environment():
    admission, _, poses, velocities = _admission()
    initial = admission.state(0)
    assert admission.state(0) == initial
    assert poses.reads == velocities.reads == 2
    poses.values[0][0] = .5
    moved = admission.state(0)
    assert moved.step_index == initial.step_index
    assert next(f for f in moved.frames if f.frame_id == "frame.by_name").world_pose.position_m[0] == .75
    assert moved.geometry_transforms[0].world_pose.position_m[0] == .5
    other = admission.state(1)
    assert next(f for f in other.frames if f.frame_id == "frame.by_name").world_pose.position_m[0] == -1.25
    poses.values[0][0] = 0.
    assert admission.state(0) == initial


def test_new_admission_owns_its_layout_and_resolves_new_joint_bindings():
    first, path, _, _ = _admission()
    second, _, _, _ = _admission()
    binding = second._entities[path]
    changed_joint = replace(binding.joint_bindings[0], local_pose=PlanningGeometryLocalPose((.5, 0., 0.)))
    second._entities[path] = replace(binding, joint_bindings=(changed_joint, *binding.joint_bindings[1:]))
    second._prepare_state_reads()
    assert first._joint_frame_bindings is not second._joint_frame_bindings
    first_pose = next(f for f in first.state(0).frames if f.frame_id == "frame.by_name").world_pose.position_m
    second_pose = next(f for f in second.state(0).frames if f.frame_id == "frame.by_name").world_pose.position_m
    assert first_pose == (.25, 0., 0.) and second_pose == (.5, 0., 0.)


def test_constructor_prepares_layout_after_catalog_admission(monkeypatch):
    fixture, _, _, _ = _admission()
    closed = []
    monkeypatch.setattr(native, "PlanningMeshCache", lambda: NS(close=lambda: closed.append(True)))

    def build(admission):
        admission._entities = fixture._entities
        admission._frames = fixture._frames
        admission._geometries = fixture._geometries
        return NS()

    monkeypatch.setattr(native._PlanningAdmission, "_build_catalog", build)
    created = native._PlanningAdmission(fixture._world)
    assert closed == [True]
    assert created.state(0) == fixture.state(0)


def test_unresolvable_joint_frame_is_not_silently_omitted():
    admission, path, _, _ = _admission()
    admission._frames["frame.invalid"] = native._FrameBinding("frame.invalid", path, "joint_id", "missing")
    with pytest.raises(StopIteration):
        admission._prepare_state_reads()
