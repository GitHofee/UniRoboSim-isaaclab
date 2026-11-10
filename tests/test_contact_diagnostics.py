import numpy as np
import pytest
from unirobosim import ContactDiagnosticsRequest, EntityPath, KinematicTarget

from unirobosim_isaaclab.contact_diagnostics import NativeContactDiagnostics, _range


class Tensor(np.ndarray):
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(value):
    return np.asarray(value, dtype=float).view(Tensor)


def zeros(shape):
    return np.zeros(shape).view(Tensor)


class Body:
    def get_masses(self):
        return tensor([[0.18]])

    def get_transforms(self):
        return tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def get_coms(self):
        return tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def get_velocities(self):
        return zeros((1, 6))


class Contact:
    def __init__(self):
        self.counts = tensor([[2]])
        self.starts = tensor([[0]])
        self.dts = []

    def get_contact_data(self, dt):
        self.dts.append(dt)
        self.counts[:] = 2
        # Two distinct application points: torque must not use their unweighted mean.
        return (
            tensor([[1.0], [3.0]]) / dt,
            tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            zeros((2, 1)),
            self.counts,
            self.starts,
        )

    def get_friction_data(self, dt):
        self.counts[:] = 1  # Reuse normal count storage, as the SDK does.
        return tensor([[1.0, 0.0, 0.0]]) / dt, tensor([[0.0, 2.0, 0.0]]), self.counts, self.starts

    def get_contact_force_matrix(self, dt):
        return tensor([[[0.0, 0.0, 4.0]]]) / dt

    def get_net_contact_forces(self, dt):
        raise AssertionError("unfiltered native aggregate must not be treated as measured pair forces")


def diagnostic(capacity=2):
    req = ContactDiagnosticsRequest(
        KinematicTarget("cup", EntityPath("/cup")),
        (KinematicTarget("finger", EntityPath("/robot"), "pad"),),
        capacity_substeps=capacity,
    )
    return NativeContactDiagnostics(req, [Body(), Body()], Contact(), (0.0, 0.0, 0.0))


def test_force_dt_application_points_torque_and_sdk_buffer_reuse():
    d = diagnostic()
    d.begin_step()
    d.sample(7, 2, 4, 1 / 240)
    s = d.read(0)[0]
    assert s.physics_dt_s == 1 / 240 and s.world_step == 7 and s.substep == 2
    assert len(s.pairs[0].normal) == 2
    assert s.pairs[0].normal_force_n == pytest.approx((0, 0, 960))
    assert s.pairs[0].friction_force_n == pytest.approx((240, 0, 0))
    assert s.pairs[0].torque_about_subject_com_nm == pytest.approx((720, -240, -480))
    assert d.read(0) == d.read(0)  # Reads are non-destructive.


def test_ring_cursor_loss_is_explicit_and_future_cursor_rejected():
    d = diagnostic()
    for i in range(3):
        d.begin_step()
        d.sample(i + 1, 1, 1, 0.01)
    with pytest.raises(OverflowError):
        d.read(0)
    assert [x.sequence for x in d.read(1)] == [2, 3]
    with pytest.raises(ValueError):
        d.read(4)


def test_full_native_buffer_and_ambiguous_pair_shape_rejected():
    with pytest.raises(OverflowError):
        _range(np.array([[16]]), np.array([[0]]), 16)
    with pytest.raises(ValueError):
        _range(np.array([[1, 1]]), np.array([[0, 1]]), 16)


def test_aggregate_force_mismatch_rejected():
    d = diagnostic()
    d.contact.get_contact_force_matrix = lambda dt: zeros((1, 1, 3))
    with pytest.raises(ValueError, match="aggregate"):
        d.begin_step()
        d.sample(1, 1, 4, 1 / 240)


def test_worker_contact_dispatch_preserves_epoch_cursor_and_rejects_missing_extension():
    from unirobosim_isaaclab.worker import _dispatch

    class Driver:
        def start_contact_diagnostics(self, request):
            assert request is req
            return 7

        def contact_diagnostics(self, epoch, after):
            assert (epoch, after) == (7, 3)
            return 3, ()

        def stop_contact_diagnostics(self):
            self.stopped = True

    req = diagnostic().request
    driver = Driver()
    assert _dispatch(None, driver, ("start_contact_diagnostics", (req,))) == (driver, 7, False)
    assert _dispatch(None, driver, ("contact_diagnostics", (7, 3))) == (driver, (3, ()), False)
    assert _dispatch(None, driver, ("stop_contact_diagnostics", ())) == (driver, None, False)
    assert driver.stopped
    with pytest.raises(RuntimeError, match="unavailable"):
        _dispatch(None, object(), ("start_contact_diagnostics", (req,)))


def test_world_optional_contact_facade_rejects_unsupported_and_preserves_identity(tmp_path):
    from unirobosim import UnsupportedCapabilityError, ValidationError

    from unirobosim_isaaclab import IsaacLabProvider

    from .helpers import FakeNativeRuntime, FakeNativeWorld, available_probe, make_articulation_asset, make_world

    class Driver(FakeNativeWorld):
        def start_contact_diagnostics(self, request):
            self.request = request
            return 9

        def contact_diagnostics(self, epoch, after):
            assert (epoch, after) == (9, 0)
            return 0, ()

        def stop_contact_diagnostics(self):
            self.request = None

    class Runtime(FakeNativeRuntime):
        def build_world(self, spec):
            world = Driver(spec)
            self.worlds.append(world)
            return world

    request = ContactDiagnosticsRequest(
        KinematicTarget("cup", EntityPath("/props/marker")),
        (KinematicTarget("pad", EntityPath("/robots/arm"), "finger"),),
    )
    asset = make_articulation_asset(tmp_path / "arm.usda")
    for runtime, count in ((Runtime(), 1), (Runtime(), 2), (FakeNativeRuntime(), 1)):
        provider = IsaacLabProvider(
            runtime_factory=lambda config, runtime=runtime: runtime, probe_function=available_probe,
        )
        with provider.open() as session:
            world = session.build(make_world(asset, environments=count))
            tick = world.tick
            if count != 1 or type(runtime) is FakeNativeRuntime:
                with pytest.raises(UnsupportedCapabilityError):
                    world.start_contact_diagnostics(request)
                continue
            epoch = world.start_contact_diagnostics(request)
            batch = world.contact_diagnostics(epoch)
            assert (batch.world_id, batch.generation, batch.epoch) == (world.world_id, world.generation, 9)
            assert world.tick == tick
            with pytest.raises(ValidationError):
                world.contact_diagnostics(epoch, True)
            world.stop_contact_diagnostics()
            assert runtime.worlds[-1].request is None


@pytest.mark.parametrize("nested_filters", [False, True])
def test_native_exact_pair_selection_epoch_and_stop_lifecycle(nested_filters):
    from types import SimpleNamespace as NS

    from unirobosim_isaaclab.native import IsaacLabNativeWorld, _native_name

    req = diagnostic().request

    class Prim:
        def __init__(self, path, name):
            self.path, self.name = path, name

        def HasAPI(self, api):
            return True

        def GetPath(self):
            return self.path

        def GetName(self):
            return self.name

    paths = [f"/World/env_0/{_native_name(t.entity_path)}" for t in (req.subject, *req.partners)]
    roots = {paths[0]: [Prim(paths[0], "cup")], paths[1]: [Prim(paths[1]+"/pad", "pad")]}

    class TensorView:
        def create_rigid_body_view(self, path):
            return NS(count=1, prim_paths=[path], get_masses=lambda: tensor([[0.18]]))

        def create_rigid_contact_view(self, path, filter_patterns, max_contact_data_count):
            return NS(sensor_count=1, filter_count=1, sensor_paths=[path],
                      filter_paths=[filter_patterns] if nested_filters else filter_patterns)

    native = object.__new__(IsaacLabNativeWorld)
    native._contact_diagnostics = None
    native._contact_diagnostics_epoch = 0
    native._has_fluid = False
    native._spec = NS(environments=NS(count=1))
    native._rigids = {req.subject.entity_path: object()}
    native._articulations = {req.partners[0].entity_path: object()}
    native._origins_cpu = [(0, 0, 0)]
    native._native_dt = 1 / 240
    native._sim = NS(physics_sim_view=TensorView(), get_physics_dt=lambda: 1 / 240)
    native._m = NS(sim_utils=NS(get_current_stage=lambda: NS(GetPrimAtPath=lambda p: roots[p])),
                   Usd=NS(PrimRange=lambda root: root), UsdPhysics=NS(RigidBodyAPI=object()))
    epoch = native.start_contact_diagnostics(req)
    assert epoch == 1 and native.contact_diagnostics(epoch, 0) == (0, ())
    with pytest.raises(RuntimeError, match="owner"):
        native.start_contact_diagnostics(req)
    native.stop_contact_diagnostics()
    with pytest.raises(ValueError, match="stale"):
        native.contact_diagnostics(epoch, 0)
    assert native.start_contact_diagnostics(req) == 2
    native.stop_contact_diagnostics()
    alias = ContactDiagnosticsRequest(
        req.subject, (KinematicTarget("alias", req.subject.entity_path, "cup"),),
    )
    with pytest.raises(ValueError, match="same physical body"):
        native.start_contact_diagnostics(alias)


def test_one_sensor_view_keeps_two_partner_force_ranges_separate():
    from dataclasses import replace

    class TwoContacts(Contact):
        def get_contact_data(self, dt):
            raw = super().get_contact_data(dt)
            return (*raw[:4], tensor([[1, 1]]), tensor([[0, 1]]))

        def get_friction_data(self, dt):
            return zeros((2, 3)), zeros((2, 3)), tensor([[0, 0]]), tensor([[0, 0]])

        def get_contact_force_matrix(self, dt):
            return tensor([[[0, 0, 1], [0, 0, 3]]]) / dt

    req = diagnostic().request
    req = replace(req, partners=(*req.partners, KinematicTarget("finger2", EntityPath("/robot"), "pad2")))
    d = NativeContactDiagnostics(req, [Body(), Body(), Body()], TwoContacts(), (0, 0, 0))
    d.begin_step()
    d.sample(1, 1, 4, 1 / 240)
    sample = d.read(0)[0]
    assert sample.subject_mass_kg == pytest.approx(0.18)
    assert [p.normal_force_n[2] for p in sample.pairs] == pytest.approx([240, 720])
    assert sum(p.normal_force_n[2] for p in sample.pairs) == pytest.approx(960)


def test_contact_torque_uses_pre_integration_com_and_retains_post_step_state():
    d = diagnostic()
    with pytest.raises(RuntimeError, match="pre-integration"):
        d.sample(1, 1, 4, 1 / 240)
    d.begin_step()
    with pytest.raises(RuntimeError, match="unfinished"):
        d.begin_step()
    d.bodies[0].get_transforms = lambda: tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    d.sample(1, 1, 4, 1 / 240)
    s = d.read(0)[0]
    assert s.subject_com_start_m == (0, 0, 0)
    assert s.subject_com_m == (1, 0, 0)
    assert s.pairs[0].torque_about_subject_com_nm == pytest.approx((720, -240, -480))


def test_static_partner_pose_and_robot_link_preserve_force_evidence():
    from unirobosim import StaticContactTarget

    request = ContactDiagnosticsRequest(
        KinematicTarget("base", EntityPath("/robot"), "base_link"),
        (StaticContactTarget("wall", "geometry.wall123"),),
    )
    pose = (2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0)
    evidence = NativeContactDiagnostics(request, [Body(), pose], Contact(), [1.0, 0.0, 0.0])
    evidence.begin_step()
    evidence.sample(1, 1, 1, 0.01)
    sample = evidence.read(0)[0]
    assert sample.pairs[0].partner_id == "wall"
    assert sample.pairs[0].partner_pose_xyzw == (1.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0)
    assert sample.pairs[0].partner_pose_start_xyzw == sample.pairs[0].partner_pose_xyzw
    assert sample.pairs[0].normal_force_n == (0.0, 0.0, 400.0)


@pytest.mark.parametrize("usd_view", [True, False])
def test_drive_diagnostics_reads_native_targets_in_selected_order_without_setters(usd_view):
    from types import SimpleNamespace

    from unirobosim import ArticulationDiagnosticsRequest

    from unirobosim_isaaclab.native import IsaacLabNativeWorld

    class View:
        def __getattr__(self, name):
            assert name.startswith("get_dof_"), "diagnostics must not mutate physics"
            return lambda: tensor([[10.0, 20.0]])

    path = EntityPath("/robot")
    world = object.__new__(IsaacLabNativeWorld)
    world._spec = SimpleNamespace(environments=SimpleNamespace(count=1))
    world._entity_specs = {path: SimpleNamespace(joint_names=("y", "x"))}
    world._joint_maps = {path: (1, 0)}
    world._usd_articulation_views = {path: View()} if usd_view else {}
    world._articulations = {} if usd_view else {path: SimpleNamespace(root_view=View())}
    result = world.articulation_diagnostics(ArticulationDiagnosticsRequest(path, ("x", "y")))
    assert [d.position_target for d in result] == [10.0, 20.0]
    assert [d.max_force for d in result] == [10.0, 20.0]
    with pytest.raises(ValueError, match="unknown joint"):
        world.articulation_diagnostics(ArticulationDiagnosticsRequest(path, ("missing",)))


def test_static_contact_mapping_uses_admitted_collider_and_rejects_dynamic():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "from tests.test_contact_diagnostics import _check_static_contact_mapping; "
         "_check_static_contact_mapping()"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_static_contact_mapping():
    from types import SimpleNamespace
    from pxr import Usd, UsdGeom, UsdPhysics
    from unirobosim import PlanningGeometryMotionClass
    from unirobosim_isaaclab.native_planning import IsaacLabNativePlanningWorld

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Cube.Define(stage, "/wall").GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    descriptor = SimpleNamespace(motion_class=PlanningGeometryMotionClass.STATIC)
    world = object.__new__(IsaacLabNativePlanningWorld)
    world._m = SimpleNamespace(UsdGeom=UsdGeom, UsdPhysics=UsdPhysics)
    world._planning_admission = SimpleNamespace(
        _geometries={"geometry.wall123": SimpleNamespace(descriptor=descriptor)},
        _contact_prim_paths={"geometry.wall123": "/wall"}, _stage=stage,
    )
    path, pose = world._static_contact_path_pose("geometry.wall123")
    assert path == "/wall" and pose == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="absent"):
        world._static_contact_path_pose("geometry.unknown")
    descriptor.motion_class = PlanningGeometryMotionClass.DYNAMIC
    with pytest.raises(ValueError, match="not static"):
        world._static_contact_path_pose("geometry.wall123")
