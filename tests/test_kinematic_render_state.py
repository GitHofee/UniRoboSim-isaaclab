from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from unirobosim import EntityPath

from unirobosim_isaaclab.native_protocols import NativeRenderRigidBodyState, NativeRenderStateFrame

from .test_render_state import _native_world


def _check(raw: bool, kinematic: bool, nonzero: bool, invalid: bool = False) -> None:
    import torch

    world, sim, _, rigid, _, events = _native_world()
    path = EntityPath("/box")
    world._kinematic_rigids[path] = kinematic
    # Render-state restoration must not wake/enable simulation or alter authored mode.
    world._physics_activation = SimpleNamespace(
        pin=lambda *_: (_ for _ in ()).throw(AssertionError("physics activated"))
    )
    world._origins = torch.tensor(((0.0, 0.0, 0.0), (10.0, 20.0, 30.0)))
    world._origins_cpu = ((0.0, 0.0, 0.0), (10.0, 20.0, 30.0))
    calls = []
    if raw:

        class View:
            def get_transforms(self):
                return torch.zeros((2, 7))

            def set_transforms(self, v, ids):
                calls.append(("pose", v.clone(), ids.tolist()))

            def set_velocities(self, v, ids):
                calls.append(("velocity", v.clone(), ids.tolist()))

        world._rigids = {}
        world._usd_rigid_views = {path: View()}
        world._usd_rigid_wrenches = {path: (torch.ones((2, 3)), torch.ones((2, 3)))}
        world._m.omni_physx = SimpleNamespace(
            get_physx_interface=lambda: SimpleNamespace(update_transformations=lambda *_: events.append("usd-publish"))
        )
    velocity = ((0.1, 0.2, 0.3),) if nonzero else ((0.0, 0.0, 0.0),)
    update = NativeRenderRigidBodyState(
        path, ((3.0, 2.0, 1.0),), ((0.0, 0.0, 0.0, 1.0),), velocity, ((0.0, 0.0, 0.0),), (1,)
    )
    if invalid:
        update = replace(update, orientations_xyzw=((0.0, 0.0, 0.0, 2.0),))
    frame = NativeRenderStateFrame((), (update,), ())
    if invalid or kinematic and nonzero:
        with pytest.raises(
            ValueError, match="unit quaternions" if invalid else "nonzero.*kinematic|kinematic.*nonzero"
        ):
            world.apply_render_state(frame)
        assert not calls and rigid.poses is None and rigid.velocities is None
        assert sim.forward_count == sim.step_count == sim.render_count == 0
        return
    world.apply_render_state(frame)
    expected = [[13.0, 22.0, 31.0, 0.0, 0.0, 0.0, 1.0]]
    if raw:
        assert calls[0][0] == "pose" and calls[0][1].tolist() == expected and calls[0][2] == [1]
        assert len(calls) == (1 if kinematic else 2)
        assert world._usd_rigid_wrenches[path][0][0].tolist() == [1.0, 1.0, 1.0]
    else:
        assert rigid.poses.tolist() == expected
        assert (rigid.velocities is None) == kinematic
        if not kinematic:
            assert rigid.velocities.tolist()[0][:3] == pytest.approx(velocity[0])
    assert world._kinematic_rigids[path] == kinematic
    assert sim.forward_count == 1 and sim.step_count == sim.render_count == 0
    assert world._step_index == 9 and world._render_revision == 5
    assert events.count("publish") == events.count("sync") == 1


@pytest.mark.parametrize("raw", [False, True], ids=["high", "usd"])
@pytest.mark.parametrize(
    "kinematic,nonzero,invalid", [(True, False, False), (True, True, False), (False, True, False), (True, False, True)]
)
def test_recorded_rigid_restoration_without_physics(raw, kinematic, nonzero, invalid):
    # Existing adapter tests isolate torch/native imports from lightweight tests.
    expression = (
        "from tests.test_kinematic_render_state import _check; "
        f"_check({raw!r}, {kinematic!r}, {nonzero!r}, {invalid!r})"
    )
    p = subprocess.run([sys.executable, "-c", expression], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
