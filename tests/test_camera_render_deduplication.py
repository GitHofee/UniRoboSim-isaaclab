from __future__ import annotations

from types import SimpleNamespace

from unirobosim_isaaclab.native import IsaacLabNativeWorld


class _Simulation:
    def __init__(self) -> None:
        self.render_count = 0
        self.forward_count = 0
        self.step_renders: list[bool] = []

    def render(self) -> None:
        self.render_count += 1

    def forward(self) -> None:
        self.forward_count += 1

    def step(self, *, render: bool) -> None:
        self.step_renders.append(render)


def _render_world() -> tuple[IsaacLabNativeWorld, _Simulation, list[str]]:
    world = object.__new__(IsaacLabNativeWorld)
    simulation = _Simulation()
    mounted_syncs: list[str] = []
    world._sim = simulation
    world._render_revision = 0
    world._rendered_revision = -1
    world._sync_all_mounted_cameras = lambda: mounted_syncs.append("all")
    return world, simulation, mounted_syncs


def test_camera_reads_share_one_global_render_per_world_revision() -> None:
    world, simulation, mounted_syncs = _render_world()

    world._ensure_camera_render()
    world._ensure_camera_render()

    assert simulation.render_count == 1
    assert mounted_syncs == ["all"]

    world._invalidate_render()
    world._ensure_camera_render()
    world._ensure_camera_render()

    assert simulation.render_count == 2
    assert mounted_syncs == ["all", "all"]


def test_visible_physics_step_satisfies_camera_render_for_the_new_revision() -> None:
    world, simulation, mounted_syncs = _render_world()
    world._config = SimpleNamespace(render=True, render_on_step=True)
    world._spec = SimpleNamespace(physics=SimpleNamespace(substeps=1))
    world._render_interval_steps = 1
    world._step_index = 0
    world._native_dt = 1.0 / 60.0
    world._usd_rigid_views = {}
    world._articulations = {}
    world._rigids = {}
    world._deformables = {}
    world._debug_expirations = {}
    world._update_assets = lambda _dt: None

    world.step(1)
    world._ensure_camera_render()

    assert simulation.step_renders == [True]
    assert simulation.render_count == 0
    assert mounted_syncs == ["all"]
    assert world._rendered_revision == world._render_revision


def test_non_rendering_physics_step_defers_exactly_one_render_until_camera_read() -> None:
    world, simulation, mounted_syncs = _render_world()
    world._config = SimpleNamespace(render=False, render_on_step=False)
    world._spec = SimpleNamespace(physics=SimpleNamespace(substeps=1))
    world._render_interval_steps = 1
    world._step_index = 0
    world._native_dt = 1.0 / 60.0
    world._usd_rigid_views = {}
    world._articulations = {}
    world._rigids = {}
    world._deformables = {}
    world._debug_expirations = {}
    world._update_assets = lambda _dt: None

    world.step(1)
    world._ensure_camera_render()
    world._ensure_camera_render()

    assert simulation.step_renders == [False]
    assert simulation.render_count == 1
    assert mounted_syncs == ["all"]


def test_reset_invalidates_a_render_even_when_tick_does_not_change() -> None:
    world, simulation, mounted_syncs = _render_world()
    world._composite_rigid_states = ()
    world._composite_articulation_states = ()
    world._articulations = {}
    world._usd_articulation_views = {}
    world._rigids = {}
    world._contacts = {}
    world._usd_rigid_views = {}
    world._deformables = {}
    world._fluids = {}
    world._cameras = {}
    world._mounted_cameras = {}
    world._debug_lifetimes = {}
    world._debug_expirations = {}
    world._update_assets = lambda _dt: None

    world._mark_rendered()
    world.reset(())
    world._ensure_camera_render()

    assert simulation.forward_count == 1
    assert simulation.render_count == 1
    assert mounted_syncs == ["all", "all"]
