from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from unirobosim import CommandMode, EntityPath

from unirobosim_isaaclab import IsaacLabAdapterConfig
from unirobosim_isaaclab.native import IsaacLabNativeWorld


class _Tensor:
    def __init__(self, rows: Any) -> None:
        self.rows = [list(row) for row in rows]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, list):
            return _Tensor([self.rows[index] for index in key])
        if isinstance(key, tuple):
            row_key, column_key = key
            if isinstance(row_key, slice) and isinstance(column_key, list):
                return _Tensor([[row[index] for index in column_key] for row in self.rows[row_key]])
            return self.rows[row_key][column_key]
        return self.rows[key]

    def __le__(self, value: float) -> _Tensor:
        return _Tensor([[item <= value for item in row] for row in self.rows])

    def masked_fill(self, mask: _Tensor, value: float) -> _Tensor:
        return _Tensor(
            [
                [value if enabled else item for item, enabled in zip(row, mask_row, strict=True)]
                for row, mask_row in zip(self.rows, mask.rows, strict=True)
            ]
        )


class _Torch:
    float32 = object()

    @staticmethod
    def tensor(rows: Any, **_: Any) -> _Tensor:
        return _Tensor(rows)

    @staticmethod
    def zeros_like(tensor: _Tensor) -> _Tensor:
        return _Tensor([[0.0 for _ in row] for row in tensor.rows])

    @staticmethod
    def full_like(tensor: _Tensor, value: float) -> _Tensor:
        return _Tensor([[value for _ in row] for row in tensor.rows])


class _Asset:
    def __init__(self) -> None:
        self.stiffness_writes: list[list[list[float]]] = []
        self.damping_writes: list[list[list[float]]] = []

    def set_joint_position_target_index(self, **_: Any) -> None:
        pass

    def set_joint_velocity_target_index(self, **_: Any) -> None:
        pass

    def set_joint_effort_target_index(self, **_: Any) -> None:
        pass

    def write_joint_stiffness_to_sim_index(self, *, stiffness: _Tensor, **_: Any) -> None:
        self.stiffness_writes.append(stiffness.rows)

    def write_joint_damping_to_sim_index(self, *, damping: _Tensor, **_: Any) -> None:
        self.damping_writes.append(damping.rows)

    def write_data_to_sim(self) -> None:
        pass


class _DeviceIndices:
    def __init__(self, values: Any, device: str) -> None:
        self.values = tuple(values)
        self.device = device


class _DeviceTensor:
    def __init__(self, name: str, device: str) -> None:
        self.name = name
        self.device = device

    def __getitem__(self, key: Any) -> _DeviceTensor:
        if isinstance(key, _DeviceIndices) and key.device != self.device:
            raise RuntimeError(f"indices on {key.device} cannot index tensor on {self.device}")
        return _DeviceTensor(f"{self.name}[selected]", self.device)

    def __setitem__(self, _key: Any, _value: Any) -> None:
        pass

    def clone(self) -> _DeviceTensor:
        return _DeviceTensor(f"{self.name}.clone", self.device)


class _DeviceTorch:
    int64 = object()

    @staticmethod
    def tensor(values: Any, *, device: str, dtype: Any) -> _DeviceIndices:
        del dtype
        return _DeviceIndices(values, device)

    @staticmethod
    def arange(stop: int, *, device: str, dtype: Any) -> _DeviceIndices:
        del dtype
        return _DeviceIndices(range(stop), device)

    @staticmethod
    def zeros_like(tensor: _DeviceTensor) -> _DeviceTensor:
        return _DeviceTensor(f"zeros({tensor.name})", tensor.device)


class _UsdArticulationView:
    def __init__(self) -> None:
        self.index_devices: dict[str, str] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("set_"):

            def setter(_values: Any, indices: _DeviceIndices) -> None:
                if isinstance(_values, _DeviceTensor) and _values.device != indices.device:
                    raise RuntimeError(f"indices on {indices.device} cannot select values on {_values.device}")
                self.index_devices[name] = indices.device

            return setter
        raise AttributeError(name)


class _Simulation:
    def forward(self) -> None:
        pass


class _PrimPath:
    def __init__(self, path: str) -> None:
        self.pathString = path


class _Prim:
    def __init__(self, path: str) -> None:
        self._path = _PrimPath(path)

    def GetPath(self) -> _PrimPath:
        return self._path


class _InitializationView(_UsdArticulationView):
    def __init__(self, prim_path: str) -> None:
        super().__init__()
        self.count = 1
        self.prim_paths = (prim_path,)
        self.shared_metatype = SimpleNamespace(dof_names=("joint",))

    def get_root_transforms(self) -> _DeviceTensor:
        return _DeviceTensor("root", "cuda:0")

    def get_root_velocities(self) -> _DeviceTensor:
        return _DeviceTensor("root_velocity", "cuda:0")

    def get_dof_positions(self) -> _DeviceTensor:
        return _DeviceTensor("position", "cuda:0")

    def get_dof_stiffnesses(self) -> _DeviceTensor:
        return _DeviceTensor("stiffness", "cpu")

    def get_dof_dampings(self) -> _DeviceTensor:
        return _DeviceTensor("damping", "cpu")


def _world(config: IsaacLabAdapterConfig) -> tuple[IsaacLabNativeWorld, EntityPath, _Asset]:
    path = EntityPath("/robot")
    asset = _Asset()
    world = object.__new__(IsaacLabNativeWorld)
    world._config = config
    world._m = SimpleNamespace(torch=_Torch)
    world._sim = SimpleNamespace(device="cpu")
    world._articulations = {path: asset}
    world._usd_articulation_views = {}
    world._joint_maps = {path: (0, 1, 2, 3)}
    world._articulation_control_modes = {path: [[None, None, None, None]]}
    world._initial_articulation_gains = {
        path: (
            _Tensor(((80_000.0, 240_000.0, 0.0, 1_000.0),)),
            _Tensor(((2_000.0, 3_000.0, 10.0, 20.0),)),
        )
    }
    return world, path, asset


def test_mode_switch_restores_heterogeneous_authored_gains_and_position_fallback() -> None:
    world, path, asset = _world(IsaacLabAdapterConfig())

    for mode in (CommandMode.VELOCITY, CommandMode.EFFORT):
        world.apply_articulation(path, mode, ((0.1, 0.2),), (0,), (0, 2))
        world.apply_articulation(path, CommandMode.POSITION, ((0.3, 0.4),), (0,), (0, 2))

    assert asset.stiffness_writes == [
        [[0.0, 0.0]],
        [[80_000.0, 1_000.0]],
        [[0.0, 0.0]],
        [[80_000.0, 1_000.0]],
    ]
    assert asset.damping_writes == [
        [[100.0, 100.0]],
        [[2_000.0, 100.0]],
        [[0.0, 0.0]],
        [[2_000.0, 100.0]],
    ]


def test_explicit_numeric_position_override_does_not_receive_zero_fallback() -> None:
    world, path, asset = _world(IsaacLabAdapterConfig(position_stiffness=0.0, position_damping=0.0))
    world._initial_articulation_gains[path] = (_Tensor(((0.0, 0.0, 0.0, 0.0),)),) * 2

    world.apply_articulation(path, CommandMode.POSITION, ((0.3,),), (0,), (2,))

    assert asset.stiffness_writes == [[[0.0]]]
    assert asset.damping_writes == [[[0.0]]]


def test_repeated_same_mode_command_does_not_rewrite_unchanged_gains() -> None:
    world, path, asset = _world(IsaacLabAdapterConfig())

    world.apply_articulation(path, CommandMode.POSITION, ((0.1, 0.2),), (0,), (0, 2))
    world.apply_articulation(path, CommandMode.POSITION, ((0.3, 0.4),), (0,), (0, 2))

    assert asset.stiffness_writes == [[[80_000.0, 1_000.0]]]
    assert asset.damping_writes == [[[2_000.0, 100.0]]]


def test_usd_articulation_reset_indexes_authored_gains_on_their_own_device() -> None:
    """Raw PhysX gain tensors can be CPU-resident while articulation state is CUDA."""

    path = EntityPath("/door")
    view = _UsdArticulationView()
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(torch=_DeviceTorch)
    world._sim = _Simulation()
    world._composite_rigid_states = ()
    world._composite_articulation_states = ()
    world._articulations = {}
    world._usd_articulation_views = {path: view}
    world._initial_usd_articulation = {
        path: tuple(_DeviceTensor(name, "cuda:0") for name in ("root", "root_velocity", "position", "velocity"))
    }
    world._initial_usd_articulation_gains = {path: (_DeviceTensor("stiffness", "cpu"), _DeviceTensor("damping", "cpu"))}
    world._rigids = {}
    world._contacts = {}
    world._usd_rigid_views = {}
    world._deformables = {}
    world._fluids = {}
    world._cameras = {}
    world._mounted_cameras = {}
    world._debug_lifetimes = {}
    world._update_assets = lambda _dt: None
    world._sync_all_mounted_cameras = lambda: None

    world.reset((0,))

    assert view.index_devices["set_dof_positions"] == "cuda:0"
    assert view.index_devices["set_dof_stiffnesses"] == "cpu"
    assert view.index_devices["set_dof_dampings"] == "cpu"


def test_usd_articulation_numeric_gain_override_uses_gain_tensor_devices() -> None:
    """Legacy numeric overrides must not reuse CUDA state indices for CPU PhysX gains."""

    path = EntityPath("/door")
    prim_path = "/World/env_0/door"
    view = _InitializationView(prim_path)
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(torch=_DeviceTorch)
    world._config = IsaacLabAdapterConfig(position_stiffness=2_000.0, position_damping=200.0)
    world._spec = SimpleNamespace(
        environments=SimpleNamespace(count=1),
        entities=(
            SimpleNamespace(
                path=path,
                joint_names=("joint",),
                initial_joint_positions=(0.0,),
            ),
        ),
    )
    world._usd_articulations = {path: (SimpleNamespace(root_prim=_Prim(prim_path)),)}
    world._embedded_joint_paths = {}
    world._joint_maps = {}
    world._initial_usd_articulation = {}
    world._initial_usd_articulation_gains = {}
    world._usd_articulation_views = {}
    world._usd_simulation_view = lambda: SimpleNamespace(create_articulation_view=lambda _paths: view)

    world._initialize_usd_articulations()

    assert view.index_devices["set_dof_stiffnesses"] == "cpu"
    assert view.index_devices["set_dof_dampings"] == "cpu"
    stiffness, damping = world._initial_usd_articulation_gains[path]
    assert stiffness.device == "cpu"
    assert damping.device == "cpu"
