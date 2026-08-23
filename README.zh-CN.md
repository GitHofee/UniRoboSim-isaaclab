# UniRoboSim Isaac Lab Adapter

[English](README.md) | [简体中文](README.zh-CN.md)

`unirobosim-isaaclab` 将 UniRoboSim `0.9.x` 接入 Isaac Lab 3.0 / Isaac Sim 6.0.1。导入包和执行 `probe()` 不产生仿真副作用；`open()` 会在 Adapter 自己管理的 worker 进程中启动 Kit，避免 Isaac Sim 接管或终止应用主进程。

## 兼容矩阵

| 项目 | 已验证版本 |
| --- | --- |
| Python | `>=3.12,<3.13` |
| UniRoboSim | `>=0.9.1,<0.10` |
| Isaac Lab profile | `release/3.0.0-beta2`（`isaaclab==6.1.17`） |
| Isaac Lab PhysX | `1.1.3` |
| Isaac Sim | `6.0.1.0` |
| PyTorch | `2.10.0+cu128` |
| Runtime contract | `v0alpha5` |

## 安装

先在独立 Python 3.12 Conda 环境中安装已验证的 NVIDIA Isaac Sim/Isaac Lab 运行栈，并确认 `import isaaclab` 和 `import isaacsim` 成功，再安装 Core、Adapter 和可选 USD 转换器：

```bash
conda create -n unirobosim-isaaclab3 python=3.12 pip -y
conda activate unirobosim-isaaclab3

git clone https://github.com/GitHofee/UniRoboSim.git
git clone https://github.com/GitHofee/UniRoboSim-isaaclab.git
git clone https://github.com/GitHofee/UniRoboSim-usd-converter.git

python -m pip install ./UniRoboSim
python -m pip install ./UniRoboSim-isaaclab
python -m pip install ./UniRoboSim-usd-converter
```

不启动 Kit 检查环境：

```bash
python -c "from unirobosim_isaaclab import create_provider; print(create_provider().probe())"
```

## EasyAPI

安装入口点后只需指定后端：

```python
from unirobosim import Sim

with Sim(backend="isaaclab", world_id="isaac-demo") as sim:
    box = sim.add_box(
        "red_box",
        size_m=0.1,
        color_rgba=(1.0, 0.0, 0.0, 1.0),
        position_m=(0.0, 0.0, 0.5),
    )
    camera = sim.add_camera("camera", resolution=(640, 360), outputs=("rgb", "depth"))
    sim.start()
    sim.step(30)
    print(box.state)
    print(camera.read("rgb").shape)
```

已安装的 `unirobosim.backends/isaaclab` 入口默认使用无头相机/渲染档位。
FastSim 或 EasyAPI 需要真实 Kit 窗口时，通过一个精确的环境变量值显式启用：

```bash
UNIROBOSIM_ISAACLAB_LAUNCH_PROFILE=visible python run_simulation.py
```

完整的公开档位合同如下：

| 值 | 启动行为 |
| --- | --- |
| 未设置 | 无头，相机开启，渲染开启 |
| `headless` | 无头，相机开启，渲染开启 |
| `headless-physics` | 无头，相机关闭，渲染关闭 |
| `visible` | 显示 Kit 窗口，相机开启，渲染开启 |

值区分大小写，其他值会在 Isaac SDK 加载前被拒绝。Adapter 不根据 `DISPLAY`
自动推断档位，因此同一批处理任务在不同机器上的默认行为一致。这个选择器只作用于
正常的已安装入口发现；`create_provider(IsaacLabAdapterConfig(...))` 仍是显式的
程序化配置接口。

需要定制非可移植启动参数时注入 Provider：

```python
from unirobosim import Sim
from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider

provider = create_provider(
    IsaacLabAdapterConfig(
        headless=True,
        device="cuda:0",
        enable_cameras=True,
        render=True,
        anti_aliasing="fxaa",
        texture_streaming=False,
        render_on_step=False,
    )
)

with Sim(provider=provider) as sim:
    sim.add_camera("camera", resolution=(1920, 1080), outputs=("rgb",))
    sim.start()
```

## 已实现原生能力

- USD 机器人及非机器人铰接物体；
- 关节位置、速度和力矩控制；
- 刚体位姿/速度、持续 wrench、聚合接触、reset 和场景位姿写入；
- 三角形表面柔性体和四面体体积柔性体；
- 体积柔性体运动学节点位置控制；
- 固定粒子数 PhysX PBD 流体状态与控制；
- RTX RGB/深度相机；
- Native 点、线、坐标轴、文本、包围盒和轨迹调试覆盖层；
- 场景快照、增量和幂等浏览器拖拽事务；
- 多环境 batch 和可重启进程隔离生命周期。

## 资产与渲染质量

原生刚体 USD 必须且只能包含一个 `UsdPhysics.RigidBodyAPI` prim。安装 `unirobosim-usd-converter` 后，可将纯视觉刚体 USD 规范化为 `isaaclab.dynamic-rigid-usd@1`：派生层引用原始视觉/材质组合，并补充 Z-up 米制单位、质量、惯量、碰撞和物理材质。

刚体规范化器会拒绝铰接体、蒙皮网格和空 Stage。杯子、碗等凹结构需要手工碰撞体或明确的凸分解策略；单个 convex hull 会封住内部空腔。

EasyAPI 默认使用 FXAA 并关闭纹理流式加载，以保持相机的全分辨率纹理。`anti_aliasing` 支持 `off`、`taa`、`fxaa`、`dlss`、`dlaa`。只有在显存收益高于材质清晰度时才应启用 `texture_streaming=True`。

无头相机档位只在读取传感器时渲染，不再每个物理 tick 都渲染。RGB 帧以连续 C-order NHWC 字节跨越 worker 边界；录制器应使用 `camera.read("rgb").to_bytes()`，避免把一帧展开成数百万个 Python 整数。程序化可视运行还可以通过 `max_render_hz` 独立限制视口渲染频率。

## 明确限制

此 Isaac Sim profile 的粒子状态通过 PhysX/USD 读取，而不是公开粒子 Tensor API。刚体 + 流体 + 相机 + Debug 已完成混合验证。桥接刚体的接触力，以及粒子流体与 Tensor 铰接体/柔性体同场景组合，会明确失败，不返回误导性结果。

## Planning Scene 兼容性

Adapter 0.9.3 要求 UniRoboSim Core `>=0.9.1,<0.10`，并提供 `planning.scene@2`。Named planning frame 是由 `name`、`owner_link_name` 和 `source` 声明的物理参考系，不承载 grasp、place、handle 或任务语义。使用过早期 semantic frame-role 草案的应用，需要将这些含义迁移到自己的任务或标注层，在仿真合同中只保留中立的物理 frame 声明；旧声明 schema 会被明确拒绝。

## 验证

```bash
python -m pip install -e '.[dev]'
ruff format --check src tests
ruff check src tests
mypy src
pytest -q
python scripts/native_conformance.py --output result.json
```

发布门禁覆盖无 SDK 源码测试、确定性制品、隔离 wheel 安装和已安装入口发现。
原生验收矩阵覆盖刚体/接触、表面/体积柔性体、机器人与非机器人铰接体、粒子流体、
RGB/深度、Native Debug、Provider 重启、planning-scene catalog/state/delta/resource
读取、多环境以及纯净解释器启动。

## 仓库关系

本包只包含 Isaac Lab Adapter。可移植合同与 EasyAPI 位于 [UniRoboSim Core](https://github.com/GitHofee/UniRoboSim.git)，浏览器和 MCP 接口保持为独立包。
