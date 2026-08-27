# UniRoboSim Isaac Lab Adapter

[English](README.md) | [简体中文](README.zh-CN.md)

`unirobosim-isaaclab` 将 UniRoboSim `0.10.x` 接入 Isaac Lab 3.0 / Isaac Sim 6.0.1。导入包和执行 `probe()` 不产生仿真副作用；`open()` 会在 Adapter 自己管理的 worker 进程中启动 Kit，避免 Isaac Sim 接管或终止应用主进程。

Worker 启动只上报一组固定且严格有序的阶段。pre-Kit 阶段继续使用较短的
fail-fast idle 上限。源码档位在单 worker 120 秒硬上限内使用 90 秒 Kit idle
预算；精确匹配的官方 NGC 档位使用有限的 300 秒预算，因为首次 GPU 启动可能在
不产生 worker 协议进度的情况下编译 RTX pipeline。所有预算都经过校验且最大为
300 秒；热启动不会等待未使用的预算。超时后 Adapter 会记录诊断、完整清理该
worker，并且只重试一次。阶段进度不能延长硬上限；配置、协议和 package
fingerprint 等确定性错误仍然立即失败。

## 兼容矩阵

两个准入档位共同要求 Python `>=3.12,<3.13`、UniRoboSim `>=0.10,<0.11`
和运行时合同 `v0alpha6`：

| 档位 | Isaac Lab | Isaac Sim | Torch 栈 |
| --- | --- | --- | --- |
| `source-isaaclab-3.0.0-beta2` | 源码 release，发行包 `6.1.17`；PhysX `1.1.3` | 发行包 `6.0.1.0` | Torch `2.11.0`、TorchVision `0.26.0`、TorchAudio `2.11.0` |
| `ngc-isaaclab-3.0.0` | 官方 NGC bundle，`VERSION=3.0.0`、发行包 `6.1.11`；PhysX `1.1.3` | 模块 bundle，`VERSION=6.0.1` | Torch `2.10.0`、TorchVision `0.25.0`；不要求 TorchAudio |

NGC 档位不是放宽后的版本区间。它只识别官方 bundle 的精确布局，并检查
Isaac Lab/Isaac Sim 版本文件、顶层模块位置、Adapter 所需的全部 API 模块和
debug-draw 扩展。源码档位仍保留完整的精确发行包门禁。只有这套完整指纹通过后才会
选用更大的 NGC 启动预算，热启动不会因此增加延迟。

## 安装

请在独立的 Python 3.12 Conda 环境中安装 NVIDIA 运行栈。上游 Isaac Lab
6.1.17 错误地把测试工具 `coverage==7.6.1` 声明成运行时依赖，而 Isaac Sim
Kernel 6.0.1.0 要求 `coverage==7.4.4`。本仓库提供了一个针对已验证上游提交的
小型可审查补丁：它会移除 Isaac Lab 的运行时 Coverage 约束，并把安装器统一到
完整的 Torch 2.11 档位。

```bash
conda create -n unirobosim-isaaclab3 python=3.12 pip -y
conda activate unirobosim-isaaclab3

git clone https://github.com/GitHofee/UniRoboSim-isaaclab.git
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout 2e44ddb2e19536579140496023b5ccb060bc4152
git -C IsaacLab apply ../UniRoboSim-isaaclab/patches/isaaclab-6.1.17-runtime-profile.patch

python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com
python -m pip install -e ./IsaacLab/source/isaaclab
python -m pip install -e ./IsaacLab/source/isaaclab_physx

git clone https://github.com/GitHofee/UniRoboSim.git
git clone https://github.com/GitHofee/UniRoboSim-usd-converter.git

python -m pip install ./UniRoboSim
python -m pip install ./UniRoboSim-isaaclab
python -m pip install ./UniRoboSim-usd-converter
```

不启动 Kit，检查已安装的依赖图和完整运行档位：

```bash
python -m pip check
python -c "from unirobosim_isaaclab import create_provider; print(create_provider().probe())"
```

上面的 CUDA 12.8 命令是已验证的 Linux x86-64 源码档位。公开版本相同且本地
CUDA 标签兼容的 Wheel 也可接受。Adapter 会有意将大型 NVIDIA 和 Torch 包排除在
自身 Wheel 元数据之外。无副作用 Probe 只会选择一个完整支持档位：源码档位读取
发行包元数据；NGC 档位还会读取官方 bundle 的模块 spec、版本文件、必需模块树和
扩展布局，但不会导入或启动 Kit。两个完整 fingerprint 都不匹配时会 fail closed。
源码档位的 `pip check` 必须输出 `No broken requirements found`。

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
不传配置调用 `create_provider()` 时，会与已安装入口一样，根据完整兼容档位自动选择
启动预算；一旦显式传入配置，其中的两个 worker 启动预算就是权威值。
FastSim 会把 Plan 中的规范值直接传给同一个入口工厂，例如
`create_easy_provider(launch_profile="headless-physics")`。显式关键字不会读取
`UNIROBOSIM_ISAACLAB_LAUNCH_PROFILE`；环境变量只保留给零参数 EasyAPI 兼容路径。

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
        worker_startup_hard_timeout_s=120,
        worker_kit_launch_idle_timeout_s=90,
    )
)

with Sim(provider=provider) as sim:
    sim.add_camera("camera", resolution=(1920, 1080), outputs=("rgb",))
    sim.start()
```

## 已实现原生能力

- USD 机器人及非机器人铰接物体；
- 每个环境只组合一次的复合 USD 场景，并把显式声明的内嵌刚体和铰接体绑定到
  已有 Prim，不重复生成实体；
- 不经刚体或接触传感器包装的不可变静态场景 USD 组合；
- 刚体、均匀缩放铰接体与静态场景资产的物理缩放；
- 关节位置、速度和力矩控制；
- 刚体位姿/速度、持续 wrench、聚合接触、reset 和场景位姿写入；
- 三角形表面柔性体和四面体体积柔性体；
- 体积柔性体运动学节点位置控制；
- 固定粒子数 PhysX PBD 流体状态与控制；
- RTX RGB/深度/法线相机，以及 articulation root 和具名 link 挂载；
- Native 点、线、坐标轴、文本、包围盒和轨迹调试覆盖层；
- 场景快照、增量和幂等浏览器拖拽事务；
- 多环境 batch 和可重启进程隔离生命周期。

## 资产与渲染质量

原生刚体 USD 必须且只能包含一个 `UsdPhysics.RigidBodyAPI` prim。安装 `unirobosim-usd-converter` 后，可将纯视觉刚体 USD 规范化为 `isaaclab.dynamic-rigid-usd@1`：派生层引用原始视觉/材质组合，并补充 Z-up 米制单位、质量、惯量、碰撞和物理材质。

刚体规范化器会拒绝铰接体、蒙皮网格和空 Stage。杯子、碗等凹结构需要手工碰撞体或明确的凸分解策略；单个 convex hull 会封住内部空腔。

EasyAPI 默认使用 FXAA 并关闭纹理流式加载，以保持相机的全分辨率纹理。`anti_aliasing` 支持 `off`、`taa`、`fxaa`、`dlss`、`dlaa`。只有在显存收益高于材质清晰度时才应启用 `texture_streaming=True`。

无头相机档位只在读取传感器时渲染，不再每个物理 tick 都渲染。同一 world revision 的所有相机读取共享一次全局 RTX 渲染；可视运行若已在物理步完成渲染，也会直接复用。RGB 帧以连续 C-order NHWC 字节跨越 worker 边界；录制器应使用 `camera.read("rgb").to_bytes()`，避免把一帧展开成数百万个 Python 整数。程序化可视运行还可以通过 `max_render_hz` 独立限制视口渲染频率。

## 明确限制

此 Isaac Sim profile 的粒子状态通过 PhysX/USD 读取，而不是公开粒子 Tensor API。刚体 + 流体 + 相机 + Debug 已完成混合验证。桥接刚体的接触力，以及粒子流体与 Tensor 铰接体/柔性体同场景组合，会明确失败，不返回误导性结果。

## Planning Scene 兼容性

Adapter 0.10.4 要求 UniRoboSim Core `>=0.10,<0.11`，并提供 `planning.scene@2`。Named planning frame 是由 `name`、`owner_link_name` 和 `source` 声明的物理参考系，不承载 grasp、place、handle 或任务语义。使用过早期 semantic frame-role 草案的应用，需要将这些含义迁移到自己的任务或标注层，在仿真合同中只保留中立的物理 frame 声明；旧声明 schema 会被明确拒绝。PhysX `convexHull` 碰撞 carrier 既可以自身就是 Mesh，也可以是仅含一个且没有嵌套 `PhysicsCollisionAPI` 的后代 Mesh 的容器；多 Mesh 歧义和嵌套碰撞 carrier 仍会 fail closed。

静态和复合 USD 场景会提供不可变 planning resource 与实时 world transform。
少于三个顶点的退化 face 会被跳过；非法索引、不一致的方向数据，以及没有任何
有效三角面的 mesh 会 fail closed，不会向规划器返回不完整几何。

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
RGB/深度/法线、挂载相机、静态 USD 场景、Native Debug、Provider 重启、planning-scene catalog/state/delta/resource
读取、多环境以及纯净解释器启动。

## 仓库关系

本包只包含 Isaac Lab Adapter。可移植合同与 EasyAPI 位于 [UniRoboSim Core](https://github.com/GitHofee/UniRoboSim.git)，浏览器和 MCP 接口保持为独立包。
