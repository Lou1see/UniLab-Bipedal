# 导入自定义机器人 XML 训练 SAC 教程

> 从一个 MuJoCo XML 机器人模型，到在 UniLab 框架中用 SAC 训练行走策略的完整接入流程。
> 以 `RLbipedal`（my_bipedal，6 执行器双足弹簧腿）为实际案例。

---

## 0. 总览：需要修改/创建的文件清单

| 序号 | 文件 | 作用 | 必须创建 |
|:---:|------|------|:--------:|
| 1 | `src/unilab/assets/robots/<RobotName>/robot.xml` | 机器人本体（body/joint/actuator/sensor） | ✅ |
| 2 | `src/unilab/assets/robots/<RobotName>/scene_flat.xml` | 场景（地面/contact sensor/keyframe） | ✅ |
| 3 | `src/unilab/assets/robots/<RobotName>/assets/*.STL` | mesh 文件 | ✅ |
| 4 | `src/unilab/envs/locomotion/<robot>/__init__.py` | 模块入口 | ✅ |
| 5 | `src/unilab/envs/locomotion/<robot>/joystick.py` | env 实现 + reward + obs + 注册 | ✅ |
| 6 | `conf/offpolicy/task/sac/<task_name>/<backend>.yaml` | task 配置（reward scale / warm_start / log_std 等） | ✅ |
| 7 | `conf/offpolicy/algo/sac.yaml` | 算法默认配置（一般不改，用 task YAML 覆盖） | 已有 |

---

## 1. XML 文件准备

### 1.1 目录结构

```
src/unilab/assets/robots/RLbipedal/
├── assets/                    # mesh 文件
│   ├── BaseLink.STL
│   ├── LHipLink.STL
│   └── ... (共 21 个 STL)
├── mujocoBipedalenv.xml       # robot.xml（机器人本体）
└── scene_flat.xml             # scene.xml（场景 + keyframe）
```

### 1.2 robot.xml 结构（`mujocoBipedalenv.xml`）

robot.xml 是**纯机器人描述**，不包含任何场景元素。

```xml
<mujoco model="URDFV2_cylinder_collision">
  <compiler angle="radian" meshdir="assets"/>

  <!-- 1. mesh 声明 -->
  <asset>
    <mesh name="BaseLink" file="BaseLink.STL"/>
    <mesh name="LFootLink" file="LFootLink.STL"/>
    <!-- ... 所有 STL ... -->
  </asset>

  <!-- 2. body 层级（根 body 含 freejoint） -->
  <worldbody>
    <body name="BaseLink" pos="0 0 0.7">          <!-- 初始高度 0.7m -->
      <inertial pos="..." mass="1.756" diaginertia="..."/>
      <geom type="mesh" mesh="BaseLink" .../>       <!-- visual -->
      <geom type="box" .../>                        <!-- collision -->
      <freejoint name="floating_base_joint"/>       <!-- floating base -->
      <site name="imu" pos="0 0 -0.05"/>            <!-- IMU site -->

      <body name="LHipLink" pos="..." quat="...">
        <joint name="LHipJoint" type="hinge" axis="0 0 -1"
               range="-0.7854 0.7854" actuatorfrcrange="-27 27" armature="0.01"/>
        <!-- ... 子 body 链 ... -->
      </body>
    </body>
  </worldbody>

  <!-- 3. 闭环约束（如有平行四边形腿） -->
  <equality>
    <connect name="L_loop1" body1="LLowerLink2" body2="LShankLink" .../>
    <!-- ... -->
  </equality>

  <!-- 4. 碰撞排除（相邻 body + 闭环 body） -->
  <contact>
    <exclude body1="BaseLink" body2="LHipLink"/>
    <!-- ... -->
  </contact>

  <!-- 5. 本体传感器（IMU 类） -->
  <sensor>
    <gyro name="imu_ang_vel" site="imu"/>
    <velocimeter name="imu_lin_vel" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
    <subtreeangmom name="root_angmom" body="BaseLink"/>
    <framezaxis name="upvector" objtype="site" objname="imu"/>
  </sensor>

  <!-- 6. 执行器 -->
  <actuator>
    <position name="LHipJoint"    joint="LHipJoint"    kp="20.0" kv="1.0" forcerange="-27 27"/>
    <position name="LUpperJoint"  joint="LUpperJoint"  kp="20.0" kv="1.0" forcerange="-27 27"/>
    <position name="LLowerJoint1" joint="LLowerJoint1" kp="20.0" kv="1.0" forcerange="-27 27"/>
    <position name="RHipJoint"    joint="RHipJoint"    kp="20.0" kv="1.0" forcerange="-27 27"/>
    <position name="RUpperJoint"  joint="RUpperJoint"  kp="20.0" kv="1.0" forcerange="-27 27"/>
    <position name="RLowerJoint1" joint="RLowerJoint1" kp="20.0" kv="1.0" forcerange="-27 27"/>
  </actuator>
</mujoco>
```

**关键规则（来自 AGENTS.md）**：
- `<keyframe>` **禁止**放在 robot.xml，必须放在 scene.xml
- robot.xml 只描述机器人本体，与 task/场景无关
- actuator 的 `name` 要和 `joint name` 一致（env 代码通过 name 索引）

### 1.3 scene_flat.xml 结构

scene.xml 通过 `<include>` 引入 robot.xml，并添加场景内容。

```xml
<mujoco model="my_bipedal flat scene">
  <include file="mujocoBipedalenv.xml"/>

  <statistic center="0 0 1.0" extent=".5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" .../>
    <material name="groundplane" texture="groundplane" .../>
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>

  <!-- 场景级传感器：脚-地面接触检测 -->
  <sensor>
    <contact name="left_foot_contact"  geom1="floor" geom2="LFootLink" data="found" reduce="mindist"/>
    <contact name="right_foot_contact" geom1="floor" geom2="RFootLink" data="found" reduce="mindist"/>
    <contact name="left_foot_force"    geom1="floor" geom2="LFootLink" data="force" reduce="netforce"/>
    <contact name="right_foot_force"   geom1="floor" geom2="RFootLink" data="force" reduce="netforce"/>
  </sensor>

  <!-- keyframe：初始姿态（必须在此处！） -->
  <keyframe>
    <key name="stand"
      qpos="0 0 0.7  1 0 0 0  -0.1 0.2 0.3 0 0 0 0 0 0 0  -0.1 0.2 0.3 0 0 0 0 0 0 0"
      ctrl="-0.1 0.2 0.3  -0.1 0.2 0.3"/>
  </keyframe>
</mujoco>
```

**keyframe qpos 格式**（共 27 维 = 7 base + 20 joint）：
```
0 0 0.7        ← base position (x=0, y=0, z=0.7m)
1 0 0 0        ← base quaternion (w=1, x=0, y=0, z=0 = 直立)
-0.1 0.2 0.3 0 0 0 0 0 0 0   ← 左腿 10 个 joint 的初始角度
-0.1 0.2 0.3 0 0 0 0 0 0 0   ← 右腿 10 个 joint 的初始角度
```

> **重要**：keyframe name（如 `"stand"`）需要在 env 代码里通过 `_keyframe_name` 引用。

---

## 2. Env 实现（`joystick.py`）

### 2.1 文件位置

```
src/unilab/envs/locomotion/rlbipedal/
├── __init__.py    # from .joystick import MyBipedalWalkEnv, MyBipedalWalkFlatCfg
└── joystick.py    # 完整 env 实现
```

### 2.2 继承关系

```
NpEnv (src/unilab/base/np_env.py)
  └── LocomotionBaseEnv (src/unilab/envs/locomotion/common/base.py)
        └── MyBipedalWalkEnv   ← 你的 env
```

### 2.3 必须实现的方法

| 方法 | 职责 |
|------|------|
| `apply_action(actions, state) -> ctrl` | 将 RL 策略输出的 normalized action 转为 MuJoCo ctrl |
| `update_state(state) -> NpEnvState` | 读传感器、算 obs、算 reward、判终止 |
| `obs_groups_spec -> dict[str, int]` | 声明 actor/critic obs 维度 |
| `_init_buffers()` | 初始化 actuated joint indices、default_angles |
| `_compute_obs(info, ...) -> dict` | 拼接 actor/critic 观测 |
| `_compute_reward(info, ...) -> np.ndarray` | 计算多目标奖励 |
| `__init__(cfg, num_envs, backend_type)` | 创建 backend、初始化 buffer |

### 2.4 核心代码骨架

```python
from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.base import LocomotionBaseEnv, ...

# ── Config dataclasses ──

@dataclass
class MyBipedalWalkEnvCfg(LocomotionBaseCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "RLbipedal" / "scene_flat.xml")
    ))
    max_episode_seconds: float = 20.0
    reward_config: MyBipedalRewardConfig = field(default_factory=MyBipedalRewardConfig)
    # ...

# ── Env class ──

class MyBipedalWalkEnv(LocomotionBaseEnv):
    _cfg: MyBipedalWalkEnvCfg
    _keyframe_name: str = "stand"  # 必须与 scene.xml 中 keyframe name 一致

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        backend = create_backend(
            backend_type, cfg.scene, num_envs, cfg.sim_dt,
            base_name=cfg.asset.base_name,        # "BaseLink"
            push_body_name=cfg.asset.foot_name,   # "LFootLink"
        )
        super().__init__(cfg, backend, num_envs)

    @property
    def obs_groups_spec(self):
        return {"obs": 57, "critic": 72}  # actor=57D, critic=72D

    def apply_action(self, actions, state):
        # ctrl = actions * action_scale + default_angles
        ctrl = actions * self._action_scale_joint + self.default_angles
        return ctrl

    def update_state(self, state):
        # 1. 读传感器
        # 2. 判终止（tilt > max_tilt 或 base_z < base_height_min）
        # 3. 算 reward
        # 4. 算 obs
        return state.replace(obs=obs, reward=reward, terminated=terminated)

# ── 注册 ──

@registry.envcfg("MyBipedalWalkFlat")
@dataclass
class MyBipedalWalkFlatCfg(MyBipedalWalkEnvCfg):
    pass

registry.register_env("MyBipedalWalkFlat", MyBipedalWalkEnv, sim_backend="mujoco")
```

### 2.5 action 映射（关键）

UniLab 的 action 合同是 `ctrl = actions * action_scale + default_angles`：

- `actions`：RL 策略输出，通常在 [-1, 1] 范围
- `action_scale`：缩放系数（RLbipedal 用 0.3375，每关节可不同）
- `default_angles`：keyframe 中 actuated joint 的初始角度
- `ctrl`：MuJoCo position actuator 的目标位置

```python
# RLbipedal 的 per-joint action scale
_BASE_ACTION_SCALE = 0.25 * 27.0 / 20.0  # = 0.3375
_PER_JOINT_SCALE = {
    "LHipJoint":    _BASE_ACTION_SCALE,
    "LUpperJoint":  _BASE_ACTION_SCALE,
    "LLowerJoint1": _BASE_ACTION_SCALE * 0.5,  # 膝关节减半
    # ... R 侧同理
}
```

### 2.6 actuated joint indices 初始化

```python
_ACTUATOR_NAMES = ("LHipJoint", "LUpperJoint", "LLowerJoint1",
                   "RHipJoint", "RUpperJoint", "RLowerJoint1")

def _init_buffers(self):
    # 从 backend 获取 actuated joint 在 qpos 中的索引
    self._actuated_dof_pos_indices = list(
        self._backend.get_joint_dof_pos_indices(list(_ACTUATOR_NAMES))
    )
    super()._init_buffers()
    # 全部 20 个 DOF 的默认位置（含弹簧关节）
    raw_qpos = self._backend.get_keyframe_qpos(self._keyframe_name)
    self._all_default_angles = np.asarray(raw_qpos[-20:])
    # actuated joint 的默认角度
    self.default_angles = self._all_default_angles[self._actuated_dof_pos_indices].copy()
```

---

## 3. 观测设计

### 3.1 Actor 观测（策略可见）

RLbipedal actor obs = 57 维：

| 分量 | 维度 | 来源 |
|------|:----:|------|
| gyro | 3 | `imu_ang_vel` sensor |
| projected gravity | 3 | `upvector` sensor（取负） |
| joint_pos_rel | 20 | `dof_pos - all_default_angles`（全部 20 关节） |
| joint_vel | 20 | `dof_vel`（全部 20 关节） |
| last_actions | 6 | 上一步的 RL action |
| command | 3 | (vx, vy, wz) 速度指令 |
| gait_phase | 2 | 步态相位（左/右脚） |

### 3.2 Critic 观测（含特权信息）

RLbipedal critic obs = 72 维 = actor(57) + 额外(15)：

| 分量 | 维度 | 来源 |
|------|:----:|------|
| actor 全部 | 57 | 同上 |
| linvel | 3 | `imu_lin_vel`（真实线速度，无噪声） |
| foot_height | 2 | 左右脚 z 坐标 |
| foot_air_time | 2 | 左右脚腾空时间 |
| foot_contact | 2 | 左右脚接触布尔值 |
| foot_contact_forces | 6 | 左右脚接触力（log 归一化） |

> `obs_groups_spec` 必须与实际拼接维度严格一致。

---

## 4. 奖励函数设计

### 4.1 奖励项总表

```python
@dataclass
class MyBipedalRewardConfig:
    scales: dict[str, float] = field(default_factory=lambda: {
        # ── 核心追踪（正奖励） ──
        "track_lin_vel_xy":     2.0,   # 跟踪 xy 线速度指令
        "track_lin_vel_y":      1.0,   # 额外的 y 方向追踪
        "track_ang_vel":        1.0,   # 跟踪角速度指令
        # ── 姿态稳定（负奖励/惩罚） ──
        "body_orientation_l2": -1.0,   # 偏离直立（roll/pitch 平方）
        "body_ang_vel":        -0.05,  # roll/pitch 角速度
        "flat_orientation":    -0.5,   # 投影重力惩罚
        "angular_momentum":    -0.025, # 角动量
        # ── 步态 ──
        "foot_gait":            0.5,   # 时钟步态匹配
        "symmetry":            -0.5,   # 左右关节对称性
        # ── 能量/耗损 ──
        "torque_penalty":      -0.03,  # 扭矩
        "action_rate_l2":      -0.05,  # action 变化率
        "joint_acc_l2":        -2.5e-7,# 关节加速度
        # ── 姿态 ──
        "pose":                 1.0,   # 关节回归默认（速度自适应 std）
        # ── 足部约束 ──
        "foot_clearance":      -1.0,   # 摆动相抬脚高度不足
        "foot_slip":           -0.25,  # 接触期脚底滑动
        "soft_landing":        -0.001, # 着地冲击力
        # ── 高度 ──
        "base_height":          0.3,   # 软边界 [min, max]
        "phase_height":         0.3,   # 接触驱动的高度振荡
        # ── 约束 ──
        "joint_pos_limits":   -10.0,   # 关节接近限位
        "stand_still":         -1.0,   # 静止时关节偏移
        "is_terminated":     -200.0,   # 终止大惩罚
    })
```

### 4.2 关键参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `base_height_min` | 0.50 | base_z 低于此值则 terminated |
| `base_height_max` | 1.0 | base_height 奖励的上界 |
| `max_tilt_deg` | 70.0 | 倾斜角超过此值则 terminated |
| `tracking_sigma` | 0.5 | 速度追踪 exp 奖励的 std |
| `gait_period` | 0.6 | 步态周期（秒） |
| `command_threshold` | 0.1 | 低于此速度视为"静止" |
| `foot_clearance_target` | 0.10 | 期望抬脚高度（米） |

### 4.3 终止条件（在 `update_state` 中）

```python
tilt = np.arccos(np.clip(gravity[:, 2], -1.0, 1.0))
fell_over = tilt > math.radians(self._reward_cfg.max_tilt_deg)  # 70度
too_low = actual_base_z < self._reward_cfg.base_height_min      # 0.50m
terminated = np.logical_or(fell_over, too_low)
```

> **调试经验**：如果 robot 默认姿态就活不过 10 步，先检查 `base_height_min` 是否合理。弹簧腿机器人默认姿态下沉很快，可能需要 curriculum（降低 `base_height_min` 到 0.35）。

---

## 5. Task YAML 配置

### 5.1 文件位置

```
conf/offpolicy/task/sac/my_bipedal_flat/mujoco.yaml
```

### 5.2 完整配置

```yaml
# @package _global_
training:
  task_name: MyBipedalWalkFlat    # 必须与 registry.register_env() 的名字一致
  sim_backend: mujoco

algo:
  num_envs: 4096
  max_iterations: 5000
  save_interval: 1000
  learning_starts: 10
  updates_per_step: 8
  replay_buffer_n: 512
  batch_size: 8192
  gamma: 0.99

  # ── Warm-start（解决 critic 冷启动）──
  warm_start:
    enabled: true
    source: standing_controller    # 或 sac_checkpoint（阶段2）
    steps: 100000
    noise_std: 0.15
    action_hold: 4
    learning_after_warm_start: true
    relax_termination_height: 0.35 # warm-start 时临时放宽终止
    checkpoint_path: null          # 阶段2 用阶段1 的 checkpoint

  # ── 算法超参 ──
  algo_params:
    alpha_init: 0.1
    target_entropy_ratio: 0.5
    log_std_min: -2.0              # 限制探索噪声
    log_std_max: -0.5              # 初始 action_std=0.29
    max_grad_norm: 10.0
    use_compile: false

# ── Env 控制 ──
env:
  control_config:
    action_scale: 0.25
  noise_config:
    level: 1.0
    scale_joint_angle: 0.01
    scale_joint_vel: 1.5
    scale_gyro: 0.2
    scale_gravity: 0.05
    scale_linvel: 0.5
  curriculum:
    enabled: true

# ── Reward ──
reward:
  scales:
    track_lin_vel_xy: 2.0
    track_lin_vel_y: 1.0
    track_ang_vel: 1.0
    body_orientation_l2: -1.0
    body_ang_vel: -0.05
    flat_orientation: -0.5
    angular_momentum: -0.025
    foot_gait: 0.5
    symmetry: -0.5
    torque_penalty: -0.03
    action_rate_l2: -0.05
    joint_acc_l2: -2.5e-7
    pose: 1.0
    foot_clearance: -1.0
    foot_slip: -0.25
    soft_landing: -0.001
    base_height: 0.3
    phase_height: 0.3
    joint_pos_limits: -10.0
    stand_still: -1.0
    is_terminated: -200.0
  tracking_sigma: 0.5
  tracking_ang_sigma: 0.7071
  base_height_min: 0.35            # curriculum：从 0.50 降到 0.35
  base_height_max: 1.0
  phase_height_base: 0.7
  phase_height_amp: 0.08
  gait_period: 0.6
  command_threshold: 0.1
  max_tilt_deg: 70.0
  walking_threshold: 0.1
  running_threshold: 1.5
  foot_clearance_target: 0.10
```

### 5.3 YAML 覆盖规则

Hydra 的 compose 顺序：`config.yaml` → `algo/sac.yaml` → `task/sac/my_bipedal_flat/mujoco.yaml`

- `algo/sac.yaml` 提供算法默认值（`max_iterations=500`, `log_std_min=-5.0` 等）
- task YAML 覆盖算法默认值（`max_iterations=5000`, `log_std_min=-2.0`）
- 命令行可进一步覆盖（`algo.max_iterations=3000`）

---

## 6. 训练命令

### 6.1 单阶段训练（standing_controller warm-start）

```bash
uv run python scripts/train_offpolicy.py \
  task=sac/my_bipedal_flat/mujoco \
  algo.max_iterations=3000
```

### 6.2 两阶段训练（推荐）

**Stage 1**：standing controller warm-start，产出能走路的 checkpoint

```bash
uv run python scripts/train_offpolicy.py \
  task=sac/my_bipedal_flat/mujoco \
  algo.max_iterations=3000
```

**Stage 2**：用 Stage 1 的 checkpoint 做 prefill，完全收敛

```bash
uv run python scripts/train_offpolicy.py \
  task=sac/my_bipedal_flat/mujoco \
  algo.max_iterations=5000 \
  algo.warm_start.source=sac_checkpoint \
  algo.warm_start.checkpoint_path=logs/fast_sac/MyBipedalWalkFlat/<stage1时间戳>_mujoco/model_3000.pt \
  algo.warm_start.steps=200000 \
  algo.warm_start.relax_termination_height=0.35
```

### 6.3 仅推理（playback）

```bash
uv run python scripts/train_offpolicy.py \
  task=sac/my_bipedal_flat/mujoco \
  training.play_only=true \
  algo.load_run=-1
```

---

## 7. 调试检查清单

### 7.1 XML 层面

- [ ] `robot.xml` 不含 `<keyframe>`
- [ ] `scene_flat.xml` 有 `<include file="robot.xml"/>`
- [ ] keyframe qpos 维度 = 7(base) + N(joint)，数值合理
- [ ] actuator name 和 joint name 一致
- [ ] `<contact>` 排除了相邻 body
- [ ] 传感器名称与 env 代码一致（`imu_ang_vel`, `upvector`, `left_foot_contact` 等）

### 7.2 Env 代码层面

- [ ] `_keyframe_name` 与 scene.xml 中 keyframe name 一致
- [ ] `obs_groups_spec` 维度与 `_compute_obs` 拼接结果一致
- [ ] `_ACTUATOR_NAMES` 与 XML actuator name 一致
- [ ] `apply_action` 的 `ctrl = actions * scale + default_angles`
- [ ] `registry.register_env("TaskName", EnvClass, sim_backend="mujoco")` 的 TaskName 与 YAML `task_name` 一致

### 7.3 训练前验证

```bash
# 验证 env 能 reset + step
uv run python -c "
from unilab.base import registry
from unilab.base.registry import ensure_registries
ensure_registries()
env = registry.make('MyBipedalWalkFlat', num_envs=4, sim_backend='mujoco')
env.init_state()
import numpy as np
obs, info = env.reset(np.arange(4, dtype=np.int32))
print('obs keys:', list(obs.keys()))
for k, v in obs.items():
    print(f'  {k}: shape={v.shape}')
state = env.step(np.zeros((4, 6), dtype=np.float32))
print('reward:', state.reward)
print('terminated:', state.terminated)
env.close()
"
```

### 7.4 常见问题

| 症状 | 可能原因 | 解决 |
|------|----------|------|
| episode 只有 ~10 步 | `base_height_min` 太严 / 默认姿态不稳 | 降低 `base_height_min` 或加 warm-start |
| reward 卡在 -3.7 不动 | critic 冷启动，buffer 全是摔倒数据 | 开启 warm_start |
| action_std 越来越大 | `log_std_max` 太大 | 限制 `log_std_max=-0.5` |
| `UnboundLocalError: torch` | warm_start.py 的 import 作用域 bug | 确认顶层 `import torch`，不在 if 分支内 import |
| `KeyError: 'warm_start'` | 当前 worktree 的 configs 被还原 | 在有 warm-start 改动的 worktree 里跑 |
| obs 维度不匹配 | `obs_groups_spec` 与 `_compute_obs` 不一致 | 检查拼接维度 |
| joint index 错误 | actuator name 与 XML 不一致 | 核对 `_ACTUATOR_NAMES` |

---

## 8. 文件对照速查

| 你要改什么 | 改哪个文件 |
|-----------|-----------|
| 机器人几何/关节/执行器 | `assets/robots/<name>/robot.xml` |
| 初始姿态 | `assets/robots/<name>/scene_flat.xml` 的 `<keyframe>` |
| 地面/接触传感器 | `assets/robots/<name>/scene_flat.xml` |
| obs 维度/内容 | `envs/locomotion/<name>/joystick.py` 的 `obs_groups_spec` + `_compute_obs` |
| 奖励项/scale | `joystick.py` 的 `MyBipedalRewardConfig` + task YAML 的 `reward.scales` |
| 终止阈值 | task YAML 的 `reward.base_height_min` / `reward.max_tilt_deg` |
| action_scale | `joystick.py` 的 `_PER_JOINT_SCALE` 或 task YAML 的 `env.control_config.action_scale` |
| 观测噪声 | task YAML 的 `env.noise_config` |
| warm_start | task YAML 的 `algo.warm_start` |
| log_std 范围 | task YAML 的 `algo.algo_params.log_std_min/max` |
| 训练迭代数 | 命令行 `algo.max_iterations=N` 或 task YAML |
| 注册新任务 | `joystick.py` 末尾的 `registry.register_env(...)` |

---

## 9. 参考：与 G1 的对比

| 维度 | RLbipedal | G1 |
|------|-----------|-----|
| robot.xml | `mujocoBipedalenv.xml` | `g1.xml` |
| DOF | 20（6 actuated） | 29（29 actuated） |
| actuator 类型 | position (kp=20) | motor/PD |
| keyframe name | `stand` | `stand` |
| contact sensor | 2（左右脚） | 8（每脚 4 个） |
| obs_dim | 57 | 76 |
| critic_dim | 72 | 88 |
| 终止阈值 | base_height_min=0.35 | base_height_min=0.50 |
| warm_start | 需要（弹簧腿不稳） | 不需要（全驱动稳定） |

G1 是全驱动机器人，默认姿态稳定，不需要 warm-start 和 curriculum。RLbipedal 有 14 个无驱动弹簧关节，默认姿态不稳定，需要三要素（warm-start + curriculum + log_std 限制）才能收敛。

---

## 10. 总结

接入自定义机器人的完整流程：

1. **准备 XML**：robot.xml（本体）+ scene_flat.xml（场景+keyframe）+ mesh 文件
2. **实现 Env**：继承 `LocomotionBaseEnv`，实现 `apply_action` / `update_state` / `obs_groups_spec` / `_compute_obs` / `_compute_reward`
3. **注册任务**：`registry.register_env("TaskName", EnvClass, sim_backend="mujoco")`
4. **写 task YAML**：配置 reward scale / warm_start / log_std / base_height_min 等
5. **验证 env**：跑 reset + step 确认维度和数值正常
6. **训练**：单阶段或两阶段（warm-start → checkpoint prefill → 正式训练）
7. **调试**：如果 10 步就摔，先检查终止条件和默认姿态稳定性

详细的 SAC 调试经验（从 10 步摔到完全收敛）见 [sac调试经验.md](./sac调试经验.md)。
