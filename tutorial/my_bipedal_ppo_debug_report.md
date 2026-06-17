# My Bipedal PPO Training Debug Report

> 将 RLbipedal_mjlab 的双足机器人（my_bipedal）迁移到 UniLab 框架，通过 PPO 训练实现平地行走。

- **训练日期**: 2026-06-16
- **运行目录**: `logs/rsl_rl_ppo/MyBipedalWalkFlat/2026-06-16_22-54-03_mujoco/`
- **机器人**: 6 个执行器关节 (LHipJoint, LUpperJoint, LLowerJoint1, RHipJoint, RUpperJoint, RLowerJoint1)，20 个总 DOF
- **框架**: UniLab + MuJoCo 3.8.0 + PyTorch CUDA

---

## 1. 训练结果

| 指标 | iter 0 | iter 100 | iter 275 | iter 460 |
|------|:-----:|:------:|:------:|:------:|
| Mean reward | -3.94 | -3.18 | -2.70 | **-1.35** |
| Episode length (steps) | 11.9 | ~25 | 40.9 | **114.3** |
| Termination rate | 9.0% | 3.9% | 2.6% | **0.95%** |
| Actual linvel_x (m/s) | 0.00 | 0.25 | 0.23 | **0.44** |
| Entropy | 8.52 | 1.27 | 0.93 | 1.33 |
| action_std | 1.00 | 0.31 | 0.30 | 0.32 |
| body_ang_vel penalty | -0.39 | -0.10 | -0.08 | -0.09 |

- Episode length 从 12 → 114（**9.6x**，约 2.3 秒行走）
- 终止率从 9% 降至 <1%
- 实际行走速度达到 0.44 m/s
- 模型 checkpoint 保存到 iter 900

---

## 2. 发现的 8 个 Bug 及修复

### Bug 1 — tracking_sigma 错误

- **文件**: `conf/ppo/task/my_bipedal_flat/mujoco.yaml`
- **问题**: `tracking_sigma: 0.25`，RLbipedal 参考值为 `0.5`
- **影响**: 速度跟踪奖励衰减过快，sigma=0.25 时方差为 0.0625，sigma=0.5 时为 0.25
- **修复**: 改为 `tracking_sigma: 0.5`

### Bug 2 — _r_is_terminated 始终返回 0

- **文件**: `src/unilab/envs/locomotion/rlbipedal/joystick.py`
- **问题**: `_r_is_terminated` 直接返回全零数组，-200 死亡惩罚从未生效
- **修复**: 在 `update_state()` 中存储 `state.info["_just_terminated"]`，在 `_r_is_terminated` 中读取

### Bug 3 — _r_joint_pos_limits 使用 ctrl_range 错误

- **文件**: `src/unilab/envs/locomotion/rlbipedal/joystick.py`
- **问题**: `actuator_ctrl_range` 在 `ctrllimited=0` 时返回 `[0, 0]`，导致每个关节都被判为超限
- **影响**: 每步产生 -6 到 -7.5 的巨大误判惩罚
- **修复**: 改用 `m.jnt_range` 读取 MuJoCo 实际关节范围

### Bug 4 — 关节索引错误：dof_pos[:, :6]

- **文件**: `src/unilab/envs/locomotion/rlbipedal/joystick.py`
- **问题**: 多个 reward 函数使用 `dof_pos[:, :6]` 索引受控关节，但前 6 个 DOF 是 6D 浮动基座（x,y,z,quat），不是执行器关节
- **实际受控关节索引**: `[6, 7, 8, 16, 17, 18]`
- **修复**: 引入 `_actuated_dof_pos_indices`，统一使用 `dof_pos[:, self._actuated_dof_pos_indices]`

### Bug 5 — self.default_angles 取错关节

- **文件**: `src/unilab/envs/locomotion/common/base.py` (父类 `LocomotionBaseEnv._init_buffers`)
- **问题**: `self.default_angles = _init_qpos[-num_action:]` 取了 qpos 最后 6 个值（这些是非受控弹簧关节 RShankSpringJointL 等的位置），而非 6 个受控关节
- **修复**: 子类 `_init_buffers` 中 override：`self.default_angles = self._all_default_angles[self._actuated_dof_pos_indices]`

### Bug 6 — _actuated_dof_pos_indices 初始化时序

- **文件**: `src/unilab/envs/locomotion/rlbipedal/joystick.py`
- **问题**: `_init_buffers` 在 `super().__init__` 中被调用，此时 `_actuated_dof_pos_indices` 尚未设置
- **修复**: 在子类 `_init_buffers` 开头先计算 `_actuated_dof_pos_indices`，再调 `super()._init_buffers()`

### Bug 7 — 缺少 armature 和错误的 ctrllimited

- **文件**: `src/unilab/assets/robots/RLbipedal/mujocoBipedalenv.xml`
- **问题**:
  - 6 个受控关节缺少 `armature="0.01"`（RLbipedal 的 `BuiltinPositionActuatorCfg(armature=0.01)` 会在运行时设置）
  - actuator 标签缺少 `ctrllimited="false"` 和 `inheritrange="0"`
- **影响**: 缺少 armature 导致关节转子惯性为 0，位置控制器不稳定
- **修复**: 给 6 个关节添加 `armature="0.01"`，actuator 添加 `ctrllimited="false" inheritrange="0"`

### Bug 8 — Spring joint damping 不一致

- **文件**: `src/unilab/assets/robots/RLbipedal/mujocoBipedalenv.xml`
- **问题**: 弹簧关节 (LShankSpringJointL, RShankSpringJointL, LSpringJointL 等) 的 `damping=2`，RLbipedal 参考值为 `5`
- **修复**: 统一改为 `damping="5"`

---

## 3. 奖励函数设计

22 项奖励函数完全对齐 RLbipedal：

| 类别 | 奖励项 | 权重 | 作用 |
|------|--------|:---:|------|
| 速度跟踪 | track_lin_vel_xy | 2.0 | 跟踪 xy 线速度指令 |
| | track_lin_vel_y | 1.0 | y 方向速度 |
| | track_ang_vel | 1.0 | 角速度跟踪 |
| 姿态稳定 | body_orientation_l2 | -1.0 | 保持躯干水平 |
| | body_ang_vel | -0.05 | 抑制躯干角速度 |
| | flat_orientation | -0.5 | 平坦地面朝向 |
| | angular_momentum | -0.025 | 角动量最小化 |
| 步态 | foot_gait | 0.5 | 交替步态模式 |
| | symmetry | -0.5 | 左右对称 |
| 能量 | torque_penalty | -0.03 | 力矩惩罚 |
| | action_rate_l2 | -0.05 | 动作平滑性 |
| | joint_acc_l2 | -2.5e-7 | 关节加速度 |
| 姿态 | pose | 1.0 | 关节角度跟踪 |
| 足部 | foot_clearance | -1.0 | 足部离地高度 |
| | foot_slip | -0.25 | 足部滑移 |
| | soft_landing | -0.001 | 软着陆 |
| 高度 | base_height | 0.3 | 基座高度(软边界) |
| | phase_height | 0.3 | 步态相位高度振荡 |
| 约束 | joint_pos_limits | -10.0 | 关节限位 |
| | stand_still | -1.0 | 静立惩罚(步态时) |
| 终止 | is_terminated | -200.0 | 死亡惩罚 |

---

## 4. PPO 超参数

```yaml
num_envs: 4096           # 并行环境数
num_steps_per_env: 24    # 每环境每迭代步数
hidden_dims: [512,256,128]  # Actor/Critic 网络
activation: elu
learning_rate: 1.0e-3
gamma: 0.99
lam: 0.95
clip_param: 0.2
entropy_coef: 0.01
desired_kl: 0.01
max_grad_norm: 1.0
empirical_normalization: true
action_scale: 0.25       # 执行器缩放
ctrl_dt: 0.02            # 控制间隔 (decimation=4, sim_dt=0.005)
```

---

## 5. Evaluation 命令

### 官方格式 (推荐)

```bash
cd /home/peterpan/UniLab && uv run eval \
  --algo ppo \
  --task my_bipedal_flat \
  --sim mujoco \
  --load-run logs/rsl_rl_ppo/MyBipedalWalkFlat/2026-06-16_22-54-03_mujoco/model_<ITER>.pt
```

### 直接调用脚本

```bash
cd /home/peterpan/UniLab && .venv/bin/python scripts/train_rsl_rl.py \
  task=my_bipedal_flat/mujoco \
  training.play_only=true \
  algo.load_run=logs/rsl_rl_ppo/MyBipedalWalkFlat/2026-06-16_22-54-03_mujoco \
  algo.checkpoint=<ITER>
```

替换 `<ITER>` 为目标 checkpoint (0, 100, 200, ..., 900)。

### 训练命令

```bash
cd /home/peterpan/UniLab && .venv/bin/python scripts/train_rsl_rl.py \
  task=my_bipedal_flat/mujoco
```

---

## 6. 关键文件清单

| 文件 | 路径 |
|------|------|
| 环境代码 | `src/unilab/envs/locomotion/rlbipedal/joystick.py` |
| MuJoCo 模型 | `src/unilab/assets/robots/RLbipedal/mujocoBipedalenv.xml` |
| 场景定义 | `src/unilab/assets/robots/RLbipedal/scene_flat.xml` |
| 训练配置 | `conf/ppo/task/my_bipedal_flat/mujoco.yaml` |
| 训练入口 | `scripts/train_rsl_rl.py` |
| 训练日志 | `logs/rsl_rl_ppo/MyBipedalWalkFlat/2026-06-16_22-54-03_mujoco/` |

---

## 7. 经验总结

1. **MuJoCo actuator 参数至关重要** — `armature`、`ctrllimited`、`inheritrange` 这些不起眼的参数直接影响物理稳定性。RLbipedal 用 `BuiltinPositionActuatorCfg` 在代码中设置这些，迁移到 UniLab 的 XML actuator 时容易遗漏
2. **关节索引不能想当然** — 双足机器人的浮动基座占 6 个 DOF，`dof_pos[:,:6]` 是基座位姿而非关节角。受控关节在 DOF 数组中的位置是不连续的
3. **default_angles 必须精确** — 位置控制器中 `ctrl = action * scale + default_angles`，错误的 default_angles 意味着始终指向错误的关节位置
4. **逐个对比 reward 项** — 通过 `state.info["log"]` 输出每项 reward 的均值是定位问题的最快方法。例如 `joint_pos_limits` 为 -6 直接暴露了 ctrl_range 问题
5. **不要等静态站立** — RL 无需机器人初始就能稳定站立，策略会从不断试错中学会动态平衡。Episode length 的增长比 reward 的绝对值更能反映学习进度
6. **XML vs 代码 actuator 配置** — RLbipedal 用程序化方式创建 actuator（自动注入 armature），UniLab 用 XML 静态定义。两者参数必须完全对齐
