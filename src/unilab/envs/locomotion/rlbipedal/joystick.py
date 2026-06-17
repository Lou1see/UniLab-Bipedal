"""MyBipedal flat-terrain locomotion environment.

Direct-style implementation matching the RLbipedal project's reward/observation
configuration for the my_bipedal robot. Uses the UniLab LocomotionBaseEnv pattern
(not the mjlab ManagerBasedRlEnvCfg pattern).

Observation dimensions:
  - actor (obs): 57  = gyro(3)+gravity(3)+command(3)+phase(2)+joint_pos(20)+joint_vel(20)+action(6)
  - critic:      72  = actor(57) + linvel(3)+foot_height(2)+foot_air_time(2)+foot_contact(2)+foot_contact_forces(6)

Robot: 20 total joints, 6 actuated (position control).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    Sensor as LocomotionSensor,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.commands import (
    Commands,
    sample_heading_commands,
    zero_small_xy_commands,
)

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MyBipedalNoiseConfig(BaseNoiseConfig):
    """Observation noise config matching RLbipedal flat terrain."""

    level: float = 1.0
    scale_joint_angle: float = 0.01
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.5  # RLbipedal U[-0.5, 0.5]


@dataclass
class MyBipedalControlConfig(ControlConfigBase):
    """Position-control config with per-joint action_scale override."""

    action_scale: float | np.ndarray = 0.25  # overridden in env init
    simulate_action_latency: bool = False


@dataclass
class MyBipedalSensor(LocomotionSensor):
    """Sensor name mapping for my_bipedal XML."""

    local_linvel: str = "imu_lin_vel"
    gyro: str = "imu_ang_vel"
    upvector: str = "upvector"


@dataclass
class MyBipedalAsset:
    """Body/site naming conventions."""

    base_name = "BaseLink"
    foot_name = "LFootLink"  # used for push_body
    ground = "floor"


@dataclass
class MyBipedalBaseCfg(LocomotionBaseCfg):
    """Base config for my_bipedal locomotion."""

    noise_config: MyBipedalNoiseConfig = field(default_factory=MyBipedalNoiseConfig)
    control_config: MyBipedalControlConfig = field(default_factory=MyBipedalControlConfig)
    sensor: MyBipedalSensor = field(default_factory=MyBipedalSensor)
    asset: MyBipedalAsset = field(default_factory=MyBipedalAsset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02


@dataclass
class MyBipedalDomainRandConfig(DomainRandConfig):
    """Domain randomization for my_bipedal (no PD gain randomization)."""

    randomize_kp: bool = False
    randomize_kd: bool = False


@dataclass
class CurriculumConfig:
    """Discrete velocity curriculum matching RLbipedal."""

    enabled: bool = True
    # 3-stage: stand → forward sprint → full range
    # Stage definitions (step counts are in environment steps):
    velocity_stages: list[dict] = field(
        default_factory=lambda: [
            {"step": 0, "lin_vel_x": (-0.05, 0.05), "lin_vel_y": (-0.05, 0.05), "ang_vel_z": (-0.05, 0.05)},
            {"step": 2000 * 24, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (2.5, 3.0)},
            {"step": 4000 * 24, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-2.5, -3.0)},
            {"step": 6000 * 24, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-3.0, 3.0)},
        ]
    )


@dataclass
class MyBipedalRewardConfig:
    """Reward scales matching RLbipedal my_bipedal_flat_env_cfg."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            # Core tracking
            "track_lin_vel_xy": 2.0,
            "track_lin_vel_y": 1.0,
            "track_ang_vel": 1.0,
            # Orientation / stability
            "body_orientation_l2": -1.0,
            "body_ang_vel": -0.05,
            "flat_orientation": -0.5,
            "angular_momentum": -0.025,
            # Gait
            "foot_gait": 0.5,
            "symmetry": -0.5,
            # Energy / effort
            "torque_penalty": -0.03,
            "action_rate_l2": -0.05,
            "joint_acc_l2": -2.5e-7,
            # Pose
            "pose": 1.0,
            # Foot constraints
            "foot_clearance": -1.0,
            "foot_slip": -0.25,
            "soft_landing": -0.001,
            # Height
            "base_height": 0.3,
            "phase_height": 0.3,
            # Constraints
            "joint_pos_limits": -10.0,
            "stand_still": -1.0,
            "is_terminated": -200.0,
        }
    )
    tracking_sigma: float = 0.5
    tracking_ang_sigma: float = math.sqrt(0.5)
    base_height_min: float = 0.50
    base_height_max: float = 1.0
    phase_height_base: float = 0.7
    phase_height_amp: float = 0.08
    gait_period: float = 0.6
    command_threshold: float = 0.1
    max_tilt_deg: float = 70.0
    pose_std_standing: dict = field(default_factory=lambda: {".*": 0.05})
    pose_std_walking: dict = field(default_factory=lambda: {
        ".*Hip.*": 0.15, ".*Upper.*": 0.3, ".*Lower.*": 0.5,
        ".*ShankSpring.*": 0.5, "^[LR]ShankJoint$": 0.5,
        ".*Foot.*": 0.3, "^[LR]Spring.*": 0.5,
    })
    pose_std_running: dict = field(default_factory=lambda: {
        ".*Hip.*": 0.2, ".*Upper.*": 0.4, ".*Lower.*": 0.5,
        ".*ShankSpring.*": 0.5, "^[LR]ShankJoint$": 0.5,
        ".*Foot.*": 0.4, "^[LR]Spring.*": 0.5,
    })
    walking_threshold: float = 0.1
    running_threshold: float = 1.5
    foot_clearance_target: float = 0.10
    knee_early_fraction: float = 0.3
    knee_late_fraction: float = 0.7
    stance_duration: float = 0.3


@dataclass
class MyBipedalWalkEnvCfg(MyBipedalBaseCfg):
    """Full env config for MyBipedal flat walking."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "RLbipedal" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    commands: Commands = field(default_factory=Commands)
    reward_config: MyBipedalRewardConfig = field(default_factory=MyBipedalRewardConfig)
    domain_rand: MyBipedalDomainRandConfig = field(default_factory=MyBipedalDomainRandConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    gait_phase_init_mode: str = "offset_phase"
    reset_base_qvel_limit: float = 0.5
    heading_command: bool = True


# ---------------------------------------------------------------------------
# Helper: per-joint action scale (RLbipedal-style)
# ---------------------------------------------------------------------------

# Actuator params from my_bipedal_constants.py
_BASE_ACTION_SCALE = 0.25 * 27.0 / 20.0  # = 0.3375
_ACTUATOR_NAMES = ("LHipJoint", "LUpperJoint", "LLowerJoint1", "RHipJoint", "RUpperJoint", "RLowerJoint1")
_ACTUATOR_STIFFNESS = 20.0
_ACTUATOR_DAMPING = 1.0

_PER_JOINT_SCALE: dict[str, float] = {
    "LHipJoint": _BASE_ACTION_SCALE,
    "LUpperJoint": _BASE_ACTION_SCALE,
    "LLowerJoint1": _BASE_ACTION_SCALE * 0.5,
    "RHipJoint": _BASE_ACTION_SCALE,
    "RUpperJoint": _BASE_ACTION_SCALE,
    "RLowerJoint1": _BASE_ACTION_SCALE * 0.5,
}


def _build_action_scale_array() -> np.ndarray:
    """Build per-joint action_scale array matching RLbipedal."""
    scales = [_PER_JOINT_SCALE.get(n, _BASE_ACTION_SCALE) for n in _ACTUATOR_NAMES]
    return np.array(scales, dtype=get_global_dtype())


# ---------------------------------------------------------------------------
# Helper: contact time tracking
# ---------------------------------------------------------------------------


def _update_contact_times(
    contact: np.ndarray,  # (N, 2) bool
    prev_contact: np.ndarray,  # (N, 2) bool
    contact_time: np.ndarray,  # (N, 2) float
    air_time: np.ndarray,  # (N, 2) float
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Track per-foot contact/air time. Returns (new_contact_time, new_air_time)."""
    ct = contact_time.copy()
    at = air_time.copy()
    ct += dt * contact.astype(get_global_dtype())
    ct *= contact.astype(get_global_dtype())  # reset on lift-off
    at += dt * (1.0 - contact.astype(get_global_dtype()))
    at *= (1.0 - contact.astype(get_global_dtype()))  # reset on touch-down
    return ct, at


# ---------------------------------------------------------------------------
# DR provider
# ---------------------------------------------------------------------------


class MyBipedalDRProvider(LocomotionDRProvider):
    """Domain randomization provider tailored for my_bipedal."""

    def __init__(self):
        super().__init__()

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        return None, None

    def _get_qvel_limit(self, env: Any) -> float:
        return float(env.cfg.reset_base_qvel_limit)

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        updates = {"gait_phase": self._sample_gait_phase(env, num_reset)}
        if env.cfg.heading_command:
            updates["heading_commands"] = sample_heading_commands(env, num_reset)
        return updates

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands)
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.05))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        if env.cfg.heading_command:
            commands[:, 2] = 0.0
        return commands

    def _sample_gait_phase(self, env: Any, num_reset: int) -> np.ndarray:
        mode = env.cfg.gait_phase_init_mode
        if mode == "independent":
            left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())
        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

    def _compute_reset_obs(
        self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel,
    ) -> dict[str, np.ndarray]:
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------


class MyBipedalWalkEnv(LocomotionBaseEnv):
    """Flat-terrain walking environment for my_bipedal robot.

    Direct port of RLbipedal my_bipedal_flat_env_cfg:
    - 20 total joints, 6 actuated (LHipJoint, LUpperJoint, LLowerJoint1, R*)
    - Actor obs: 57D, Critic obs: 72D (includes foot contact info)
    - 22-term reward function matching RLbipedal
    - 3-stage velocity curriculum
    """

    _cfg: MyBipedalWalkEnvCfg
    _keyframe_name: str = "stand"
    _use_global_dtype: bool = False

    def __init__(self, cfg: MyBipedalWalkEnvCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.asset.foot_name,
            motrix_max_iterations=getattr(cfg, "motrix_max_iterations", None),
            post_step_forward_sensor=getattr(cfg, "post_step_forward_sensor", False),
        )
        super().__init__(cfg, backend, num_envs)

        self._reward_cfg = cfg.reward_config
        self._curriculum_cfg = cfg.curriculum

        # Per-joint action scale (LLowerJoint1 scaled 0.5x)
        self._action_scale_joint = _build_action_scale_array()
        joint_range = self._backend.get_joint_range()
        if joint_range is None:
            self._actuated_joint_range = None
        else:
            joint_range = np.asarray(joint_range, dtype=get_global_dtype())
            self._actuated_joint_range = joint_range[self._actuated_dof_pos_indices].copy()
            margin = (self._actuated_joint_range[:, 1] - self._actuated_joint_range[:, 0]) * 0.05
            self._actuated_joint_range[:, 0] += margin
            self._actuated_joint_range[:, 1] -= margin
        self._gait_phase_delta = float(2.0 * math.pi * (1.0 / self._reward_cfg.gait_period) * cfg.ctrl_dt)
        self._enable_reward_log = True

        # Contact tracking buffers
        self._contact_time = np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        self._air_time = np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        self._prev_contact = np.zeros((self._num_envs, 2), dtype=bool)

        # Torque buffer (estimated from position actuator model)
        self._last_torques = np.zeros((self._num_envs, self._num_action), dtype=get_global_dtype())

        # Joint DOF position indices for actuated joints (set in _init_buffers, used here for reference)
        # self._actuated_dof_pos_indices is initialized in _init_buffers before super().__init__

        # Initialize domain randomization
        dr_provider = MyBipedalDRProvider()
        self._init_domain_randomization(dr_provider)

        # Command resampling timer (matches RLbipedal: every 3-8s)
        self._cmd_resample_interval = np.random.uniform(3.0, 8.0, size=self._num_envs).astype(get_global_dtype())
        self._cmd_resample_timer = self._cmd_resample_interval.copy()

        # Curriculum
        self._curriculum_step = 0

    # ── buffer overrides ─────────────────────────────────────────

    def _init_buffers(self) -> None:
        """Store full 20-DOF default positions (not just 6 actuated)."""
        # Compute actuated DOF indices early (needed for default_angles override)
        self._actuated_dof_pos_indices = list(self._backend.get_joint_dof_pos_indices(list(_ACTUATOR_NAMES)))
        super()._init_buffers()
        dtype = get_global_dtype() if self._use_global_dtype else np.float32
        # Full 20-DOF default positions (all joints)
        raw_qpos = self._backend.get_keyframe_qpos(self._keyframe_name)
        self._all_default_angles = np.asarray(raw_qpos[-20:], dtype=dtype)
        # Override default_angles to use correct actuated joint positions
        # (parent class takes _init_qpos[-6:] which is wrong joint indices)
        self.default_angles = self._all_default_angles[self._actuated_dof_pos_indices].copy()

    # ── obs spec ─────────────────────────────────────────────────

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3)+gravity(3)+command(3)+phase(2)+joint_pos(20)+joint_vel(20)+action(6) = 57
        return {"obs": 57, "critic": 72}

    # ── state update ─────────────────────────────────────────────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()  # 20D
        dof_vel = self.get_dof_vel()  # 20D

        # Termination
        max_tilt_rad = math.radians(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1.0, 1.0))
        base_height = self._backend.get_sensor_data("left_foot_pos")[:, 2]  # use as relative
        actual_base_z = self._terrain_relative_base_height()
        fell_over = tilt > max_tilt_rad
        too_low = actual_base_z < self._reward_cfg.base_height_min

        terminated = np.logical_or(fell_over, too_low)
        state.info["_just_terminated"] = terminated.astype(get_global_dtype())

        # ── Periodic command resampling (matching RLbipedal 3-8s) ──
        self._cmd_resample_timer -= self._cfg.ctrl_dt
        need_resample = self._cmd_resample_timer <= 0.0
        if np.any(need_resample):
            self._cmd_resample_timer[need_resample] = np.random.uniform(3.0, 8.0, size=int(np.sum(need_resample)))
            new_cmds = self._sample_commands(int(np.sum(need_resample)))
            state.info["commands"][need_resample] = new_cmds
            # Also update heading commands if needed
            if self._cfg.heading_command:
                from unilab.envs.locomotion.common.commands import sample_heading_commands
                hdg = sample_heading_commands(self, int(np.sum(need_resample)))
                if "heading_commands" in state.info:
                    state.info["heading_commands"][need_resample] = hdg

        # Track foot contact
        left_fc = np.asarray(self._backend.get_sensor_data("left_foot_contact") > 0.5, dtype=bool)
        right_fc = np.asarray(self._backend.get_sensor_data("right_foot_contact") > 0.5, dtype=bool)
        current_contact = np.column_stack([left_fc, right_fc])

        self._contact_time, self._air_time = _update_contact_times(
            current_contact, self._prev_contact, self._contact_time, self._air_time, self._cfg.ctrl_dt
        )
        self._prev_contact = current_contact.copy()

        state.info["feet_contact"] = current_contact.astype(get_global_dtype())
        state.info["feet_contact_time"] = self._contact_time.copy()
        state.info["feet_air_time"] = self._air_time.copy()

        # Estimate actuator torques from position control model
        # tau = kp * (q_des - q) - kd * dq
        last_ctrl = state.info.get("_last_ctrl", self.default_angles.copy())
        dof_pos_act = dof_pos[:, self._actuated_dof_pos_indices]
        dof_vel_act = dof_vel[:, self._actuated_dof_pos_indices]
        torques = _ACTUATOR_STIFFNESS * (last_ctrl - dof_pos_act) - _ACTUATOR_DAMPING * dof_vel_act
        self._last_torques = torques
        state.info["torques"] = torques

        # Foot positions for slip computation
        lf_pos = self._backend.get_sensor_data("left_foot_pos")
        rf_pos = self._backend.get_sensor_data("right_foot_pos")
        prev_lf = state.info.get("_prev_lf_pos", lf_pos)
        prev_rf = state.info.get("_prev_rf_pos", rf_pos)
        lf_vel = (lf_pos - prev_lf) / self._cfg.ctrl_dt
        rf_vel = (rf_pos - prev_rf) / self._cfg.ctrl_dt
        state.info["_prev_lf_pos"] = lf_pos
        state.info["_prev_rf_pos"] = rf_pos
        state.info["foot_vel_l"] = lf_vel
        state.info["foot_vel_r"] = rf_vel

        # Pre-compute critic foot privileges for _compute_obs (avoids
        # backend calls during DR reset where env_ids are sliced).
        foot_height_l = lf_pos[:, 2:3]
        foot_height_r = rf_pos[:, 2:3]
        state.info["_critic_foot_height"] = np.concatenate([foot_height_l, foot_height_r], axis=1)

        lf_raw = self._backend.get_sensor_data("left_foot_force")
        rf_raw = self._backend.get_sensor_data("right_foot_force")
        if lf_raw.ndim == 1:
            lf_raw = lf_raw[:, None]
        if rf_raw.ndim == 1:
            rf_raw = rf_raw[:, None]
        raw_forces = np.concatenate([lf_raw, rf_raw], axis=1).reshape(self._num_envs, -1)
        state.info["_critic_foot_forces"] = np.sign(raw_forces) * np.log(np.abs(raw_forces) + 1.0)

        # Curriculum: update command ranges
        if self._curriculum_cfg.enabled:
            self._curriculum_step += 1
            cmd_info = self._get_curriculum_command_range()
            state.info["curriculum_cmd_range"] = cmd_info

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel, terminated)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)

        # ── Logging for PPO training diagnostics ──
        if self._enable_reward_log and self.step_counter % 24 == 0:
            log_dict = state.info.get("_reward_log", {})
            log_dict["metric_base_z"] = float(np.mean(actual_base_z))
            log_dict["termination_rate"] = float(np.mean(terminated))
            log_dict["mean_linvel_x"] = float(np.mean(linvel[:, 0]))
            log_dict["mean_linvel_y"] = float(np.mean(linvel[:, 1]))
            log_dict["mean_cmd_x"] = float(np.mean(state.info["commands"][:, 0]))
            log_dict["mean_cmd_y"] = float(np.mean(state.info["commands"][:, 1]))
            state.info["log"] = log_dict

        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _terrain_relative_base_height(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())

    def _sample_commands(self, n: int) -> np.ndarray:
        """Sample n commands within the current curriculum stage range."""
        rng = self._get_curriculum_command_range()
        cmds = np.zeros((n, 3), dtype=get_global_dtype())
        cmds[:, 0] = np.random.uniform(*rng["lin_vel_x"], size=n)
        cmds[:, 1] = np.random.uniform(*rng["lin_vel_y"], size=n)
        cmds[:, 2] = np.random.uniform(*rng["ang_vel_z"], size=n)
        return cmds

    def _get_curriculum_command_range(self) -> dict:
        """Get current command range based on curriculum step."""
        stages = self._curriculum_cfg.velocity_stages
        step = self._curriculum_step
        current_stage = stages[0]
        for s in stages:
            if step >= s["step"]:
                current_stage = s
        return {
            "lin_vel_x": current_stage.get("lin_vel_x", (-1.0, 2.0)),
            "lin_vel_y": current_stage.get("lin_vel_y", (-1.0, 1.0)),
            "ang_vel_z": current_stage.get("ang_vel_z", (-1.0, 1.0)),
        }

    # ── observation ───────────────────────────────────────────────

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self._all_default_angles  # 20D
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros((self._num_envs, self._num_action), dtype=get_global_dtype()))
        gait_phase = info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))

        # Apply observation noise (matching RLbipedal)
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)

        # Actor observation (57D)
        actor = np.concatenate(
            [
                noisy_gyro,
                -noisy_gravity,  # projected gravity
                noisy_diff,       # 20D joint pos rel
                noisy_dof_vel,    # 20D joint vel
                last_actions,     # 6D
                command,          # 3D
                gait_phase,       # 2D
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        # Critic observation (72D = actor-like + linvel + foot privileges)
        # Use batch size from gyro (which is 1 during reset, num_envs during step).
        batch = gyro.shape[0]
        foot_height = info.get("_critic_foot_height", np.zeros((batch, 2), dtype=get_global_dtype()))
        foot_contact = info.get("feet_contact", np.zeros((batch, 2), dtype=get_global_dtype()))
        foot_air_time = info.get("feet_air_time", np.zeros((batch, 2), dtype=get_global_dtype()))
        contact_forces = info.get("_critic_foot_forces", np.zeros((batch, 6), dtype=get_global_dtype()))

        critic_base = np.concatenate(
            [
                gyro, -gravity, diff, dof_vel, last_actions, command, gait_phase,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [
                critic_base,
                np.asarray(linvel, dtype=get_global_dtype()),
                foot_height,
                foot_air_time,
                foot_contact,
                contact_forces,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        return {"obs": actor, "critic": critic}

    # ── reward ────────────────────────────────────────────────────

    def _compute_reward(
        self,
        info: dict,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        terminated: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute combined reward matching RLbipedal my_bipedal_flat_env_cfg."""
        dtype = get_global_dtype()
        s = self._reward_cfg.scales
        reward = np.zeros((self._num_envs,), dtype=dtype)
        commands = info.get("commands", np.zeros((self._num_envs, 3), dtype=dtype))
        reward_log = {} if self._enable_reward_log else None

        # Core tracking
        def _add(key, val):
            nonlocal reward
            w = s[key]
            weighted = w * val
            reward += weighted
            if reward_log is not None:
                reward_log[key] = float(np.mean(weighted))

        if s.get("track_lin_vel_xy", 0) != 0:
            _add("track_lin_vel_xy", self._r_track_lin_vel_xy(commands, linvel))
        if s.get("track_lin_vel_y", 0) != 0:
            _add("track_lin_vel_y", self._r_track_lin_vel_y(commands, linvel))
        if s.get("track_ang_vel", 0) != 0:
            _add("track_ang_vel", self._r_track_ang_vel(commands, gyro))

        # Orientation / stability
        if s.get("body_orientation_l2", 0) != 0:
            _add("body_orientation_l2", self._r_body_orientation_l2(gravity))
        if s.get("body_ang_vel", 0) != 0:
            _add("body_ang_vel", self._r_body_ang_vel(gyro))
        if s.get("flat_orientation", 0) != 0:
            _add("flat_orientation", self._r_flat_orientation(gravity))
        if s.get("angular_momentum", 0) != 0:
            _add("angular_momentum", self._r_angular_momentum())

        # Gait
        if s.get("foot_gait", 0) != 0:
            _add("foot_gait", self._r_foot_gait(info, commands))
        if s.get("symmetry", 0) != 0:
            _add("symmetry", self._r_symmetry(dof_pos))

        # Energy / effort
        if s.get("torque_penalty", 0) != 0:
            _add("torque_penalty", self._r_torque_penalty())
        if s.get("action_rate_l2", 0) != 0:
            _add("action_rate_l2", self._r_action_rate_l2(info))
        if s.get("joint_acc_l2", 0) != 0:
            _add("joint_acc_l2", self._r_joint_acc_l2(info))

        # Pose
        if s.get("pose", 0) != 0:
            _add("pose", self._r_pose(dof_pos, commands))

        # Foot constraints
        if s.get("foot_clearance", 0) != 0:
            _add("foot_clearance", self._r_foot_clearance(commands))
        if s.get("foot_slip", 0) != 0:
            _add("foot_slip", self._r_foot_slip(info, commands))
        if s.get("soft_landing", 0) != 0:
            _add("soft_landing", self._r_soft_landing(info, commands))

        # Height
        if s.get("base_height", 0) != 0:
            _add("base_height", self._r_base_height_soft_bounds())
        if s.get("phase_height", 0) != 0:
            _add("phase_height", self._r_phase_height_oscillation(info))

        # Constraints
        if s.get("joint_pos_limits", 0) != 0:
            _add("joint_pos_limits", self._r_joint_pos_limits(dof_pos))
        if s.get("stand_still", 0) != 0:
            _add("stand_still", self._r_stand_still(dof_pos, commands))
        if s.get("is_terminated", 0) != 0:
            _add("is_terminated", self._r_is_terminated(terminated))

        if reward_log is not None:
            info["_reward_log"] = reward_log

        return reward * self._cfg.ctrl_dt

    # ── individual reward terms ───────────────────────────────────

    def _r_track_lin_vel_xy(self, commands: np.ndarray, linvel: np.ndarray) -> np.ndarray:
        """Reward for tracking xy velocity only (no z penalty)."""
        std = self._reward_cfg.tracking_sigma
        xy_err = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
        return np.exp(-xy_err / (std * std))

    def _r_track_lin_vel_y(self, commands: np.ndarray, linvel: np.ndarray) -> np.ndarray:
        """Extra penalty for y-direction (lateral) velocity error."""
        std = 0.3
        y_err = np.square(commands[:, 1] - linvel[:, 1])
        return np.exp(-y_err / (std * std))

    def _r_track_ang_vel(self, commands: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Exponential reward for angular velocity tracking."""
        std = self._reward_cfg.tracking_ang_sigma
        err = np.square(commands[:, 2] - gyro[:, 2])
        return np.exp(-err / (std * std))

    def _r_body_orientation_l2(self, gravity: np.ndarray) -> np.ndarray:
        """Penalty for deviation from upright (roll/pitch squared)."""
        return np.sum(np.square(gravity[:, :2]), axis=1)  # noqa

    def _r_body_ang_vel(self, gyro: np.ndarray) -> np.ndarray:
        """Penalty for roll/pitch angular velocity (x, y components)."""
        return np.sum(np.square(gyro[:, :2]), axis=1)  # noqa

    def _r_flat_orientation(self, gravity: np.ndarray) -> np.ndarray:
        """Direct pitch/roll penalty from projected gravity."""
        return np.sum(np.square(gravity[:, :2]), axis=1)  # noqa

    def _r_angular_momentum(self) -> np.ndarray:
        """Penalty for angular momentum (if sensor exists)."""
        try:
            am = self._backend.get_sensor_data("root_angmom")
            return np.sum(np.square(am), axis=1)  # noqa
        except Exception:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())

    def _r_foot_gait(self, info: dict, commands: np.ndarray) -> np.ndarray:
        """Clock-based foot gait tracking (period=0.6, offset [0, 0.5])."""
        phase = info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        contact = info.get("feet_contact", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        period = 2.0 * math.pi
        expected_l = (phase[:, 0] % period) < period * 0.56
        expected_r = (phase[:, 1] % period) < period * 0.56
        match_l = (contact[:, 0] > 0.5) == expected_l
        match_r = (contact[:, 1] > 0.5) == expected_r
        reward = (match_l.astype(get_global_dtype()) + match_r.astype(get_global_dtype())) * 0.5
        moving = np.linalg.norm(commands[:, :2], axis=1) > self._reward_cfg.command_threshold
        return reward * moving

    def _r_symmetry(self, dof_pos: np.ndarray) -> np.ndarray:
        """L/R joint amplitude symmetry penalty."""
        dtype = get_global_dtype()
        dev = np.zeros((self._num_envs,), dtype=dtype)
        all_diff = dof_pos - self._all_default_angles
        # Map actuated joint names to their DOF position indices
        indices = {}
        for name in _ACTUATOR_NAMES:
            idx_list = self._backend.get_joint_dof_pos_indices([name])
            indices[name] = int(idx_list[0])
        pairs = [("LHipJoint", "RHipJoint"), ("LUpperJoint", "RUpperJoint"), ("LLowerJoint1", "RLowerJoint1")]
        for ln, rn in pairs:
            if ln in indices and rn in indices:
                li = indices[ln]
                ri = indices[rn]
                lm = np.abs(all_diff[:, li])
                rm = np.abs(all_diff[:, ri])
                dev += np.square(lm - rm)
        return dev

    def _r_torque_penalty(self) -> np.ndarray:
        """Penalize mean absolute actuator torque (estimated from position control)."""
        return np.mean(np.abs(self._last_torques), axis=1)

    def _r_action_rate_l2(self, info: dict) -> np.ndarray:
        """Penalty for change in actions between timesteps."""
        current = info.get("current_actions", np.zeros((self._num_envs, self._num_action), dtype=get_global_dtype()))
        last = info.get("last_actions", np.zeros((self._num_envs, self._num_action), dtype=get_global_dtype()))
        return np.sum(np.square(current - last), axis=1)

    def _r_joint_acc_l2(self, info: dict) -> np.ndarray:
        """Penalty for joint acceleration."""
        qacc = info.get("qacc", np.zeros((self._num_envs, 20), dtype=get_global_dtype()))
        return np.sum(np.square(qacc), axis=1)

    def _r_pose(self, dof_pos: np.ndarray, commands: np.ndarray) -> np.ndarray:
        """Variable-posture regularization with speed-dependent std."""
        dtype = get_global_dtype()
        diff = dof_pos - self._all_default_angles
        speed = np.linalg.norm(commands[:, :2], axis=1)
        walking = speed > self._reward_cfg.walking_threshold
        running = speed > self._reward_cfg.running_threshold
        # We don't have a good way to build per-joint std here without name mapping
        # Use uniform std for simplicity (RLbipedal's variable_posture uses regex)
        std_stand = np.full(diff.shape[1], 0.05, dtype=dtype)  # std_standing
        std_walk = np.full(diff.shape[1], 0.4, dtype=dtype)  # relaxed
        std_run = np.full(diff.shape[1], 0.5, dtype=dtype)
        r_stand = np.exp(-np.sum(np.square(diff) / np.square(std_stand), axis=1))
        r_walk = np.exp(-np.sum(np.square(diff) / np.square(std_walk), axis=1))
        r_run = np.exp(-np.sum(np.square(diff) / np.square(std_run), axis=1))
        result = np.where(running, r_run, np.where(walking, r_walk, r_stand))
        return result

    def _r_foot_clearance(self, commands: np.ndarray) -> np.ndarray:
        """Penalty for foot clearance below target during swing."""
        dtype = get_global_dtype()
        target = self._reward_cfg.foot_clearance_target
        moving = np.linalg.norm(commands[:, :2], axis=1) > self._reward_cfg.command_threshold
        l_z = self._backend.get_sensor_data("left_foot_pos")[:, 2]
        r_z = self._backend.get_sensor_data("right_foot_pos")[:, 2]
        l_err = np.clip(target - l_z, 0.0, None)
        r_err = np.clip(target - r_z, 0.0, None)
        return (np.square(l_err) + np.square(r_err)) * moving

    def _r_foot_slip(self, info: dict, commands: np.ndarray) -> np.ndarray:
        """Penalty for foot slipping (xy velocity of foot during contact)."""
        dtype = get_global_dtype()
        moving = np.linalg.norm(commands[:, :2], axis=1) > self._reward_cfg.command_threshold
        contact = info.get("feet_contact", np.zeros((self._num_envs, 2), dtype=dtype))
        l_vel = info.get("foot_vel_l", np.zeros((self._num_envs, 3), dtype=dtype))
        r_vel = info.get("foot_vel_r", np.zeros((self._num_envs, 3), dtype=dtype))
        l_slip = np.sum(np.square(l_vel[:, :2]), axis=1) * contact[:, 0]
        r_slip = np.sum(np.square(r_vel[:, :2]), axis=1) * contact[:, 1]
        return (l_slip + r_slip) * moving

    def _r_soft_landing(self, info: dict, commands: np.ndarray) -> np.ndarray:
        """Penalty for impact force at foot touchdown."""
        dtype = get_global_dtype()
        moving = np.linalg.norm(commands[:, :2], axis=1) > self._reward_cfg.command_threshold
        contact = info.get("feet_contact", np.zeros((self._num_envs, 2), dtype=dtype))
        prev = self._prev_contact.astype(dtype)
        touch = np.maximum(contact - prev, 0.0)
        lf = self._backend.get_sensor_data("left_foot_force")
        rf = self._backend.get_sensor_data("right_foot_force")
        if lf.ndim == 1:
            lf = lf[:, None]
        if rf.ndim == 1:
            rf = rf[:, None]
        lf_sum = np.sum(np.abs(lf), axis=1) * touch[:, 0]
        rf_sum = np.sum(np.abs(rf), axis=1) * touch[:, 1]
        return (lf_sum + rf_sum) * moving

    def _r_base_height_soft_bounds(self) -> np.ndarray:
        """Soft bounds on base height: [0.50, 1.0], reward=0 inside."""
        h = self._backend.get_base_pos()[:, 2]
        lo = np.square(np.clip(self._reward_cfg.base_height_min - h, 0.0, None))
        hi = np.square(np.clip(h - self._reward_cfg.base_height_max, 0.0, None))
        return -(lo + hi)

    def _r_phase_height_oscillation(self, info: dict) -> np.ndarray:
        """Contact-driven height oscillation reward."""
        h = self._backend.get_base_pos()[:, 2]
        contact = info.get("feet_contact", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        frac = np.mean(contact, axis=1)
        phase_angle = frac * 2.0 * math.pi
        target_z = self._reward_cfg.phase_height_base + self._reward_cfg.phase_height_amp * np.cos(phase_angle)
        return -np.square(h - target_z)

    def _r_joint_pos_limits(self, dof_pos: np.ndarray) -> np.ndarray:
        """Penalty for joints approaching their limits (using actual joint range)."""
        if self._actuated_joint_range is None:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        actuated_pos = dof_pos[:, self._actuated_dof_pos_indices]
        low_err = np.clip(self._actuated_joint_range[:, 0] - actuated_pos, 0.0, None)
        high_err = np.clip(actuated_pos - self._actuated_joint_range[:, 1], 0.0, None)
        return np.sum(low_err + high_err, axis=1)

    def _r_stand_still(self, dof_pos: np.ndarray, commands: np.ndarray) -> np.ndarray:
        """Penalty for joint deviation while stopped."""
        stopped = np.linalg.norm(commands[:, :2], axis=1) <= self._reward_cfg.command_threshold
        actuated_pos = dof_pos[:, self._actuated_dof_pos_indices]
        diff = np.sum(np.abs(actuated_pos - self.default_angles), axis=1)
        return diff * stopped.astype(get_global_dtype())

    def _r_is_terminated(self, terminated: np.ndarray | None = None) -> np.ndarray:
        """Termination penalty: 1.0 if just terminated, 0.0 otherwise."""
        if terminated is None:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        return np.asarray(terminated, dtype=get_global_dtype())

    # ── action ────────────────────────────────────────────────────

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        # Advance gait phase
        gait_phase = state.info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype()))
        gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * math.pi)
        gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * math.pi)
        state.info["gait_phase"] = gait_phase

        # Per-joint action scale
        ctrl: np.ndarray = actions * self._action_scale_joint + self.default_angles
        state.info["_last_ctrl"] = ctrl
        return ctrl


# ---------------------------------------------------------------------------
# Registered configs
# ---------------------------------------------------------------------------


@registry.envcfg("MyBipedalWalkFlat")
@dataclass
class MyBipedalWalkFlatCfg(MyBipedalWalkEnvCfg):
    """Flat-terrain walking config for MyBipedal."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "RLbipedal" / "scene_flat.xml")
        )
    )
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    heading_command: bool = True


registry.register_env("MyBipedalWalkFlat", MyBipedalWalkEnv, sim_backend="mujoco")
