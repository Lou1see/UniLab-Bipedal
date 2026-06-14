"""Quick script to inspect G1WalkFlat joint actuator torques via jointactuatorfrc sensors.

Usage:
    uv run scripts/check_g1_torque.py
"""

from __future__ import annotations

import numpy as np
from unilab.training import ensure_registries
from unilab.base import registry

# ── joint names for G1, grouped logically ────────────────────────────
LEFT_LEG = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEG = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
TORSO = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
LEFT_ARM = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ALL_JOINTS = LEFT_LEG + RIGHT_LEG + TORSO + LEFT_ARM + RIGHT_ARM

# XML sensor names drop "_joint": e.g. "left_knee_joint" → "left_knee_torque"
def _joint_to_torque_sensor(joint_name: str) -> str:
    return joint_name.replace("_joint", "") + "_torque"

TORQUE_SENSORS = [_joint_to_torque_sensor(j) for j in ALL_JOINTS]


def main() -> None:
    ensure_registries()

    # ── env config override ──────────────────────────────────────────
    env_cfg_override = {
        "reward_config": {
            "scales": {"tracking_lin_vel": 1.0},
            "tracking_sigma": 0.25,
            "gait_frequency": 1.5,
            "feet_phase_swing_height": 0.09,
            "feet_phase_tracking_sigma": 0.008,
            "base_height_target": 0.754,
            "min_base_height": 0.55,
            "max_tilt_deg": 25.0,
            "pose_weights": [0.01] * 29,
        },
        "curriculum": {"enabled": False},
    }

    env = registry.make(
        "G1WalkFlat",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=env_cfg_override,
    )

    backend = env._backend
    state = env.init_state()

    # ── step a few times with zero action ────────────────────────────
    n_dof = env.action_space.shape[0]
    n_steps = 50
    print(f"Stepping {n_steps} times with zero action ...")
    for _ in range(n_steps):
        state = env.step(np.zeros((1, n_dof)))

    # ── read joint torques ───────────────────────────────────────────
    def print_group(title: str, joints: list[str]) -> None:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"{'Joint':<35} {'Torque (N·m)':>15}")
        print(f"{'-'*50}")
        for j in joints:
            sensor_name = _joint_to_torque_sensor(j)
            val = backend.get_sensor_data(sensor_name)
            # val shape: (1, 1) — scalar
            print(f"  {sensor_name:<35} {val[0, 0]:>15.4f}")

    print_group("LEFT LEG", LEFT_LEG)
    print_group("RIGHT LEG", RIGHT_LEG)
    print_group("TORSO", TORSO)
    print_group("LEFT ARM", LEFT_ARM)
    print_group("RIGHT ARM", RIGHT_ARM)

    # ── quick summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    all_vals = []
    for sn in TORQUE_SENSORS:
        val = float(backend.get_sensor_data(sn)[0, 0])
        all_vals.append(val)
    all_vals = np.array(all_vals)
    print(f"  Max torque:  {np.max(np.abs(all_vals)):.4f} N·m")
    print(f"  Mean torque: {np.mean(np.abs(all_vals)):.4f} N·m")
    print(f"  Min torque:  {np.min(np.abs(all_vals)):.4f} N·m")

    env.close()


if __name__ == "__main__":
    main()
