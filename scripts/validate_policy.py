"""Validate a SAC checkpoint: run the policy in the env and print per-step
base_z, linvel_x, cmd_x, and whether the robot is walking or kneeling.

Usage:
    uv run python scripts/validate_policy.py <checkpoint_path> [--steps 200] [--num-envs 16]
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import argparse
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", type=str, default="MyBipedalWalkFlat")
    parser.add_argument("--sim", type=str, default="mujoco")
    args = parser.parse_args()

    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from unilab.base.observations import split_obs_dict
    from unilab.algos.torch.common.actor_factory import build_actor

    ensure_registries()

    env = registry.make(args.task, num_envs=args.num_envs, sim_backend=args.sim)
    env.init_state()

    obs_dim = env.obs_groups_spec["obs"]
    action_dim = int(env.action_space.shape[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = build_actor(
        "sac", obs_dim, action_dim, 512, True, device,
        log_std_min=-2.0, log_std_max=-0.5,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_dict, info = env.reset(np.arange(args.num_envs, dtype=np.int32))

    # Force a forward velocity command to test if robot can walk
    if "commands" in info:
        info["commands"][:, 0] = 1.0   # vx = 1.0 m/s forward
        info["commands"][:, 1] = 0.0   # vy = 0
        info["commands"][:, 2] = 0.0   # wz = 0
        print(f"Forced command: vx=1.0, vy=0.0, wz=0.0")

    obs_np, _ = split_obs_dict(obs_dict)
    obs_np = np.asarray(obs_np, dtype=np.float32)

    base_z_history = []
    linvel_x_history = []
    cmd_x_history = []
    terminated_count = 0

    with torch.no_grad():
        for step in range(args.steps):
            obs_t = torch.from_numpy(obs_np).to(device)
            actions = actor.explore(obs_t, deterministic=True).cpu().numpy()
            state = env.step(actions)
            obs_np, _ = split_obs_dict(state.obs)
            obs_np = np.asarray(obs_np, dtype=np.float32)

            base_z = env._backend.get_base_pos()[:, 2]
            linvel = env.get_local_linvel()
            commands = state.info.get("commands", np.zeros((args.num_envs, 3)))
            # Re-force command after resampling
            state.info["commands"][:, 0] = 1.0
            state.info["commands"][:, 1] = 0.0
            state.info["commands"][:, 2] = 0.0

            base_z_history.append(float(np.mean(base_z)))
            linvel_x_history.append(float(np.mean(linvel[:, 0])))
            cmd_x_history.append(float(np.mean(commands[:, 0])))
            terminated_count += int(np.sum(state.terminated))

    env.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Steps: {args.steps}, Envs: {args.num_envs}")
    print(f"{'='*60}")
    print(f"Terminations: {terminated_count} / {args.steps * args.num_envs}")
    print(f"\nBase Z (mean over envs):")
    print(f"  mean: {np.mean(base_z_history):.3f}  min: {np.min(base_z_history):.3f}  max: {np.max(base_z_history):.3f}")
    print(f"  first 10: {[f'{z:.3f}' for z in base_z_history[:10]]}")
    print(f"  last 10:  {[f'{z:.3f}' for z in base_z_history[-10:]]}")
    print(f"\nLinvel X (mean over envs):")
    print(f"  mean: {np.mean(linvel_x_history):.3f}  min: {np.min(linvel_x_history):.3f}  max: {np.max(linvel_x_history):.3f}")
    print(f"  last 10: {[f'{v:.3f}' for v in linvel_x_history[-10:]]}")
    print(f"\nCommand X (mean over envs):")
    print(f"  mean: {np.mean(cmd_x_history):.3f}")
    print(f"  last 10: {[f'{v:.3f}' for v in cmd_x_history[-10:]]}")

    # Verdict
    mean_base_z = np.mean(base_z_history[-50:])
    mean_linvel = np.mean(linvel_x_history[-50:])
    mean_cmd = np.mean(cmd_x_history[-50:])
    print(f"\n{'='*60}")
    print(f"VERDICT (last 50 steps):")
    print(f"  base_z={mean_base_z:.3f}  linvel_x={mean_linvel:.3f}  cmd_x={mean_cmd:.3f}")
    if mean_base_z < 0.50:
        print(f"  ❌ Robot is KNEELING (base_z < 0.50)")
    elif mean_base_z > 0.60:
        print(f"  ✅ Robot is STANDING (base_z > 0.60)")
    else:
        print(f"  ⚠️  Robot is between kneeling and standing (0.50-0.60)")

    if abs(mean_cmd) > 0.1:
        if abs(mean_linvel - mean_cmd) < 0.3:
            print(f"  ✅ Robot is TRACKING velocity command ({mean_linvel:.2f} vs {mean_cmd:.2f})")
        else:
            print(f"  ❌ Robot is NOT tracking velocity ({mean_linvel:.2f} vs {mean_cmd:.2f})")
    else:
        print(f"  (command near zero, standing still expected)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
