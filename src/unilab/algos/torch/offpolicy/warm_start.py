"""Warm-start replay buffer prefill for off-policy RL.

Runs a simple standing controller (small noise around the default pose) to
collect transitions that survive longer than a random untrained policy, then
writes them into the shared ``ReplayBuffer`` before normal collection starts.

This does **not** bypass the runner lifecycle: it creates its own env via the
registry, steps it through the standard ``NpEnv`` contract, and writes to the
same ``ReplayBuffer`` the collector would use.  It runs in the learner process,
before the collector subprocess is spawned.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.observations import split_obs_dict


@dataclass
class WarmStartConfig:
    """Owner-config for warm-start replay prefill."""

    enabled: bool = False
    source: str = "standing_controller"
    steps: int = 100_000
    noise_std: float = 0.15
    action_hold: int = 4
    learning_after_warm_start: bool = True
    # When set, overrides env reward_config.base_height_min during warm-start
    # to delay termination and produce longer episodes.  The normal collector
    # still uses the original threshold, so the learner sees a mix.
    relax_termination_height: float | None = None
    # For source="sac_checkpoint": path to a SAC checkpoint .pt file whose
    # actor weights are loaded to drive the warm-start rollout.
    checkpoint_path: str | None = None
    # For source="sac_checkpoint": actor hidden dim / layer norm / log_std
    # must match the checkpoint.  These are read from the runner config.
    actor_hidden_dim: int = 512
    use_layer_norm: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 0.0


class StandingController:
    """Conservative standing controller: small Gaussian noise around zero action.

    Since ``apply_action`` computes ``ctrl = actions * action_scale +
    default_angles``, an action of zero holds the default (keyframe) pose.
    Adding small noise perturbs the robot gently without immediately
    destabilising it, producing episodes longer than the ~10-step falls caused
    by a random untrained SAC actor.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        noise_std: float = 0.15,
        action_hold: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self._num_envs = num_envs
        self._action_dim = action_dim
        self._noise_std = float(noise_std)
        self._action_hold = max(1, int(action_hold))
        self._rng = rng or np.random.default_rng()
        self._held = np.zeros((num_envs, action_dim), dtype=np.float32)
        self._hold_counter = 0

    def sample(self) -> np.ndarray:
        """Return the next action batch ``(num_envs, action_dim)``."""
        if self._hold_counter <= 0:
            self._held = (
                self._rng.standard_normal((self._num_envs, self._action_dim))
                .astype(np.float32)
                * self._noise_std
            )
            self._hold_counter = self._action_hold
        self._hold_counter -= 1
        return self._held.copy()


def run_warm_start(
    *,
    env_name: str,
    num_envs: int,
    sim_backend: str,
    env_cfg_override: dict[str, Any] | None,
    replay_buffer,
    config: WarmStartConfig | dict[str, Any] | None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Prefill ``replay_buffer`` with standing-controller transitions.

    Creates a temporary env, runs the standing controller for ``config.steps``
    transitions, and writes every step into ``replay_buffer`` using the same
    ``add()`` contract as the normal collector.

    Returns a stats dict with ``total_steps``, ``mean_ep_length``,
    ``buffer_size``, and ``terminated_rate``.
    """
    if config is None:
        return {"total_steps": 0, "mean_ep_length": 0.0, "buffer_size": 0, "terminated_rate": 0.0}
    if isinstance(config, dict):
        config = WarmStartConfig(**config)
    if not config.enabled:
        return {"total_steps": 0, "mean_ep_length": 0.0, "buffer_size": 0, "terminated_rate": 0.0}
    if config.source not in ("standing_controller", "sac_checkpoint"):
        raise NotImplementedError(
            f"warm_start source={config.source!r} is not supported; "
            "only 'standing_controller' and 'sac_checkpoint' are implemented"
        )

    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    # Merge relax_termination_height into env_cfg_override so the warm-start
    # env survives longer before hitting the base_height_min termination.
    warm_env_cfg_override = dict(env_cfg_override) if env_cfg_override else {}
    if config.relax_termination_height is not None:
        rc_override = dict(warm_env_cfg_override.get("reward_config", {}))
        rc_override["base_height_min"] = float(config.relax_termination_height)
        warm_env_cfg_override["reward_config"] = rc_override

    print(
        f"[WarmStart] Prefilling replay buffer with standing_controller: "
        f"steps={config.steps}, noise_std={config.noise_std}, "
        f"action_hold={config.action_hold}, num_envs={num_envs}"
        + (
            f", relax_termination_height={config.relax_termination_height}"
            if config.relax_termination_height is not None
            else ""
        )
    )

    env = registry.make(
        env_name,
        num_envs=num_envs,
        sim_backend=sim_backend,
        env_cfg_override=warm_env_cfg_override or None,
    )
    try:
        if env.state is None:
            env.init_state()

        assert env.action_space.shape is not None
        action_dim = int(env.action_space.shape[0])
        from unilab.base.observations import get_obs_dims as _get_obs_dims

        obs_dim, _ = _get_obs_dims(env.obs_groups_spec)

        controller = None
        checkpoint_actor = None
        if config.source == "sac_checkpoint":
            if not config.checkpoint_path:
                raise ValueError("warm_start source='sac_checkpoint' requires checkpoint_path")
            import torch

            from unilab.algos.torch.common.actor_factory import build_actor

            ckpt = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
            checkpoint_actor = build_actor(
                "sac",
                obs_dim,
                action_dim,
                config.actor_hidden_dim,
                config.use_layer_norm,
                "cpu",
                log_std_min=config.log_std_min,
                log_std_max=config.log_std_max,
            )
            checkpoint_actor.load_state_dict(ckpt["actor"])
            checkpoint_actor.eval()
            print(
                f"[WarmStart] Loaded SAC checkpoint actor from {config.checkpoint_path}"
            )
        else:
            controller = StandingController(
                num_envs=num_envs,
                action_dim=action_dim,
                noise_std=config.noise_std,
                action_hold=config.action_hold,
                rng=rng,
            )

        # Prime the env with one zero-action step to get the first observation,
        # mirroring the collector's bootstrap (worker.py:370-376).
        actions_np = np.zeros((num_envs, action_dim), dtype=np.float32)
        state = env.step(actions_np)
        obs_np, critic_np = split_obs_dict(state.obs)
        obs_np = np.asarray(obs_np, dtype=np.float32)
        critic_np = np.asarray(critic_np, dtype=np.float32)

        target_transitions = int(config.steps)
        num_iters = max(1, target_transitions // num_envs)

        total_steps = 0
        ep_lengths: list[int] = []
        current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
        done_count = 0
        terminated_count = 0

        warm_start_t0 = time.time()
        for i in range(num_iters):
            if checkpoint_actor is not None:
                import torch as _torch

                with _torch.no_grad():
                    obs_t = _torch.from_numpy(obs_np)
                    actions_t = checkpoint_actor.explore(obs_t, deterministic=False)
                    actions_np = actions_t.numpy()
            else:
                actions_np = controller.sample()

            state = env.step(actions_np)
            next_obs_np, next_critic_np = split_obs_dict(state.obs)
            next_obs_np = np.asarray(next_obs_np, dtype=np.float32)
            next_critic_np = np.asarray(next_critic_np, dtype=np.float32)
            rewards_np = np.asarray(state.reward, dtype=np.float32).ravel()

            terminated_np = state.terminated.astype(np.float32, copy=False).ravel()
            truncated_np = state.truncated.astype(np.float32, copy=False).ravel()
            combined_dones = (
                (state.terminated | state.truncated).astype(np.float32, copy=False).ravel()
            )
            done_mask_np = combined_dones > 0.5
            timeout_mask_np = truncated_np > 0.5
            terminated_mask_np = np.logical_and(done_mask_np, ~timeout_mask_np)

            done_count += int(np.count_nonzero(done_mask_np))
            terminated_count += int(np.count_nonzero(terminated_mask_np))

            terminal_contract = resolve_terminal_observation_contract(
                next_obs_batch_size=next_obs_np.shape[0],
                final_observation=state.final_observation,
                done=done_mask_np,
                info=state.info,
                truncated=truncated_np,
            )

            # Write to replay buffer using the exact same contract as the
            # collector (worker.py:492-513).
            replay_buffer.add(
                torch.from_numpy(obs_np),
                torch.from_numpy(actions_np),
                torch.from_numpy(rewards_np),
                torch.from_numpy(next_obs_np),
                torch.from_numpy(combined_dones),
                torch.from_numpy(truncated_np),
                terminal_mask=torch.from_numpy(terminal_contract.terminal_mask),
                terminal_next_obs=(
                    torch.from_numpy(terminal_contract.terminal_obs)
                    if terminal_contract.terminal_obs is not None
                    else None
                ),
                critic=torch.from_numpy(critic_np),
                next_critic=torch.from_numpy(next_critic_np),
                terminal_next_critic=(
                    torch.from_numpy(terminal_contract.terminal_critic)
                    if terminal_contract.terminal_critic is not None
                    else None
                ),
            )

            # Track episode lengths (vectorised).
            current_ep_lengths += 1
            reset_mask = combined_dones > 0.5
            reset_indices = np.where(reset_mask)[0]
            if len(reset_indices) > 0:
                ep_lengths.extend(current_ep_lengths[reset_indices].tolist())
                current_ep_lengths[reset_indices] = 0

            obs_np = next_obs_np
            critic_np = next_critic_np
            total_steps += num_envs

            if (i + 1) % 50 == 0 or i == 0:
                mean_ep_len = statistics.mean(ep_lengths[-100:]) if ep_lengths else 0.0
                elapsed = time.time() - warm_start_t0
                print(
                    f"[WarmStart] iter {i + 1}/{num_iters} | "
                    f"steps={total_steps}/{target_transitions} | "
                    f"buf={int(replay_buffer.size[0])} | "
                    f"ep_len={mean_ep_len:.1f} | "
                    f"eps={total_steps / max(elapsed, 1e-6):.0f} "
                    f"({elapsed:.1f}s)"
                )
    finally:
        env.close()

    mean_ep_len = statistics.mean(ep_lengths[-100:]) if ep_lengths else 0.0
    terminated_rate = terminated_count / max(done_count, 1)
    elapsed = time.time() - warm_start_t0
    print(
        f"[WarmStart] Done: total_steps={total_steps}, "
        f"buf_size={int(replay_buffer.size[0])}, "
        f"mean_ep_length={mean_ep_len:.1f}, "
        f"terminated_rate={terminated_rate:.3f}, "
        f"elapsed={elapsed:.1f}s"
    )
    return {
        "total_steps": total_steps,
        "mean_ep_length": float(mean_ep_len),
        "buffer_size": int(replay_buffer.size[0]),
        "terminated_rate": float(terminated_rate),
    }
