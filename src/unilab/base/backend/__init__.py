from typing import Any, cast

from unilab.base.scene import SceneCfg

from .base import (
    BackendHeightScanner,
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    SimBackend,
    normalize_play_render_mode,
)


def _load_mujoco_backend() -> Any:
    from .mujoco.backend import MuJoCoBackend

    return MuJoCoBackend


def _load_motrix_backend() -> tuple[Any, bool]:
    from .motrix.backend import MOTRIX_AVAILABLE, MotrixBackend

    return MotrixBackend, bool(MOTRIX_AVAILABLE)


def create_backend(
    backend_type: str,
    scene: SceneCfg,
    num_envs: int,
    sim_dt: float,
    **kwargs,
) -> SimBackend:
    """Create a simulation backend from the scene/config contract."""
    if scene is None:
        raise ValueError("SceneCfg must be provided")

    position_actuator_gains = kwargs.pop("position_actuator_gains", None)
    motrix_max_iterations = kwargs.pop("motrix_max_iterations", None)
    if backend_type == "mujoco":
        MuJoCoBackend = _load_mujoco_backend()
        if position_actuator_gains is not None:
            kwargs["position_actuator_gains"] = position_actuator_gains
        return cast(SimBackend, MuJoCoBackend(scene, num_envs, sim_dt, **kwargs))
    if backend_type == "motrix":
        MotrixBackend, motrix_available = _load_motrix_backend()
        if not motrix_available:
            raise ImportError("MotrixSim not available, install motrixsim package")
        if motrix_max_iterations is not None:
            kwargs["max_iterations"] = motrix_max_iterations
        return cast(SimBackend, MotrixBackend(scene, num_envs, sim_dt, **kwargs))
    raise ValueError(f"Unknown backend: {backend_type}")


def __getattr__(name: str):
    if name == "MuJoCoBackend":
        return _load_mujoco_backend()
    if name == "MotrixBackend":
        return _load_motrix_backend()[0]
    if name == "MOTRIX_AVAILABLE":
        return _load_motrix_backend()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BackendHeightScanner",
    "BackendPlayCapabilities",
    "BackendPlayRenderPlan",
    "SimBackend",
    "MuJoCoBackend",
    "MotrixBackend",
    "create_backend",
    "normalize_play_render_mode",
]
