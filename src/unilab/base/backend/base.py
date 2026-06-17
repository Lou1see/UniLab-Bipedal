import abc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np

from unilab.dr.types import (
    DomainRandomizationCapabilities,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)

PreStepControlFn = Callable[[Any, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BackendPlayCapabilities:
    """Backend-native play/render capabilities surfaced through env contracts."""

    supports_native_interactive_renderer: bool = False
    supports_physics_state_playback: bool = False
    supports_native_video_capture: bool = False


class BackendHeightScanner(abc.ABC):
    """Backend-owned height-field scanner created on the env init path."""

    @abc.abstractmethod
    def scan(self) -> np.ndarray:
        """Return sampled values with shape ``(num_envs, num_points)``."""


PLAY_RENDER_MODES = frozenset({"auto", "interactive", "record", "none"})


@dataclass(frozen=True)
class BackendPlayRenderPlan:
    """Backend-resolved playback rendering behavior."""

    mode: str
    headless: bool
    record_video: bool
    num_steps: int | None
    output_video: str | PathLike[str] | None


def normalize_play_render_mode(play_render_mode: str | None) -> str:
    mode = "auto" if play_render_mode is None else str(play_render_mode).strip().lower()
    if mode not in PLAY_RENDER_MODES:
        joined = ", ".join(sorted(PLAY_RENDER_MODES))
        raise ValueError(f"training.play_render_mode must be one of: {joined}; got {mode!r}.")
    return mode


class SimBackend(abc.ABC):
    """Unified simulation backend contract."""

    _pre_step_control_fn: PreStepControlFn | None
    _scene_cleanup_handle: Any | None
    backend_type: str

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """Number of vectorized environments."""

    @property
    @abc.abstractmethod
    def model(self):
        """Backend-native physics model."""

    @property
    @abc.abstractmethod
    def num_actuators(self) -> int:
        """Number of actuators."""

    @property
    @abc.abstractmethod
    def num_dof_vel(self) -> int:
        """Number of joint velocity DoFs, excluding floating base."""

    @abc.abstractmethod
    def get_actuator_ctrl_range(self) -> np.ndarray:
        """Return actuator control ranges with shape ``(num_actuators, 2)``."""

    @abc.abstractmethod
    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        """Return full qpos for a named keyframe."""

    def get_default_qpos(self) -> np.ndarray:
        """Return the backend/model default qpos through a stable contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose default qpos")

    @abc.abstractmethod
    def get_init_qvel(self) -> np.ndarray:
        """Return a zero qvel vector with the backend state dimension."""

    @abc.abstractmethod
    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve body/link names to backend ids."""

    def get_body_id(self, name: str) -> int:
        """Resolve one body/link name through the backend contract."""
        return int(self.get_body_ids([name])[0])

    def get_geom_id(self, name: str) -> int:
        """Resolve one geom name through the backend contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom ids")

    def get_geom_size(self, name: str) -> np.ndarray:
        """Return one geom size vector through the backend contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom sizes")

    def create_hfield_scanner(
        self,
        *,
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str = "yaw",
        output: str = "height",
    ) -> BackendHeightScanner:
        """Create a reusable height-field scanner on the init/cold path."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native height-field scanners"
        )

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        """Return body ids in the subtree rooted at ``root_body_id``."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body subtree ids")

    def get_geom_names(self) -> tuple[str, ...]:
        """Return backend geom names in backend id order."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom names")

    def get_geom_body_ids(self) -> np.ndarray:
        """Return the owning body id for each geom."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom body ids")

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-geom contact type and affinity masks."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom contact masks")

    def get_geom_friction(self) -> np.ndarray:
        """Return the backend geom-friction table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom friction")

    def get_gravity(self) -> np.ndarray:
        """Return the backend gravity vector."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose gravity")

    def get_body_mass(self) -> np.ndarray:
        """Return the backend body-mass table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body mass")

    def get_body_ipos(self) -> np.ndarray:
        """Return the backend body inertial-position table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body ipos")

    def get_dof_armature(self) -> np.ndarray:
        """Return the backend DoF armature table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose dof armature")

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve MuJoCo-style body IDs used by motion datasets."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose motion body ids")

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve site names to backend ids."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose site ids")

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to velocity DoF indices."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose joint dof indices")

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to qpos indices."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose joint dof position indices"
        )

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to qvel indices."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose joint dof velocity indices"
        )

    def cleanup_scene_assets(self) -> None:
        """Release cold-path scene artifacts owned by the backend."""
        cleanup_handle = getattr(self, "_scene_cleanup_handle", None)
        if cleanup_handle is None:
            return
        cleanup_handle.cleanup()
        self._scene_cleanup_handle = None

    def __del__(self) -> None:
        try:
            self.cleanup_scene_assets()
        except Exception:
            pass

    @abc.abstractmethod
    def get_joint_range(self) -> np.ndarray | None:
        """Return joint position limits, excluding floating base."""

    @abc.abstractmethod
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        """Advance physics."""

    def set_pre_step_control(self, fn: PreStepControlFn | None) -> None:
        """Register an env-owned policy-control to physics-control converter."""
        self._pre_step_control_fn = fn

    def _apply_pre_step_control(self, ctrl: np.ndarray) -> np.ndarray:
        if self._pre_step_control_fn is None:
            return ctrl
        converted = np.asarray(self._pre_step_control_fn(self, ctrl), dtype=ctrl.dtype)
        if converted.shape != ctrl.shape:
            raise ValueError(
                f"pre-step control must return shape {ctrl.shape}, got {converted.shape}"
            )
        return converted

    @abc.abstractmethod
    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        """Set the physics state for selected environments."""

    @abc.abstractmethod
    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Return supported domain-randomization capabilities for this backend."""

    def apply_init_randomization(self, plan: InitRandomizationPlan) -> None:
        """Apply cold-path model/materialization randomization."""
        if plan.is_empty():
            return
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support init-lifecycle randomization"
        )

    def materialize(self) -> None:
        """Finalize cold-path backend resources before reset/step."""

    @abc.abstractmethod
    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        """Apply a scheduled interval randomization plan."""

    def apply_body_linear_velocity_delta(
        self,
        body_ids: np.ndarray,
        velocity_delta: np.ndarray,
    ) -> None:
        """Apply a world-frame linear-velocity delta to specific bodies."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support interval body velocity perturbation"
        )

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
    ) -> None:
        """Apply a world-frame force to specific bodies for the upcoming step."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support interval body force perturbation"
        )

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        """Return backend-native play/render capabilities."""
        return BackendPlayCapabilities()

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        """Resolve high-level playback mode into backend-owned render parameters."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not define playback render mode semantics"
        )

    def run_playback(
        self,
        *,
        env: Any,
        initialize: Callable[[], Any],
        step: Callable[[Any], Any],
        num_steps: int | None,
        output_video: str | PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Callable[[], np.ndarray] | None = None,
        camera_kwargs: dict[str, Any] | None = None,
        extra_data_getter: Callable[[], np.ndarray | None] | None = None,
    ) -> str | None:
        """Execute backend-owned playback for an env wrapper."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support playback execution")

    def init_renderer(
        self,
        spacing: float = 1.0,
        *,
        offset_mode: str = "grid",
        headless: bool = False,
        capture: bool = False,
        width: int = 1280,
        height: int = 720,
        camera_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a backend-native renderer."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support native rendering")

    def render(self) -> None:
        """Render one frame through a backend-native interactive renderer."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native interactive rendering"
        )

    def capture_video_frame(self) -> np.ndarray:
        """Capture one RGB frame through a backend-native renderer."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native video capture"
        )

    def get_physics_state(self) -> np.ndarray:
        """Return a physics snapshot suitable for offline playback/video export."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support physics-state playback"
        )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return the playback model for a specific env when variants exist."""
        return self.model

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-joint (kp, kd) arrays from the backend model."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support reading actuator gains"
        )

    @abc.abstractmethod
    def get_base_pos(self) -> np.ndarray:
        """Return base position in world frame, shape ``(num_envs, 3)``."""

    @abc.abstractmethod
    def get_base_quat(self) -> np.ndarray:
        """Return base quaternion in world frame as wxyz, shape ``(num_envs, 4)``."""

    @abc.abstractmethod
    def get_base_lin_vel(self) -> np.ndarray:
        """Return base linear velocity in world frame."""

    @abc.abstractmethod
    def get_base_ang_vel(self) -> np.ndarray:
        """Return base angular velocity in world frame."""

    @abc.abstractmethod
    def get_dof_pos(self) -> np.ndarray:
        """Return joint positions, excluding floating base."""

    @abc.abstractmethod
    def get_dof_vel(self) -> np.ndarray:
        """Return joint velocities, excluding floating base."""

    @abc.abstractmethod
    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return body positions in world frame."""

    @abc.abstractmethod
    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return body quaternions in world frame as wxyz."""

    def get_body_pose_w_rows(
        self,
        body_ids: np.ndarray,
        env_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return body world poses for paired env/body rows."""
        pos = self.get_body_pos_w(body_ids)
        quat = self.get_body_quat_w(body_ids)
        return pos[env_ids, np.arange(len(body_ids))], quat[env_ids, np.arange(len(body_ids))]

    def get_body_pose_w(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return body world positions and quaternions."""
        return self.get_body_pos_w(body_ids), self.get_body_quat_w(body_ids)

    @abc.abstractmethod
    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return body linear velocities in world frame."""

    @abc.abstractmethod
    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return body angular velocities in world frame."""

    def get_body_state_w(
        self,
        body_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return world-frame body position, quaternion, linear velocity, angular velocity."""
        return (
            self.get_body_pos_w(body_ids),
            self.get_body_quat_w(body_ids),
            self.get_body_lin_vel_w(body_ids),
            self.get_body_ang_vel_w(body_ids),
        )

    def get_body_vel_w(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return world-frame body linear and angular velocity."""
        return self.get_body_lin_vel_w(body_ids), self.get_body_ang_vel_w(body_ids)

    @abc.abstractmethod
    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return body positions in baselink frame."""

    @abc.abstractmethod
    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return body quaternions in baselink frame as wxyz."""

    @abc.abstractmethod
    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return body linear velocities in baselink frame."""

    @abc.abstractmethod
    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return body angular velocities in baselink frame."""

    @abc.abstractmethod
    def get_sensor_data(self, name: str) -> np.ndarray:
        """Return sensor data by name."""

    def get_sensor_data_rows(self, name: str, env_ids: np.ndarray) -> np.ndarray:
        """Return sensor rows by env id."""
        return self.get_sensor_data(name)[env_ids]

    def get_sensor_data_batch(self, names: Sequence[str]) -> np.ndarray:
        """Return multiple sensors stacked on the last axis."""
        return np.concatenate([self.get_sensor_data(name) for name in names], axis=-1)

    def get_site_jacobian_w(
        self,
        site_ids: np.ndarray,
        *,
        env_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return world-frame translational and rotational site Jacobians."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose site Jacobians")
