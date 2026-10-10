"""Headless noisy-plant episodes and background off-policy training."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from integrators import IntegratorType
from plant import Plant
from rl import (
    DDPGConfig,
    OffPolicyAgent,
    OffPolicyConfig,
    ReplayBuffer,
    SACConfig,
    TD3Config,
    create_agent,
)


@dataclass
class RLTrainingConfig:
    """Episode and selected-agent settings captured when training starts."""

    algorithm: str
    agent_config: OffPolicyConfig
    initial_state_min: np.ndarray
    initial_state_max: np.ndarray
    setpoint_min: float
    setpoint_max: float
    episode_count: int = 500
    steps_per_episode: int = 600
    setpoint_changes_per_episode: int = 1
    seed: int = 0

    def validate(self):
        agent_configs = {
            'DDPG': DDPGConfig,
            'TD3': TD3Config,
            'SAC': SACConfig,
        }
        try:
            expected_config = agent_configs[self.algorithm]
        except KeyError as error:
            raise ValueError(f'Unsupported RL algorithm: {self.algorithm}.') from error
        if not isinstance(self.agent_config, expected_config):
            raise TypeError(
                f'{self.algorithm} requires {expected_config.__name__}.'
            )
        self.agent_config.validate()
        if self.episode_count < 1 or self.steps_per_episode < 1:
            raise ValueError('Episode count and episode length must be positive.')
        if self.setpoint_changes_per_episode < 0:
            raise ValueError('Setpoint changes per episode cannot be negative.')
        if self.setpoint_min > self.setpoint_max:
            raise ValueError('Setpoint minimum cannot exceed its maximum.')
        lower = np.asarray(self.initial_state_min)
        upper = np.asarray(self.initial_state_max)
        if lower.shape != upper.shape or lower.ndim != 1 or lower.size < 1:
            raise ValueError('Initial-state bounds must be matching one-dimensional arrays.')
        if np.any(lower > upper):
            raise ValueError('Every initial-state minimum must be less than or equal to its maximum.')


class NoisyPlantEpisode:
    """Advance an independent plant using only noisy measured-error observations."""

    def __init__(
        self,
        plant: Plant,
        initial_state_min: np.ndarray,
        initial_state_max: np.ndarray,
        integrator_method: IntegratorType,
        use_ode_mode: bool,
        rng: np.random.Generator,
    ):
        self.plant = Plant(
            dims=plant.dims,
            A=plant.A.copy(),
            B=plant.B.copy(),
            C=plant.C.copy(),
            D=plant.D.copy(),
            Q=plant.Q.copy(),
            R=plant.R.copy(),
            x_0=plant.x_0.copy(),
            u_0=plant.u_0.copy(),
        )
        self.plant.dt_anim = plant.dt_anim
        self.plant.dt_int = plant.dt_int
        self.plant.set_cached_arrays()
        self.initial_state_min = np.asarray(initial_state_min, dtype=float).copy()
        self.initial_state_max = np.asarray(initial_state_max, dtype=float).copy()
        self.integrator_method = integrator_method
        self.use_ode_mode = use_ode_mode
        self.rng = rng
        self.setpoint = 0.0

    def reset(self, setpoint: float) -> np.ndarray:
        self.plant.x = self.rng.uniform(
            self.initial_state_min, self.initial_state_max
        ).reshape(self.plant.dims, 1)
        self.plant.u = np.zeros((1, 1))
        self.setpoint = float(setpoint)
        self.plant.y_meas = (
            self.plant.calc_y(self.plant.x, self.plant.u)
            + self._sample_measurement_noise()
        )
        return np.array([self.setpoint - float(self.plant.y_meas.item())], dtype=np.float32)

    def step(self, action: float, setpoint: float) -> tuple[np.ndarray, float]:
        self.setpoint = float(setpoint)
        self.plant.u = np.array([[float(action)]])
        self._integrate_frame()
        y = self.plant.calc_y(self.plant.x, self.plant.u)
        self.plant.y_meas = y + self._sample_measurement_noise()
        error = self.setpoint - float(self.plant.y_meas.item())
        return np.array([error], dtype=np.float32), -(error ** 2)

    def _sample_measurement_noise(self) -> np.ndarray:
        return self.rng.normal(
            loc=0.0,
            scale=np.sqrt(self.plant.R[0, 0]),
            size=(1, 1),
        )

    def _integrate_frame(self):
        plant = self.plant
        time_steps = np.diff(plant.t_span_0)
        state = plant.x.copy()
        control_effect = plant.B @ plant.u

        if self.use_ode_mode:
            process_noise = self.rng.multivariate_normal(
                np.zeros(plant.dims), plant.Q
            ).reshape(plant.dims, 1)
            if self.integrator_method is IntegratorType.ANALYTIC_ODE:
                plant.x = (
                    plant.exp_A_t_span[-1] @ state
                    + plant.A_inv_exp_At_minus_I[-1]
                    @ (control_effect + process_noise)
                )
                return
            if self.integrator_method is not IntegratorType.RK4:
                raise ValueError(f'Unsupported ODE integrator: {self.integrator_method}.')
            for dt in time_steps:
                k1 = plant.A @ state + control_effect + process_noise
                k2 = plant.A @ (state + 0.5 * k1 * dt) + control_effect + process_noise
                k3 = plant.A @ (state + 0.5 * k2 * dt) + control_effect + process_noise
                k4 = plant.A @ (state + k3 * dt) + control_effect + process_noise
                state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            plant.x = state
            return

        if self.integrator_method is IntegratorType.EULER_MARUYAMA:
            for dt in time_steps:
                process_noise = self.rng.multivariate_normal(
                    np.zeros(plant.dims), plant.Q
                ).reshape(plant.dims, 1)
                state += (plant.A @ state + control_effect) * dt + process_noise * np.sqrt(dt)
            plant.x = state
            return

        if self.integrator_method is IntegratorType.ANALYTIC_SDE:
            for index, _dt in enumerate(time_steps):
                if index == len(time_steps) - 1:
                    transition = plant.Phi_last
                    input_gain = plant.Gamma_last
                    covariance = plant.Q_d_last
                else:
                    transition = plant.Phi
                    input_gain = plant.Gamma
                    covariance = plant.Q_d
                process_noise = self.rng.multivariate_normal(
                    np.zeros(plant.dims), covariance
                ).reshape(plant.dims, 1)
                state = transition @ state + input_gain @ plant.u + process_noise
            plant.x = state
            return

        raise ValueError(f'Unsupported SDE integrator: {self.integrator_method}.')


class RLTrainingWorker(QThread):
    """Train the selected off-policy agent off the GUI thread."""

    episode_completed = pyqtSignal(int, float, float, float)
    paused_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)
    training_failed = pyqtSignal(str)
    policy_ready = pyqtSignal(object, bool)

    def __init__(
        self,
        plant: Plant,
        integrator_method: IntegratorType,
        use_ode_mode: bool,
        config: RLTrainingConfig,
        parent=None,
    ):
        super().__init__(parent)
        config.validate()
        if config.initial_state_min.size != plant.dims:
            raise ValueError('Initial-state bounds must have one pair per plant state.')
        self.plant = plant
        self.integrator_method = integrator_method
        self.use_ode_mode = use_ode_mode
        self.config = config
        self.agent: OffPolicyAgent | None = None
        self._pause_requested = Event()
        self._resume_event = Event()
        self._resume_event.set()
        self._stop_requested = Event()
        self._paused = False
        self._control_lock = Lock()
        self._setpoint_changes = config.setpoint_changes_per_episode

    def request_pause(self):
        self._pause_requested.set()
        self._resume_event.clear()

    @property
    def is_paused(self) -> bool:
        return self._paused

    def resume(self, setpoint_changes_per_episode: int):
        with self._control_lock:
            self._setpoint_changes = int(setpoint_changes_per_episode)
        self._pause_requested.clear()
        self._resume_event.set()

    def request_stop(self):
        self._stop_requested.set()
        self._pause_requested.clear()
        self._resume_event.set()

    def update_setpoint_change_count(self, count: int):
        with self._control_lock:
            self._setpoint_changes = int(count)

    def run(self):
        self._loss_history: list[tuple[float, float]] = []
        self._training_updates = 0
        try:
            self.agent = create_agent(
                self.config.algorithm,
                self.config.agent_config,
                seed=self.config.seed,
            )
            replay_buffer = ReplayBuffer(
                self.config.agent_config.replay_capacity,
                self.config.agent_config.observation_dim,
                self.agent.rng,
            )
            env = NoisyPlantEpisode(
                self.plant,
                self.config.initial_state_min,
                self.config.initial_state_max,
                self.integrator_method,
                self.use_ode_mode,
                np.random.default_rng(self.config.seed),
            )
            self._train_episodes(env, replay_buffer)
        except Exception as error:
            self.training_failed.emit(f'{type(error).__name__}: {error}')
        finally:
            if self.agent is not None:
                self.policy_ready.emit(self.agent, self._training_updates > 0)

    def _train_episodes(self, env: NoisyPlantEpisode, replay_buffer: ReplayBuffer):
        for episode in range(self.config.episode_count):
            if self._wait_if_paused() or self._stop_requested.is_set():
                break
            with self._control_lock:
                changes_per_episode = self._setpoint_changes
            observation = env.reset(self._sample_setpoint(env.rng))
            change_steps = self._make_setpoint_schedule(changes_per_episode, start=1)
            changes_done = 0
            episode_reward = 0.0
            actor_losses: list[float] = []
            critic_losses: list[float] = []

            for step in range(self.config.steps_per_episode):
                if self._wait_if_paused() or self._stop_requested.is_set():
                    break
                with self._control_lock:
                    requested_changes = self._setpoint_changes
                if requested_changes != changes_per_episode:
                    changes_per_episode = requested_changes
                    change_steps = self._make_setpoint_schedule(
                        max(0, changes_per_episode - changes_done),
                        start=step,
                    )
                if step in change_steps:
                    env.setpoint = self._sample_setpoint(env.rng)
                    changes_done += 1

                warmup_steps = max(
                    self.config.agent_config.learning_starts,
                    self.config.agent_config.batch_size,
                )
                if len(replay_buffer) < warmup_steps:
                    action = env.rng.uniform(
                        self.config.agent_config.action_min,
                        self.config.agent_config.action_max,
                        size=1,
                    )
                else:
                    action = self.agent.explore_action(observation, env.rng)
                action = np.clip(
                    action,
                    self.config.agent_config.action_min,
                    self.config.agent_config.action_max,
                )
                next_observation, reward = env.step(float(action[0]), env.setpoint)
                replay_buffer.add(
                    observation,
                    action,
                    reward,
                    next_observation,
                    step == self.config.steps_per_episode - 1,
                )
                observation = next_observation
                episode_reward += reward

                losses = self.agent.update(replay_buffer)
                if losses is not None:
                    if np.isfinite(losses[0]):
                        actor_losses.append(losses[0])
                    if np.isfinite(losses[1]):
                        critic_losses.append(losses[1])
                    self._training_updates += 1

            if actor_losses or critic_losses:
                mean_actor_loss = (
                    float(np.mean(actor_losses)) if actor_losses else float('nan')
                )
                mean_critic_loss = (
                    float(np.mean(critic_losses)) if critic_losses else float('nan')
                )
                self._loss_history.append((mean_actor_loss, mean_critic_loss))
            else:
                mean_actor_loss = float('nan')
                mean_critic_loss = float('nan')
            self.episode_completed.emit(
                episode + 1,
                episode_reward,
                mean_actor_loss,
                mean_critic_loss,
            )
            if self._stop_requested.is_set():
                break

    def _wait_if_paused(self) -> bool:
        if not self._pause_requested.is_set():
            return False
        if not self._paused:
            self._paused = True
            self.paused_changed.emit(True)
            self.status_changed.emit('Training paused')
        while self._pause_requested.is_set() and not self._stop_requested.is_set():
            self._resume_event.wait(0.05)
        if self._paused:
            self._paused = False
            self.paused_changed.emit(False)
            self.status_changed.emit('Training resumed')
        return self._stop_requested.is_set()

    def _sample_setpoint(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.config.setpoint_min, self.config.setpoint_max))

    def _make_setpoint_schedule(self, count: int, start: int = 0) -> set[int]:
        available_steps = max(0, self.config.steps_per_episode - start)
        count = min(max(0, int(count)), available_steps)
        if count == 0:
            return set()
        return {
            int(step)
            for step in np.linspace(
                start,
                self.config.steps_per_episode - 1,
                count,
                dtype=int,
            )
        }
