"""First-principles reinforcement-learning components for continuous control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn


class OffPolicyAgent(Protocol):
    """Shared interface for replay-based continuous-control agents."""

    algorithm: str
    config: OffPolicyConfig
    rng: np.random.Generator

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """Return a bounded action for one observation."""

    def explore_action(
        self, observation: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Return an exploratory action for training."""

    def update(self, replay_buffer: ReplayBuffer) -> tuple[float, float] | None:
        """Run one learning update, returning actor and critic losses."""


@dataclass
class OffPolicyConfig:
    """Shared network, replay, and optimizer settings for off-policy agents."""

    observation_dim: int = 1
    action_min: float = -4.0
    action_max: float = 4.0
    hidden_layers: int = 2
    neurons: int = 128
    activation: str = 'ReLU'
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    discount_factor: float = 0.99
    target_update_rate: float = 0.005
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 1_000
    exploration_noise_std: float = 0.1

    def validate(self):
        if self.observation_dim < 1:
            raise ValueError('Observation dimension must be positive.')
        if not self.action_min < self.action_max:
            raise ValueError('RL minimum action must be less than maximum action.')
        if self.hidden_layers < 1 or self.neurons < 1:
            raise ValueError('The actor and critic must have at least one hidden layer and neuron.')
        if self.activation not in {'Sigmoid', 'ReLU', 'Tanh', 'Softmax'}:
            raise ValueError(f'Unsupported hidden-layer activation: {self.activation}.')
        if self.actor_learning_rate <= 0 or self.critic_learning_rate <= 0:
            raise ValueError('Actor and critic learning rates must be positive.')
        if not 0 <= self.discount_factor <= 1:
            raise ValueError('Discount factor must be between 0 and 1.')
        if not 0 < self.target_update_rate <= 1:
            raise ValueError('Target update rate must be in (0, 1].')
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError('Replay capacity must be at least the positive batch size.')
        if self.learning_starts < 0:
            raise ValueError('Random warm-up transition count cannot be negative.')
        if self.exploration_noise_std < 0:
            raise ValueError('Exploration noise standard deviation cannot be negative.')


@dataclass
class DDPGConfig(OffPolicyConfig):
    """Configuration for DDPG."""


@dataclass
class TD3Config(OffPolicyConfig):
    """Configuration for Twin Delayed DDPG."""

    target_policy_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2

    def validate(self):
        super().validate()
        if self.target_policy_noise_std < 0 or self.target_noise_clip < 0:
            raise ValueError('TD3 target policy noise and noise clip cannot be negative.')
        if self.policy_delay < 1:
            raise ValueError('TD3 policy delay must be positive.')


@dataclass
class SACConfig(OffPolicyConfig):
    """Configuration for Soft Actor-Critic."""

    initial_entropy_coefficient: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: float = -1.0
    entropy_learning_rate: float = 3e-4

    def validate(self):
        super().validate()
        if self.initial_entropy_coefficient <= 0:
            raise ValueError('SAC entropy coefficient must be positive.')
        if self.entropy_learning_rate <= 0:
            raise ValueError('SAC entropy learning rate must be positive.')


def _activation(name: str) -> nn.Module:
    activations = {
        'Sigmoid': nn.Sigmoid,
        'ReLU': nn.ReLU,
        'Tanh': nn.Tanh,
        'Softmax': lambda: nn.Softmax(dim=-1),
    }
    try:
        return activations[name]()
    except KeyError as error:
        raise ValueError(f'Unsupported hidden-layer activation: {name}.') from error


def _make_hidden_layers(input_dim: int, hidden_layers: int, neurons: int, activation: str):
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(current_dim, neurons), _activation(activation)))
        current_dim = neurons
    return layers, current_dim


class Actor(nn.Module):
    """Fully connected deterministic policy with bounded scalar action output."""

    def __init__(self, config: OffPolicyConfig):
        super().__init__()
        layers, last_dim = _make_hidden_layers(
            config.observation_dim,
            config.hidden_layers,
            config.neurons,
            config.activation,
        )
        layers.extend((nn.Linear(last_dim, 1), nn.Tanh()))
        self.network = nn.Sequential(*layers)
        self.action_center = (config.action_max + config.action_min) / 2
        self.action_scale = (config.action_max - config.action_min) / 2

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action_center + self.action_scale * self.network(observation)


class Critic(nn.Module):
    """Fully connected scalar Q-value estimator for observation-action pairs."""

    def __init__(self, config: OffPolicyConfig):
        super().__init__()
        layers, last_dim = _make_hidden_layers(
            config.observation_dim + 1,
            config.hidden_layers,
            config.neurons,
            config.activation,
        )
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1))


class GaussianActor(nn.Module):
    """Squashed Gaussian policy for SAC's bounded stochastic action space."""

    def __init__(self, config: SACConfig):
        super().__init__()
        layers, last_dim = _make_hidden_layers(
            config.observation_dim,
            config.hidden_layers,
            config.neurons,
            config.activation,
        )
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(last_dim, 1)
        self.log_std = nn.Linear(last_dim, 1)
        self.action_center = (config.action_max + config.action_min) / 2
        self.action_scale = (config.action_max - config.action_min) / 2

    def distribution(self, observation: torch.Tensor):
        hidden = self.trunk(observation)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(-20, 2)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        latent = distribution.mean if deterministic else distribution.rsample()
        squashed = torch.tanh(latent)
        action = self.action_center + self.action_scale * squashed
        log_probability = distribution.log_prob(latent)
        log_probability = log_probability - torch.log(1 - squashed.square() + 1e-6)
        log_probability = log_probability - np.log(self.action_scale)
        return action, log_probability.sum(dim=-1, keepdim=True)


class ReplayBuffer:
    """Fixed-capacity cyclic replay buffer with uniform random sampling."""

    def __init__(self, capacity: int, observation_dim: int, rng: np.random.Generator):
        if capacity < 1 or observation_dim < 1:
            raise ValueError('Replay capacity and observation dimension must be positive.')
        self.capacity = capacity
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = rng

    def add(self, observation, action, reward, next_observation, done):
        self.observations[self.position] = np.asarray(observation, dtype=np.float32).reshape(-1)
        self.actions[self.position] = np.asarray(action, dtype=np.float32).reshape(1)
        self.rewards[self.position] = float(reward)
        self.next_observations[self.position] = np.asarray(
            next_observation, dtype=np.float32
        ).reshape(-1)
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size < 1 or batch_size > self.size:
            raise ValueError(f'Cannot sample batch of {batch_size} from {self.size} transitions.')
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


class DDPGAgent:
    """Deep Deterministic Policy Gradient agent with target networks and replay."""

    algorithm = 'DDPG'

    def __init__(self, config: DDPGConfig, seed: int = 0):
        config.validate()
        self.config = config
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.target_actor = Actor(config)
        self.target_critic = Critic(config)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).reshape(1, -1)
        )
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(observation_tensor).cpu().numpy()[0]
        self.actor.train()
        return np.clip(action, self.config.action_min, self.config.action_max)

    def explore_action(
        self, observation: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        action = self.select_action(observation)
        action += rng.normal(0.0, self.config.exploration_noise_std, size=action.shape)
        return np.clip(action, self.config.action_min, self.config.action_max)

    def update(self, replay_buffer: ReplayBuffer) -> tuple[float, float] | None:
        if len(replay_buffer) < self.config.batch_size:
            return None
        batch = replay_buffer.sample(self.config.batch_size)
        tensors = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in batch.items()
        }

        with torch.no_grad():
            next_actions = self.target_actor(tensors['next_observations'])
            target_values = self.target_critic(
                tensors['next_observations'], next_actions
            )
            targets = tensors['rewards'] + self.config.discount_factor * (
                1 - tensors['dones']
            ) * target_values

        predicted_values = self.critic(tensors['observations'], tensors['actions'])
        critic_loss = nn.functional.mse_loss(predicted_values, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(
            tensors['observations'],
            self.actor(tensors['observations']),
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        return float(actor_loss.detach()), float(critic_loss.detach())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        rate = self.config.target_update_rate
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.lerp_(source_parameter, rate)

    def save_policy(self, path: str | Path):
        """Save policy weights and architecture/action metadata for inference."""
        torch.save(
            {
                'algorithm': self.algorithm,
                'config': asdict(self.config),
                'actor_state_dict': self.actor.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load_policy(cls, path: str | Path) -> OffPolicyAgent:
        """Load a saved policy when its algorithm matches this agent class."""
        agent = load_agent_policy(path)
        if agent.algorithm != cls.algorithm:
            raise ValueError(
                f'The saved policy uses {agent.algorithm}, not {cls.algorithm}.'
            )
        return agent


class TD3Agent(DDPGAgent):
    """Twin Delayed DDPG with clipped target noise and twin critics."""

    algorithm = 'TD3'

    def __init__(self, config: TD3Config, seed: int = 0):
        config.validate()
        self.config = config
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.actor = Actor(config)
        self.critic_one = Critic(config)
        self.critic_two = Critic(config)
        self.target_actor = Actor(config)
        self.target_critic_one = Critic(config)
        self.target_critic_two = Critic(config)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_one.load_state_dict(self.critic_one.state_dict())
        self.target_critic_two.load_state_dict(self.critic_two.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_one_optimizer = torch.optim.Adam(
            self.critic_one.parameters(), lr=config.critic_learning_rate
        )
        self.critic_two_optimizer = torch.optim.Adam(
            self.critic_two.parameters(), lr=config.critic_learning_rate
        )
        self.update_count = 0

    def update(self, replay_buffer: ReplayBuffer) -> tuple[float, float] | None:
        if len(replay_buffer) < self.config.batch_size:
            return None
        batch = replay_buffer.sample(self.config.batch_size)
        tensors = _batch_tensors(batch)
        with torch.no_grad():
            target_actions = self.target_actor(tensors['next_observations'])
            noise = torch.randn_like(target_actions) * self.config.target_policy_noise_std
            noise = noise.clamp(
                -self.config.target_noise_clip,
                self.config.target_noise_clip,
            )
            target_actions = (target_actions + noise).clamp(
                self.config.action_min,
                self.config.action_max,
            )
            target_values = torch.minimum(
                self.target_critic_one(tensors['next_observations'], target_actions),
                self.target_critic_two(tensors['next_observations'], target_actions),
            )
            targets = tensors['rewards'] + self.config.discount_factor * (
                1 - tensors['dones']
            ) * target_values

        observations = tensors['observations']
        actions = tensors['actions']
        prediction_one = self.critic_one(observations, actions)
        prediction_two = self.critic_two(observations, actions)
        critic_loss = (
            nn.functional.mse_loss(prediction_one, targets)
            + nn.functional.mse_loss(prediction_two, targets)
        )
        self.critic_one_optimizer.zero_grad()
        self.critic_two_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_one_optimizer.step()
        self.critic_two_optimizer.step()

        self.update_count += 1
        actor_loss_value = float('nan')
        if self.update_count % self.config.policy_delay == 0:
            actor_loss = -self.critic_one(
                observations, self.actor(observations)
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach())
            self._soft_update(self.target_actor, self.actor)
            self._soft_update(self.target_critic_one, self.critic_one)
            self._soft_update(self.target_critic_two, self.critic_two)
        return actor_loss_value, float(critic_loss.detach())


class SACAgent:
    """Soft Actor-Critic with a stochastic squashed-Gaussian policy."""

    algorithm = 'SAC'

    def __init__(self, config: SACConfig, seed: int = 0):
        config.validate()
        self.config = config
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.actor = GaussianActor(config)
        self.critic_one = Critic(config)
        self.critic_two = Critic(config)
        self.target_critic_one = Critic(config)
        self.target_critic_two = Critic(config)
        self.target_critic_one.load_state_dict(self.critic_one.state_dict())
        self.target_critic_two.load_state_dict(self.critic_two.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_one_optimizer = torch.optim.Adam(
            self.critic_one.parameters(), lr=config.critic_learning_rate
        )
        self.critic_two_optimizer = torch.optim.Adam(
            self.critic_two.parameters(), lr=config.critic_learning_rate
        )
        self.log_entropy_coefficient = torch.tensor(
            np.log(config.initial_entropy_coefficient),
            dtype=torch.float32,
            requires_grad=config.automatic_entropy_tuning,
        )
        self.entropy_optimizer = (
            torch.optim.Adam(
                [self.log_entropy_coefficient],
                lr=config.entropy_learning_rate,
            )
            if config.automatic_entropy_tuning
            else None
        )

    @property
    def entropy_coefficient(self) -> torch.Tensor:
        return self.log_entropy_coefficient.exp()

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).reshape(1, -1)
        )
        with torch.no_grad():
            action, _ = self.actor.sample(observation_tensor, deterministic=True)
        return np.clip(
            action.cpu().numpy()[0],
            self.config.action_min,
            self.config.action_max,
        )

    def explore_action(
        self, observation: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        del rng
        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).reshape(1, -1)
        )
        with torch.no_grad():
            action, _ = self.actor.sample(observation_tensor)
        return np.clip(
            action.cpu().numpy()[0],
            self.config.action_min,
            self.config.action_max,
        )

    def update(self, replay_buffer: ReplayBuffer) -> tuple[float, float] | None:
        if len(replay_buffer) < self.config.batch_size:
            return None
        tensors = _batch_tensors(replay_buffer.sample(self.config.batch_size))
        observations = tensors['observations']
        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(
                tensors['next_observations']
            )
            next_values = torch.minimum(
                self.target_critic_one(tensors['next_observations'], next_actions),
                self.target_critic_two(tensors['next_observations'], next_actions),
            ) - self.entropy_coefficient.detach() * next_log_probability
            targets = tensors['rewards'] + self.config.discount_factor * (
                1 - tensors['dones']
            ) * next_values

        prediction_one = self.critic_one(observations, tensors['actions'])
        prediction_two = self.critic_two(observations, tensors['actions'])
        critic_loss = (
            nn.functional.mse_loss(prediction_one, targets)
            + nn.functional.mse_loss(prediction_two, targets)
        )
        self.critic_one_optimizer.zero_grad()
        self.critic_two_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_one_optimizer.step()
        self.critic_two_optimizer.step()

        sampled_actions, log_probability = self.actor.sample(observations)
        actor_values = torch.minimum(
            self.critic_one(observations, sampled_actions),
            self.critic_two(observations, sampled_actions),
        )
        actor_loss = (
            self.entropy_coefficient.detach() * log_probability - actor_values
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.entropy_optimizer is not None:
            entropy_loss = -(
                self.log_entropy_coefficient
                * (log_probability + self.config.target_entropy).detach()
            ).mean()
            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_optimizer.step()

        self._soft_update(self.target_critic_one, self.critic_one)
        self._soft_update(self.target_critic_two, self.critic_two)
        return float(actor_loss.detach()), float(critic_loss.detach())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.lerp_(source_parameter, self.config.target_update_rate)

    def save_policy(self, path: str | Path):
        torch.save(
            {
                'algorithm': self.algorithm,
                'config': asdict(self.config),
                'actor_state_dict': self.actor.state_dict(),
            },
            Path(path),
        )


def _batch_tensors(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in batch.items()}


def create_agent(
    algorithm: str,
    config: OffPolicyConfig,
    seed: int = 0,
) -> OffPolicyAgent:
    """Construct the selected supported off-policy agent."""
    agents = {
        'DDPG': (DDPGConfig, DDPGAgent),
        'TD3': (TD3Config, TD3Agent),
        'SAC': (SACConfig, SACAgent),
    }
    try:
        config_type, agent_type = agents[algorithm]
    except KeyError as error:
        raise ValueError(f'Unsupported RL algorithm: {algorithm}.') from error
    if not isinstance(config, config_type):
        raise TypeError(f'{algorithm} requires {config_type.__name__}.')
    return agent_type(config, seed=seed)


def load_agent_policy(path: str | Path) -> OffPolicyAgent:
    """Load any supported policy checkpoint, including older DDPG files."""
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not {
        'config', 'actor_state_dict'
    }.issubset(checkpoint):
        raise ValueError('File does not contain a valid RL policy checkpoint.')
    algorithm = checkpoint.get('algorithm', 'DDPG')
    config_types = {
        'DDPG': DDPGConfig,
        'TD3': TD3Config,
        'SAC': SACConfig,
    }
    try:
        config_type = config_types[algorithm]
    except KeyError as error:
        raise ValueError(f'Unsupported saved RL algorithm: {algorithm}.') from error
    config = config_type(**checkpoint['config'])
    agent = create_agent(algorithm, config)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor.eval()
    return agent
