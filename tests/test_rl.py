import numpy as np
import pytest
import torch

from controllers import RLController
from integrators import IntegratorType
from plant import Plant
from rl import (
    Actor,
    Critic,
    DDPGAgent,
    DDPGConfig,
    ReplayBuffer,
    SACConfig,
    TD3Config,
    create_agent,
    load_agent_policy,
)
from rl_training import NoisyPlantEpisode, RLTrainingConfig, RLTrainingWorker


@pytest.mark.parametrize('activation', ['Sigmoid', 'ReLU', 'Tanh', 'Softmax'])
def test_ddpg_network_shapes_and_bounded_actor_output(activation):
    config = DDPGConfig(
        action_min=-2.0,
        action_max=3.0,
        hidden_layers=2,
        neurons=8,
        activation=activation,
    )
    actor = Actor(config)
    critic = Critic(config)
    observation = torch.zeros((4, 1))
    action = actor(observation)
    value = critic(observation, action)

    assert action.shape == (4, 1)
    assert torch.all(action >= config.action_min)
    assert torch.all(action <= config.action_max)
    assert value.shape == (4, 1)
    assert isinstance(actor.network[-1], torch.nn.Tanh)
    assert isinstance(critic.network[-1], torch.nn.Linear)


def test_replay_buffer_wraps_and_samples_transitions():
    buffer = ReplayBuffer(2, 1, np.random.default_rng(0))
    for index in range(3):
        observation = np.array([index], dtype=np.float32)
        buffer.add(observation, [index], index, observation + 1, False)

    batch = buffer.sample(2)

    assert len(buffer) == 2
    assert set(batch['observations'].reshape(-1)) == {1.0, 2.0}
    assert batch['actions'].shape == (2, 1)


def test_ddpg_updates_and_policy_checkpoint_round_trip(tmp_path):
    config = DDPGConfig(
        action_min=-1.5,
        action_max=2.5,
        hidden_layers=1,
        neurons=8,
        batch_size=2,
        replay_capacity=8,
    )
    agent = DDPGAgent(config, seed=7)
    buffer = ReplayBuffer(config.replay_capacity, 1, np.random.default_rng(7))
    for index in range(3):
        buffer.add([index], [0.25], -float(index ** 2), [index + 1], False)

    losses = agent.update(buffer)
    action_before = agent.select_action(np.array([0.5]))
    checkpoint = tmp_path / 'policy.pt'
    agent.save_policy(checkpoint)
    loaded_agent = DDPGAgent.load_policy(checkpoint)
    action_after = loaded_agent.select_action(np.array([0.5]))

    assert losses is not None
    assert np.all(np.isfinite(losses))
    assert config.action_min <= action_before[0] <= config.action_max
    assert np.allclose(action_before, action_after)


@pytest.mark.parametrize(
    ('algorithm', 'config'),
    [
        ('DDPG', DDPGConfig),
        ('TD3', TD3Config),
        ('SAC', SACConfig),
    ],
)
def test_off_policy_algorithms_train_bound_actions_and_load_policies(
    algorithm, config, tmp_path
):
    settings = config(
        action_min=-2.0,
        action_max=3.0,
        hidden_layers=1,
        neurons=8,
        batch_size=2,
        replay_capacity=8,
        learning_starts=0,
    )
    agent = create_agent(algorithm, settings, seed=3)
    replay_buffer = ReplayBuffer(8, 1, np.random.default_rng(3))
    for index in range(4):
        replay_buffer.add([index], [0.2], -float(index), [index + 1], False)

    losses = [agent.update(replay_buffer) for _ in range(3)]
    action = agent.select_action(np.array([0.5]))
    checkpoint = tmp_path / f'{algorithm.lower()}_policy.pt'
    agent.save_policy(checkpoint)
    loaded = load_agent_policy(checkpoint)

    assert any(result is not None for result in losses)
    assert np.all(settings.action_min <= action)
    assert np.all(action <= settings.action_max)
    assert loaded.algorithm == algorithm
    assert np.allclose(loaded.select_action(np.array([0.5])), action)


def _make_plant() -> Plant:
    return Plant(
        dims=1,
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        D=np.array([[0.0]]),
        Q=np.array([[0.02]]),
        R=np.array([[0.01]]),
    )


def test_noisy_episode_observes_error_and_rewards_measured_error_only():
    env = NoisyPlantEpisode(
        _make_plant(),
        np.array([-0.5]),
        np.array([0.5]),
        IntegratorType.EULER_MARUYAMA,
        False,
        np.random.default_rng(4),
    )

    observation = env.reset(setpoint=0.5)
    next_observation, reward = env.step(action=0.0, setpoint=0.5)

    assert observation.shape == (1,)
    assert next_observation.shape == (1,)
    assert reward == pytest.approx(-(next_observation[0] ** 2))


def test_rl_controller_saturates_actions_to_policy_bounds():
    class OutOfRangeAgent:
        config = DDPGConfig(action_min=-2.0, action_max=3.0)

        @staticmethod
        def select_action(_observation):
            return np.array([10.0])

    controller = RLController()
    controller.set_agent(OutOfRangeAgent())

    action = controller.calc_u(np.array([[1.0]]))

    assert action.shape == (1, 1)
    assert action[0, 0] == pytest.approx(3.0)


def test_training_worker_runs_a_headless_episode_and_updates_agent():
    config = RLTrainingConfig(
        algorithm='DDPG',
        agent_config=DDPGConfig(
            action_min=-1.0,
            action_max=1.0,
            hidden_layers=1,
            neurons=8,
            batch_size=2,
            replay_capacity=8,
            learning_starts=0,
        ),
        initial_state_min=np.array([-0.1]),
        initial_state_max=np.array([0.1]),
        setpoint_min=0.2,
        setpoint_max=0.8,
        episode_count=1,
        steps_per_episode=4,
        setpoint_changes_per_episode=1,
        seed=4,
    )
    worker = RLTrainingWorker(
        _make_plant(),
        IntegratorType.EULER_MARUYAMA,
        False,
        config,
    )
    completed_episodes = []
    worker.episode_completed.connect(lambda *values: completed_episodes.append(values))

    worker.run()

    assert worker.agent is not None
    assert worker._training_updates > 0
    assert len(completed_episodes) == 1
    assert completed_episodes[0][0] == 1
    assert np.isfinite(completed_episodes[0][1])


@pytest.mark.parametrize(
    ('algorithm', 'agent_config'),
    [
        ('TD3', TD3Config),
        ('SAC', SACConfig),
    ],
)
def test_training_worker_trains_selected_algorithm(algorithm, agent_config):
    config = RLTrainingConfig(
        algorithm=algorithm,
        agent_config=agent_config(
            action_min=-1.0,
            action_max=1.0,
            hidden_layers=1,
            neurons=8,
            batch_size=2,
            replay_capacity=8,
            learning_starts=0,
        ),
        initial_state_min=np.array([-0.1]),
        initial_state_max=np.array([0.1]),
        setpoint_min=0.2,
        setpoint_max=0.8,
        episode_count=1,
        steps_per_episode=4,
        seed=5,
    )
    worker = RLTrainingWorker(
        _make_plant(),
        IntegratorType.EULER_MARUYAMA,
        False,
        config,
    )
    completed_episodes = []
    worker.episode_completed.connect(lambda *values: completed_episodes.append(values))

    worker.run()

    assert worker.agent is not None
    assert worker.agent.algorithm == algorithm
    assert worker._training_updates > 0
    assert completed_episodes[0][0] == 1
