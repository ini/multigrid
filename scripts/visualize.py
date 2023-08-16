import argparse
import json
import numpy as np

from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.typing import AgentID
from typing import Any, Callable, Iterable

from train import algorithm_config
from utils import find_checkpoint_dir, get_policy_mapping_fn



def get_actions(
    agent_ids: Iterable[AgentID],
    algorithm: Algorithm,
    policy_mapping_fn: Callable[[AgentID], str],
    observations: dict[AgentID, Any],
    states: dict[AgentID, Any]) -> tuple[dict[AgentID, Any], dict[AgentID, Any]]:
    """
    Get actions for the given agents.

    Parameters
    ----------
    agent_ids : Iterable[AgentID]
        Agent IDs for which to get actions
    algorithm : Algorithm
        RLlib algorithm instance with trained policies
    policy_mapping_fn : Callable(AgentID) -> str
        Function mapping agent IDs to policy IDs
    observations : dict[AgentID, Any]
        Observations for each agent
    states : dict[AgentID, Any]
        States for each agent

    Returns
    -------
    actions : dict[AgentID, Any]
        Actions for each agent
    states : dict[AgentID, Any]
        Updated states for each agent
    """
    actions = {}
    for agent_id in agent_ids:
        if states[agent_id]:
            actions[agent_id], states[agent_id], _ = algorithm.compute_single_action(
                observations[agent_id],
                states[agent_id],
                policy_id=policy_mapping_fn(agent_id)
            )
        else:
            actions[agent_id] = algorithm.compute_single_action(
                observations[agent_id],
                policy_id=policy_mapping_fn(agent_id)
            )

    return actions, states

def visualize(
    algorithm: Algorithm,
    policy_mapping_fn: Callable[[AgentID], str],
    num_episodes: int = 10) -> list[np.ndarray]:
    """
    Visualize trajectories from trained agents.

    Parameters
    ----------
    algorithm : Algorithm
        RLlib algorithm instance with trained policies
    policy_mapping_fn : Callable(AgentID) -> str
        Function mapping agent IDs to policy IDs
    num_episodes : int, default=10
        Number of episodes to visualize
    """
    frames = []
    env = algorithm.env_creator(algorithm.config.env_config)

    for episode in range(num_episodes):
        print('\n', '-' * 32, '\n', 'Episode', episode, '\n', '-' * 32)

        episode_rewards = {agent_id: 0.0 for agent_id in env.get_agent_ids()}
        terminations, truncations = {'__all__': False}, {'__all__': False}
        observations, infos = env.reset()
        states = {
            agent_id: algorithm.get_policy(policy_mapping_fn(agent_id)).get_initial_state()
            for agent_id in env.get_agent_ids()
        }
        while not terminations['__all__'] and not truncations['__all__']:
            frames.append(env.get_frame())
            actions, states = get_actions(
                env.get_agent_ids(), algorithm, policy_mapping_fn, observations, states)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id in rewards:
                episode_rewards[agent_id] += rewards[agent_id]

        frames.append(env.get_frame())
        print('Rewards:', episode_rewards)

    env.close()
    return frames



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algo', type=str, default='PPO',
        help="The name of the RLlib-registered algorithm to use.")
    parser.add_argument(
        '--framework', type=str, choices=['torch', 'tf', 'tf2'], default='torch',
        help="Deep learning framework to use.")
    parser.add_argument(
        '--lstm', action='store_true', help="Use LSTM model.")
    parser.add_argument(
        '--env', type=str, default='MultiGrid-Empty-8x8-v0',
        help="MultiGrid environment to use.")
    parser.add_argument(
        '--env-config', type=json.loads, default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')")
    parser.add_argument(
        '--num-agents', type=int, default=2, help="Number of agents in environment.")
    parser.add_argument(
        '--num-episodes', type=int, default=10, help="Number of episodes to visualize.")
    parser.add_argument(
        '--load-dir', type=str,
        help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument(
        '--gif', type=str, help="Store output as GIF at given path.")

    args = parser.parse_args()
    args.env_config.update(render_mode='human')
    config = algorithm_config(
        **vars(args),
        num_workers=0,
        num_gpus=0,
    )
    algorithm = config.build()
    checkpoint = find_checkpoint_dir(args.load_dir)
    policy_mapping_fn = lambda agent_id, *args, **kwargs: f'policy_{agent_id}'
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        algorithm.restore(checkpoint)
        policy_mapping_fn = get_policy_mapping_fn(checkpoint, args.num_agents)

    frames = visualize(algorithm, policy_mapping_fn, num_episodes=args.num_episodes)
    if args.gif:
        from array2gif import write_gif
        filename = args.gif if args.gif.endswith('.gif') else f'{args.gif}.gif'
        print(f"Saving GIF to {filename}")
        write_gif(np.array(frames), filename, fps=10)
