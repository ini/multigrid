import argparse
import json
import numpy as np

from ray.rllib.utils.typing import AgentID
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from typing import Callable

from train import get_algorithm_config, find_checkpoint_dir, get_policy_mapping_fn
from ray.rllib.core.rl_module import RLModule



def visualize(
    modules: dict[str, RLModule],
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
        print()
        print('-' * 32, '\n', 'Episode', episode, '\n', '-' * 32)

        episode_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}
        terminations, truncations = {'__all__': False}, {'__all__': False}
        observations, infos = env.reset()

        # Get initial states for each agent
        states = {
            agent_id: modules[policy_mapping_fn(agent_id)].get_initial_state()
            for agent_id in env.agents
        }

        while not terminations['__all__'] and not truncations['__all__']:
            # Store current frame
            frames.append(env.env.unwrapped.get_frame())

            # Compute actions for each agent
            actions = {}
            observations = convert_to_torch_tensor(observations)
            for agent_id in env.agents:
                agent_module = modules[policy_mapping_fn(agent_id)]
                out = agent_module.forward_inference({'obs': observations[agent_id]})
                logits = out['action_dist_inputs']
                action_dist_class = agent_module.get_inference_action_dist_cls()
                action_dist = action_dist_class.from_logits(logits)
                actions[agent_id] = action_dist.sample().item()

            # Take actions in environment and accumulate rewards
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id in rewards:
                episode_rewards[agent_id] += rewards[agent_id]

        frames.append(env.env.unwrapped.get_frame())
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
    config = get_algorithm_config(
        **vars(args),
        num_workers=0,
        num_gpus=0,
    )
    algorithm = config.build()
    checkpoint = find_checkpoint_dir(args.load_dir)
    policy_mapping_fn = lambda agent_id, *args, **kwargs: f'policy_{agent_id}'

    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        path = checkpoint / 'learner_group' / 'learner' / 'rl_module/'
        modules = RLModule.from_checkpoint(path)
        policy_mapping_fn = get_policy_mapping_fn(checkpoint, args.num_agents)

    frames = visualize(modules, policy_mapping_fn, num_episodes=args.num_episodes)
    if args.gif:
        from array2gif import write_gif
        filename = args.gif if args.gif.endswith('.gif') else f'{args.gif}.gif'
        print(f"Saving GIF to {filename}")
        write_gif(np.array(frames), filename, fps=10)
