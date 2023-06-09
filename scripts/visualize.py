import argparse
import json
import numpy as np

from ray.rllib.algorithms import Algorithm
from train import algorithm_config, get_checkpoint_dir, policy_mapping_fn



def visualize(algorithm: Algorithm, num_episodes: int = 100) -> list[np.ndarray]:
    """
    Visualize trajectories from trained agents.
    """
    frames = []
    env = algorithm.env_creator(algorithm.config.env_config)

    for episode in range(num_episodes):
        print('\n', '-' * 32, '\n', 'Episode', episode, '\n', '-' * 32)

        episode_reward = {agent_id: 0.0 for agent_id in env.get_agent_ids()}
        terminated, truncated = {'__all__': False}, {'__all__': False}
        obs, info = env.reset()
        while not terminated['__all__'] and not truncated['__all__']:
            frames.append(env.get_frame())
            action = {
                agent_id: algorithm.compute_single_action(
                    obs[agent_id], policy_id=policy_mapping_fn(agent_id))
                for agent_id in env.get_agent_ids()
            }
            obs, reward, terminated, truncated, info = env.step(action)
            for agent_id in reward:
                episode_reward[agent_id] += reward[agent_id]

        frames.append(env.get_frame())
        print('Rewards:', episode_reward)

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
    checkpoint = get_checkpoint_dir(args.load_dir)
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        algorithm.restore(checkpoint)

    frames = visualize(algorithm, num_episodes=args.num_episodes)
    if args.gif:
        from array2gif import write_gif
        filename = args.gif if args.gif.endswith('.gif') else f'{args.gif}.gif'
        print(f"Saving GIF to {filename}")
        write_gif(np.array(frames), filename, fps=10)
