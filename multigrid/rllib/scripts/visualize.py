import argparse
import glob
import os

from ray.rllib.algorithms import Algorithm
from train import algorithm_config, policy_mapping_fn



def load_from_path(algorithm: Algorithm, path: str) -> Algorithm:
    """
    Load the latest checkpoint from the given path.
    """
    checkpoints = glob.glob(f'{path}/checkpoint_*/') or [path]
    lastest_checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]
    algorithm.restore(lastest_checkpoint)
    return algorithm

def visualize(algorithm: Algorithm, num_episodes: int = 100):
    """
    Visualize trajectories from trained agents.
    """
    env = algorithm.env_creator(algorithm.config.env_config)

    for episode in range(num_episodes):
        print('\n', '-' * 32, '\n', 'Episode', episode, '\n', '-' * 32)

        episode_reward = {agent_id: 0.0 for agent_id in env.get_agent_ids()}
        terminated, truncated = {'__all__': False}, {'__all__': False}
        obs, info = env.reset(seed=episode)
        while not terminated['__all__'] and not truncated['__all__']:
            action = {
                agent_id: algorithm.compute_single_action(
                    obs[agent_id], policy_id=policy_mapping_fn(agent_id))
                for agent_id in env.get_agent_ids()
            }
            obs, reward, terminated, truncated, info = env.step(action)
            for agent_id in reward:
                episode_reward[agent_id] += reward[agent_id]

        print('Rewards:', episode_reward)



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
        '--num-agents', type=int, default=2, help="Number of agents in environment.")
    parser.add_argument(
        '--num-episodes', type=int, default=1, help="Number of episodes to visualize.")
    parser.add_argument(
        '--save-dir', type=str, default='~/ray_results/',
        help="Directory for saved policy checkpoint.")

    args = parser.parse_args()
    config = algorithm_config(
        **vars(args),
        env_config={'render_mode': 'human'},
        num_workers=0,
        num_gpus=0,
    )
    algorithm = config.build()
    visualize(algorithm, num_episodes=args.num_episodes)
