import glob
from train_multi import algorithm_config, policy_mapping_fn

# import torch
# torch.manual_seed(0)


config = algorithm_config(
    env_config={'render_mode': 'human'},
    num_workers=0,
)
algorithm = config.build()



path = '/Users/ini/ray_results/PPO/PPO_RLlib_BlockedUnlockPickupEnv_f0b41_00000_0_2023-06-05_07-56-00'
checkpoints = glob.glob(f'{path}/checkpoint_*/')
lastest_checkpoint = sorted(checkpoints)[-1]
algorithm.restore(lastest_checkpoint)


def visualize(algorithm, num_episodes=100):
    #env = algorithm.env_creator(algorithm.config.env_config)

    from multigrid.envs.minigrid import BlockedUnlockPickupEnv
    from multigrid.rllib.env import rllib_multi_agent_env
    from multigrid.wrappers import ActionMaskWrapper, RewardShaping, RewardShaping2
    env = rllib_multi_agent_env(BlockedUnlockPickupEnv, RewardShaping2)(algorithm.config.env_config)
    env.agent_state.color = ('yellow', 'purple')

    for episode in range(num_episodes):
        print('\n', '-' * 32, '\n', 'Episode', episode, '\n', '-' * 32)

        checkpoints = glob.glob(f'{path}/checkpoint_*/')
        lastest_checkpoint = sorted(checkpoints)[-1]
        algorithm.restore(lastest_checkpoint)

        episode_reward = {agent_id: 0.0 for agent_id in env.get_agent_ids()}
        terminated, truncated = {'__all__': False}, {'__all__': False}
        obs, info = env.reset(seed=episode)
        while not terminated['__all__'] and not truncated['__all__']:
            action = {
                agent_id: algorithm.compute_single_action(
                    obs[agent_id], policy_id=policy_mapping_fn(agent_id, episode, None))
                for agent_id in env.get_agent_ids()
            }
            obs, reward, terminated, truncated, info = env.step(action)
            for agent_id in reward:
                episode_reward[agent_id] += reward[agent_id]

        print('Rewards:', episode_reward)


if __name__ == '__main__':
    visualize(algorithm)