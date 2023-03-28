from multigrid.envs import LockedRoomEnvMultiGrid
from minigrid.envs import LockedRoomEnvMiniGrid

#import matplotlib; matplotlib.use('TkAgg')
import time

def random_walk(num_episodes=1):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    env_multi = LockedRoomEnvMultiGrid()
    env_mini = LockedRoomEnvMiniGrid()
    #env = LockedRoomEnv(render_mode='human', screen_size=500)

    for episode in range(num_episodes):
        obs_multi, _ = env_multi.reset(seed=0)
        obs_mini, _ = env_multi.reset(seed=0)

        print(obs_multi, obs_mini)
        truncated = False
        while not truncated:
            #env.render()
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(random_action)

        #print('Episode:', episode, 'Score:', env.score)



if __name__ == '__main__':
    random_walk(1)
    # import cProfile
    # cProfile.run('random_walk(2000)', sort='cumtime')
