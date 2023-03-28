from multigrid.envs import LockedRoomEnv as LockedRoomEnvMultiGrid
from minigrid.envs import LockedRoomEnv as LockedRoomEnvMiniGrid


import numpy as np

def random_walk(num_episodes=1):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    kwargs = {
        #'see_through_walls': True,
    }
    env_multi = LockedRoomEnvMultiGrid(**kwargs)
    env_mini = LockedRoomEnvMiniGrid(**kwargs)
    #env = LockedRoomEnv(render_mode='human', screen_size=500)

    for episode in range(num_episodes):
        print('Episode', episode)
        obs_multi, _ = env_multi.reset(seed=0)
        obs_mini, _ = env_mini.reset(seed=0)
        locs = np.argwhere(obs_multi['image'] != obs_mini['image'])
        print(locs)
        for loc in locs:
            print("LOC", loc)
            cell_multi = env_multi.grid.get(*loc[:2])
            cell_mini = env_mini.grid.get(*loc[:2])
        assert np.all(obs_multi['image'] == obs_mini['image'])
        assert obs_multi['direction'] == obs_mini['direction']

        truncated = False
        while not truncated:
            #env.render()
            random_action = env_multi.action_space.sample()
            obs_multi, reward_multi, terminated, truncated, _ = env_multi.step(random_action)
            obs_mini, reward_mini, terminated, truncated, _ = env_mini.step(random_action)
            # assert np.all(obs_multi['image'] == obs_mini['image'])
            # assert obs_multi['direction'] == obs_mini['direction']
            # assert reward_multi == reward_mini
            print(obs_multi['direction'], obs_mini['direction'])

        #print('Episode:', episode, 'Score:', env.score)



if __name__ == '__main__':
    random_walk(100)
    # import cProfile
    # cProfile.run('random_walk(2000)', sort='cumtime')
