from multigrid.envs import LockedRoomEnv as LockedRoomEnvMultiGrid
from minigrid.envs import LockedRoomEnv as LockedRoomEnvMiniGrid

from multigrid.core.actions import Actions

from multigrid.envs import RedBlueDoorEnv

import numpy as np




def random_walk(num_episodes=1, render=False):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    kwargs = {}
    if render:
        kwargs.update({'render_mode': 'human', 'screen_size': 500})

    env = RedBlueDoorEnv(**kwargs)
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        terminated, truncated = False, False
        while not (terminated or truncated):
            if render: env.render()
            random_action = env.np_random.integers(env.action_space.n)
            obs, reward, terminated, truncated, _ = env.step(random_action)


def compare(num_episodes=1):
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
        obs_multi, _ = env_multi.reset(seed=episode)
        obs_mini, _ = env_mini.reset(seed=episode)
        assert np.all(obs_multi['image'] == obs_mini['image'])
        assert obs_multi['direction'] == obs_mini['direction']

        truncated = False
        while not truncated:
            #env.render()
            random_action_multi = env_multi.np_random.integers(len(Actions))
            random_action_mini = env_mini.np_random.integers(len(Actions))
            assert random_action_multi == random_action_mini
            random_action = random_action_mini
            #print(Actions(random_action))

            obs_multi, reward_multi, terminated, truncated, _ = env_multi.step(random_action)
            obs_mini, reward_mini, terminated, truncated, _ = env_mini.step(random_action)

            locs = np.argwhere(obs_multi['image'] != obs_mini['image'])
            if len(locs) > 0:
                print("CARRY", env_multi.carrying, env_multi.carrying.color, env_mini.carrying, env_mini.carrying.color)
                for loc in locs:
                    print("LOC", loc)
                    print(obs_multi['image'][loc[0], loc[1]], obs_mini['image'][loc[0], loc[1]])
                    print(env_multi.grid.get(*loc[:2]), env_mini.grid.get(*loc[:2]))

            assert np.all(obs_multi['image'] == obs_mini['image'])
            assert obs_multi['direction'] == obs_mini['direction']
            assert reward_multi == reward_mini

        #print('Episode:', episode, 'Score:', env.score)



if __name__ == '__main__':
    #random_walk(1, render=False)
    #compare(1000)
    import cProfile
    cProfile.run('random_walk(1000)', sort='cumtime')
