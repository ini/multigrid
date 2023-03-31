from multigrid.envs import LockedRoomEnv as LockedRoomEnvMultiGrid
from minigrid.envs import LockedRoomEnv as LockedRoomEnvMiniGrid
from minigrid.minigrid_env import MiniGridEnv

from multigrid.core.actions import Actions

from multigrid.envs import RedBlueDoorEnv
from multigrid.envs import ObstructedMaze_Full

import numpy as np






# import inspect
# import sys


# def recompile_nb_code():
#     this_module = sys.modules[__name__]
#     module_members = inspect.getmembers(this_module)

#     for member_name, member in module_members:
#         if hasattr(member, 'recompile') and hasattr(member, 'inspect_llvm'):
#             member.recompile()

# recompile_nb_code()






num_actions = len(Actions)

def item(x):
    if isinstance(x, dict):
        return x.items()
    
def done(terminated, truncated):
    if isinstance(terminated, bool) and isinstance(truncated, bool):
        return terminated or truncated
    
    for agent_id in terminated:
        if not terminated[agent_id] and not truncated[agent_id]:
            return False

    return True

def random_walk(num_episodes=1, render=False, **kwargs):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    if render:
        kwargs.update({'render_mode': 'human', 'screen_size': 500})

    env = LockedRoomEnvMultiGrid(**kwargs)
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        terminated, truncated = False, False
        while not done(terminated, truncated):
            if render:
                env.render()
            if isinstance(env, MiniGridEnv):
                random_action = env.np_random.integers(num_actions)
            else:
                random_action = [
                    #agent_id: 
                    env.np_random.integers(num_actions)
                    for agent_id in env.agents
                ]
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
        obs_mini, _ = env_mini.reset(seed=episode)
        obs_multi, _ = env_multi.reset(seed=episode)
        obs_multi = obs_multi[0]
        assert np.all(obs_multi['image'] == obs_mini['image'])
        assert obs_multi['direction'] == obs_mini['direction']

        random1 = np.random.default_rng(seed=episode)
        random2 = np.random.default_rng(seed=episode)

        truncated = False
        while not truncated:
            # print()
            # print()
            # print('STEP', env_mini.step_count)
            #env.render()
            random_action_multi = random1.integers(len(Actions))
            random_action_mini = random2.integers(len(Actions))
            assert random_action_multi == random_action_mini
            random_action_multi = [random_action_multi]
            # print(Actions(random_action_mini))

            obs_mini, reward_mini, terminated, truncated, _ = env_mini.step(random_action_mini)
            obs_multi, reward_multi, _, _, _ = env_multi.step(random_action_multi)
            obs_multi = obs_multi[0]

            # print(tuple(env_multi.agents[0].state.pos), env_multi.agents[0].state.dir)
            # print(env_mini.agent_pos, env_mini.agent_dir)
            # try:
            #     print('a', env_mini.grid.get(7, 3).is_open)
            #     print('b', env_multi.grid.get(7, 3).is_open)
            # except Exception as e:
            #     print(e)

            locs = np.argwhere(obs_multi['image'] != obs_mini['image'])
            if len(locs) > 0:
                # obj = env_multi.agents[0].state.carrying
                # print("CARRY MULTI", obj, obj.color)
                # print("CARRY MINI", env_mini.carrying, env_mini.carrying.color)
                for loc in locs:
                    print("LOC", loc)
                    print(obs_multi['image'][loc[0], loc[1]], obs_mini['image'][loc[0], loc[1]])
                    # a, b = env_multi.grid.get(*loc[:2]), env_mini.grid.get(*loc[:2])
                    # try:
                    #     print(a.type, a.color, b.type, b.color)
                    # except:
                    #     assert a == b

            assert env_mini.agent_pos == tuple(env_multi.agents[0].state.pos)
            assert np.all(obs_multi['image'] == obs_mini['image'])
            assert obs_multi['direction'] == obs_mini['direction']
            assert reward_multi[0] == reward_mini

        #print('Episode:', episode, 'Score:', env.score)



if __name__ == '__main__':
    #random_walk(agents=3)
    #random_walk(1, agents=3, render=True)
    #compare(1000)
    import cProfile
    cProfile.run('random_walk(5000, agents=1)', sort='cumtime')
