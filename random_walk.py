import numpy as np
import multigrid

from multigrid.core.actions import Actions
from multigrid.minigrid_interface import MiniGridInterface
from multigrid.multigrid_env import MultiGridEnv
from tqdm import tqdm

from multigrid.envs import BlockedUnlockPickupEnv
from multigrid.envs import RedBlueDoorEnv

# from multigrid.envs import LockedRoomEnv
# from multigrid.envs import RedBlueDoorEnv
# from multigrid.envs import ObstructedMaze_Full







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



def done(terminated, truncated):
    if not hasattr(terminated, '__iter__'):
        return terminated or truncated

    for agent_id in terminated:
        if not terminated[agent_id] and not truncated[agent_id]:
            return False

    return True


from minigrid.minigrid_env import MiniGridEnv
from multigrid.envs import LockedRoomEnv
def random_walk(num_episodes=1, render=False, **kwargs):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    if render:
        kwargs.update({'render_mode': 'human', 'screen_size': 500})

    env = LockedRoomEnv(**kwargs)
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        terminated, truncated = False, False
        while not done(terminated, truncated):
            if render:
                env.render()
            if isinstance(env, (MiniGridEnv, MiniGridInterface)):
                random_action = env.np_random.integers(num_actions)
            else:
                random_action = {
                    agent_id: 
                    env.np_random.integers(num_actions)
                    for agent_id in env.agents
                }
            obs, reward, terminated, truncated, _ = env.step(random_action)


import minigrid.envs






def compare_clean(env_cls, num_episodes=1, **env_kwargs):
    """
    Test that the logic for `multigrid.MultiGridEnv` and `minigrid.MiniGridEnv`
    is the same in the single-agent case.
    """
    EnvMini = getattr(minigrid.envs, env_cls.__name__)
    EnvMulti = env_cls

    env_mini = EnvMini(**env_kwargs)
    env_multi = EnvMulti(**env_kwargs)

    loop = tqdm(
        range(num_episodes),
        desc=f'Testing {type(env_multi).__name__}',
        leave=True,
    )
    for episode in loop:
        obs_mini, _ = env_mini.reset(seed=episode)
        obs_multi, _ = env_multi.reset(seed=episode)
        assert np.array_equal(obs_mini['image'], obs_multi['image'])
        assert obs_mini['direction'] == obs_multi['direction']
        assert obs_mini['mission'] == obs_multi['mission']

        rng = np.random.default_rng(seed=episode)
        term_mini, trunc_mini = False, False
        while not (term_mini or trunc_mini):
            action = rng.integers(len(Actions))
            obs_mini, reward_mini, term_mini, trunc_mini, _ = env_mini.step(action)
            obs_multi, reward_multi, term_multi, trunc_multi, _ = env_multi.step(action)

            assert np.array_equal(obs_mini['image'], obs_multi['image'])
            assert obs_mini['direction'] == obs_multi['direction']
            assert reward_mini == reward_multi
            assert term_mini == term_multi
            assert trunc_mini == trunc_multi


def test_clean(num_episodes_per_env=10000):
    import inspect
    import multigrid.envs

    for attr_name in dir(multigrid.envs):
        attr = getattr(multigrid.envs, attr_name)
        if inspect.isclass(attr):
            if issubclass(attr, MultiGridEnv):
                compare_clean(attr, num_episodes=num_episodes_per_env)


def compare(num_episodes=1):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    kwargs = {
        #'see_through_walls': True,
    }

    EnvMulti = LockedRoomEnv
    EnvMini = getattr(minigrid.envs, EnvMulti.__name__)

    env_multi = EnvMulti(**kwargs)
    env_mini = EnvMini(**kwargs)
    #env = LockedRoomEnv(render_mode='human', screen_size=500)

    for episode in range(0, num_episodes):
        print('Episode', episode)
        obs_mini, _ = env_mini.reset(seed=episode)
        obs_multi, _ = env_multi.reset(seed=episode)
        print(env_mini.mission, env_multi.mission)

        locs = np.argwhere(obs_multi['image'] != obs_mini['image'])
        if len(locs) > 0:
            for loc in locs:
                print("LOC", loc)
                print(obs_multi['image'][loc[0], loc[1]], obs_mini['image'][loc[0], loc[1]])

        #print(env_mini.agent_dir, env_multi.agent_dir)
        assert np.all(obs_multi['image'] == obs_mini['image'])
        assert obs_multi['direction'] == obs_mini['direction']

        random1 = np.random.default_rng(seed=episode)
        random2 = np.random.default_rng(seed=episode)

        terminated, truncated = False, False
        while not (terminated or truncated):
            # print()
            # print()
            # print('STEP', env_mini.step_count)
            #env.render()
            random_action_multi = random1.integers(len(Actions))
            random_action_mini = random2.integers(len(Actions))
            assert random_action_multi == random_action_mini
            random_action_multi = random_action_multi
            print(Actions(random_action_mini))

            obs_mini, reward_mini, terminated, truncated, _ = env_mini.step(random_action_mini)
            obs_multi, reward_multi, mult_term, _, _ = env_multi.step(random_action_multi)
            #obs_multi = obs_multi[0]

            # print(tuple(env_multi.agents[0].state.pos), env_multi.agents[0].state.dir)
            print('pos', env_mini.agent_pos, env_multi.agent_pos)
            print('dir', env_mini.agent_dir, env_multi.agent_dir)
            print('front', env_mini.grid.get(*env_mini.front_pos), env_multi.grid.get(*env_multi.front_pos))
            print('term', terminated, mult_term)
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
            ##print(env_mini.carrying, env_multi.carrying)
            ##print(reward_mini, reward_multi)
            assert reward_multi == reward_mini

        #print('Episode:', episode, 'Score:', env.score)



if __name__ == '__main__':
    #random_walk(agents=3)
    #random_walk(1, render=True)

    # from multigrid.envs import BlockedUnlockPickupEnv
    # compare_clean(BlockedUnlockPickupEnv, 1000)

    test_clean(1000)

    # import cProfile
    # cProfile.run('random_walk(1000, agents=1)', sort='cumtime')
