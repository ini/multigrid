from minigrid.envs import LockedRoomEnv

#import matplotlib; matplotlib.use('TkAgg')
import time

def random_walk(num_episodes=1):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    env = LockedRoomEnv()
    #env = LockedRoomEnv(render_mode='human', screen_size=500)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        truncated = False
        while not truncated:
            #env.render()
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(random_action)

        #print('Episode:', episode, 'Score:', env.score)



if __name__ == '__main__':
    import cProfile
    cProfile.run('random_walk(500)', sort='cumtime')
