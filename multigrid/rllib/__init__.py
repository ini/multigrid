"""
This package provides tools for using MultiGrid environments with
the RLlib MultiAgentEnv API.

*****
Usage
*****

Use a specific environment configuration from :mod:`multigrid.envs` by name:

    >>> import multigrid.rllib # registers environment configurations with RLlib
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> algorithm_config = PPOConfig().environment(env='MultiGrid-Empty-8x8-v0')

Wrap an environment instance with :class:`.RLlibWrapper`:

    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-8x8-v0', agents=2, render_mode='human')

    >>> from multigrid.rllib import RLlibWrapper
    >>> env = RLlibWrapper(env)

Wrap an environment class with :func:`.to_rllib_env()`:

    >>> from multigrid.envs import EmptyEnv
    >>> from multigrid.rllib import to_rllib_env
    >>> MyEnv = to_rllib_env(EmptyEnv, default_config={'size': 8})
    >>> config = {'agents': 2, 'render_mode': 'human'}
    >>> env = MyEnv(config)
"""

import gymnasium as gym

from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env

from ..base import MultiGridEnv
from ..envs import CONFIGURATIONS
from ..wrappers import OneHotObsWrapper



class RLlibWrapper(gym.Wrapper, MultiAgentEnv):
    """
    Wrapper for a ``MultiGridEnv`` environment that implements the
    RLlib ``MultiAgentEnv`` interface.
    """

    def __init__(self, env: MultiGridEnv):
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        gym.Wrapper.__init__(self, env)
        MultiAgentEnv.__init__(self)

    def get_agent_ids(self):
        return {agent.index for agent in self.agents}

    def step(self, *args, **kwargs):
        obs, rewards, terminations, truncations, infos = super().step(*args, **kwargs)
        terminations['__all__'] = all(terminations.values())
        truncations['__all__'] = all(truncations.values())
        return obs, rewards, terminations, truncations, infos



def to_rllib_env(
    env_cls: type[MultiGridEnv],
    *wrappers: gym.Wrapper,
    default_config: dict = {}) -> type[MultiAgentEnv]:
    """
    Convert a ``MultiGridEnv`` environment class to an RLLib ``MultiAgentEnv`` class.

    Note that this is a wrapper around the environment **class**,
    not environment instances.

    Parameters
    ----------
    env_cls : type[MultiGridEnv]
        ``MultiGridEnv`` environment class
    wrappers : gym.Wrapper
        Gym wrappers to apply to the environment
    default_config : dict
        Default configuration for the environment

    Returns
    -------
    rllib_env_cls : type[MultiAgentEnv]
        RLlib ``MultiAgentEnv`` environment class
    """
    class RLlibEnv(RLlibWrapper):
        def __init__(self, config: dict = {}):
            config = {**default_config, **config}
            env = env_cls(**config)
            for wrapper in wrappers:
                env = wrapper(env)
            super().__init__(env)

    RLlibEnv.__name__ = f"RLlib_{env_cls.__name__}"
    return RLlibEnv



# Register environments with RLlib
for name, (env_cls, config) in CONFIGURATIONS.items():
    register_env(name, to_rllib_env(env_cls, OneHotObsWrapper, default_config=config))
