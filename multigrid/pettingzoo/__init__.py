"""
This package provides tools for using MultiGrid environments with
the PettingZoo ParallelEnv API.

*****
Usage
*****

Wrap an environment instance with :class:`.PettingZooWrapper`:

    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-8x8-v0', agents=2, render_mode='human')

    >>> from multigrid.pettingzoo import PettingZooWrapper
    >>> env = PettingZooWrapper(env)

Wrap an environment class with :func:`.to_pettingzoo_env()`:

    >>> from multigrid.envs import EmptyEnv
    >>> from multigrid.pettingzoo import to_pettingzoo_env
    >>> PZEnv = to_pettingzoo_env(EmptyEnv, metadata={'name': 'empty_v0'})
    >>> env = PZEnv(agents=2, render_mode='human')
"""

import gymnasium as gym

from gymnasium import spaces
from pettingzoo import ParallelEnv
from typing import Any

from ..base import AgentID, MultiGridEnv



class PettingZooWrapper(ParallelEnv):
    """
    Wrapper for a ``MultiGridEnv`` environment that implements the
    PettingZoo ``ParallelEnv`` interface.
    """

    def __init__(self, env: MultiGridEnv):
        self.env = env
        self.reset = self.env.reset
        self.step = self.env.step
        self.render = self.env.render
        self.close = self.env.close

    @property
    def agents(self) -> list[AgentID]:
        if self.env.is_done():
            return []
        return [agent.index for agent in self.env.agents if not agent.terminated]

    @property
    def possible_agents(self) -> list[AgentID]:
        return [agent.index for agent in self.env.agents]

    @property
    def observation_spaces(self) -> dict[AgentID, spaces.Space]:
        return dict(self.env.observation_space)

    @property
    def action_spaces(self) -> dict[AgentID, spaces.Space]:
        return dict(self.env.action_space)

    def observation_space(self, agent_id: AgentID) -> spaces.Space:
        return self.env.observation_space[agent_id]

    def action_space(self, agent_id: AgentID) -> spaces.Space:
        return self.env.action_space[agent_id]



def to_pettingzoo_env(
    env_cls: type[MultiGridEnv],
    *wrappers: gym.Wrapper,
    metadata: dict[str, Any] = {}) -> type[ParallelEnv]:
    """
    Convert a ``MultiGridEnv`` environment class to a PettingZoo ``ParallelEnv`` class.

    Note that this is a wrapper around the environment **class**,
    not environment instances.

    Parameters
    ----------
    env_cls : type[MultiGridEnv]
        ``MultiGridEnv`` environment class
    wrappers : gym.Wrapper
        Gym wrappers to apply to the environment
    metadata : dict[str, Any]
        Environment metadata

    Returns
    -------
    pettingzoo_env_cls : type[ParallelEnv]
        PettingZoo ``ParallelEnv`` environment class
    """
    class PettingZooEnv(PettingZooWrapper):
        def __init__(self, *args, **kwargs):
            env = env_cls(*args, **kwargs)
            for wrapper in wrappers:
                env = wrapper(env)
            super().__init__(env)

    PettingZooEnv.__name__ = f"PettingZoo_{env_cls.__name__}"
    PettingZooEnv.metadata = metadata
    return PettingZooEnv