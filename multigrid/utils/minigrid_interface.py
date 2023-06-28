import numpy as np

from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from typing import Any, Sequence, SupportsFloat

from ..core.world_object import WorldObj
from ..base import MultiGridEnv



class MiniGridInterface(MultiGridEnv):
    """
    MultiGridEnv interface for compatibility with single-agent MiniGrid environments.

    Most environment implementations deriving from `minigrid.MiniGridEnv` can be
    converted to a single-agent `MultiGridEnv` by simply inheriting from
    `MiniGridInterface` instead (along with using the multigrid grid and grid objects).

    Examples
    --------
    Start with a single-agent minigrid environment:

    >>> from minigrid.core.world_object import Ball, Key, Door
    >>> from minigrid.core.grid import Grid
    >>> from minigrid import MiniGridEnv

    >>> class MyEnv(MiniGridEnv):
    >>>    ... # existing class definition

    Now use multigrid imports, keeping the environment class definition the same:

    >>> from multigrid.core.world_object import Ball, Key, Door
    >>> from multigrid.core.grid import Grid
    >>> from multigrid.utils.minigrid_interface import MiniGridInterface as MiniGridEnv

    >>> class MyEnv(MiniGridEnv):
    >>>    ... # same class definition
    """

    def reset(self, *args, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment.
        """
        result = super().reset(*args, **kwargs)
        return tuple(item[0] for item in result)

    def step(self, action: ActType) -> tuple[
        ObsType,
        SupportsFloat,
        bool,
        bool,
        dict[str, Any]]:
        """
        Run one timestep of the environment’s dynamics
        using the provided agent action.
        """
        result = super().step({0: action})
        return tuple(item[0] for item in result)

    @property
    def action_space(self) -> spaces.Space:
        """
        Get action space.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].action_space` instead."
        )
        return self.agents[0].action_space

    @action_space.setter
    def action_space(self, space: spaces.Space):
        """
        Set action space.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].action_space` instead."
        )
        self.agents[0].action_space = space

    @property
    def observation_space(self) -> spaces.Space:
        """
        Get observation space.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].observation_space` instead."
        )
        return self.agents[0].observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space):
        """
        Set observation space.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].observation_space` instead."
        )
        self.agents[0].observation_space = space

    @property
    def agent_pos(self) -> np.ndarray[int]:
        """
        Get agent position.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].pos` instead."
        )
        return self.agents[0].pos

    @agent_pos.setter
    def agent_pos(self, value: Sequence[int]):
        """
        Set agent position.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].pos` instead."
        )
        if value is not None:
            self.agents[0].pos = value

    @property
    def agent_dir(self) -> int:
        """
        Get agent direction.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].dir` instead."
        )
        return self.agents[0].dir

    @agent_dir.setter
    def agent_dir(self, value: Sequence[int]):
        """
        Set agent direction.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].dir` instead."
        )
        self.agents[0].dir = value

    @property
    def carrying(self) -> WorldObj:
        """
        Get object carried by agent.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].carrying` instead."
        )
        return self.agents[0].carrying

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].dir.to_vec()` instead."
        )
        return self.agents[0].dir.to_vec()

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Use `env.agents[i].front_pos` instead."
        )
        return self.agents[0].front_pos

    def place_agent(self, *args, **kwargs) -> tuple[int, int]:
        """
        Set agent starting point at an empty position in the grid.
        """
        return super().place_agent(self.agents[0], *args, **kwargs)
