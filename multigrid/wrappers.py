from __future__ import annotations

import gymnasium as gym
import numba as nb
import numpy as np

from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from numpy.typing import NDArray as ndarray

from .multigrid_env import MultiGridEnv, AgentID, ObsType
from .core.constants import Color, Direction, State, Type



class OneHotObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Examples
    --------
    >>> from multigrid.envs import EmptyEnv
    >>> from multigrid.wrappers import OneHotObsWrapper
    >>> env = EmptyEnv()
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0]])
    >>> env = OneHotObsWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
            dtype=uint8)
    """

    def __init__(self, env: MultiGridEnv):
        """
        :meta private:
        """
        super().__init__(env)
        self.dim_sizes = np.array([
            len(Type), len(Color), max(len(State), len(Direction))])

        # Update agent observation spaces
        dim = sum(self.dim_sizes)
        for agent in self.env.agents:
            view_height, view_width, _ = agent.observation_space['image'].shape
            agent.observation_space['image'] = spaces.Box(
                low=0, high=1, shape=(view_height, view_width, dim), dtype=np.uint8)

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        for agent_id in obs:
            obs[agent_id]['image'] = self.one_hot(obs[agent_id]['image'], self.dim_sizes)

        return obs

    @staticmethod
    @nb.njit(cache=True)
    def one_hot(x: ndarray[np.int], dim_sizes: ndarray[np.int]) -> ndarray[np.uint8]:
        """
        Return a one-hot encoding of a 3D integer array,
        where each 2D slice is encoded separately.

        Parameters
        ----------
        x : ndarray[int] of shape (view_height, view_width, dim)
            3D array of integers to be one-hot encoded
        dim_sizes : ndarray[int] of shape (dim,)
            Number of possible values for each dimension

        Returns
        -------
        out : ndarray[uint8] of shape (view_height, view_width, sum(dim_sizes))
            One-hot encoding

        :meta private:
        """
        out = np.zeros((x.shape[0], x.shape[1], sum(dim_sizes)), dtype=np.uint8)

        dim_offset = 0
        for d in range(len(dim_sizes)):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k = dim_offset + x[i, j, d]
                    out[i, j, k] = 1

            dim_offset += dim_sizes[d]

        return out


class SingleAgentWrapper(gym.Wrapper):
    """
    Wrapper to convert a multi-agent environment into a
    single-agent environment.
    """

    def __init__(self, env: MultiGridEnv):
        super().__init__(env)
        self.observation_space = env.agents[0].observation_space
        self.action_space = env.agents[0].action_space

    def reset(self, *args, **kwargs):
        """
        :meta private:
        """
        result = super().reset(*args, **kwargs)
        return tuple(item[0] for item in result)

    def step(self, action):
        """
        :meta private:
        """
        result = super().step({0: action})
        return tuple(item[0] for item in result)


# class Wrapper(gym.Wrapper):
#     """
#     :meta private:
#     """

#     @classmethod
#     def config(cls, **kwargs) -> Callable[[gym.Env], Wrapper]:
#         """
#         Return wrapper class with the given configuration,
#         whose constructor accepts a single argument: the base environment.

#         `MyWrapper.config(**kwargs)(env)` is equivalent to `MyWrapper(env, **kwargs)`. 

#         Parameters
#         ----------
#         kwargs : dict
#             Configuration parameters
#         """
#         class ConfigWrapper(cls):
#             def __init__(self, env):
#                 super().__init__(env, **kwargs)

#         ConfigWrapper.__name__ = cls.__name__
#         return ConfigWrapper


# class MovementReward(Wrapper):

#     def __init__(self, env: MultiGridEnv, scale: float = 1e-5):
#         super().__init__(env)
#         self._scale = scale

#     def step(self, actions):
#         prev_pos = {agent_id: self.agents[agent_id].pos for agent_id in actions}
#         obs, reward, terminated, truncated, info = super().step(actions)
#         for agent_id, action in actions.items():
#             if action == Action.forward:
#                 reward[agent_id] = self.reward_scale
#         return obs, reward, terminated, truncated, info







# class ActionMaskWrapper(Wrapper):

#     # def __init__(self, env: MultiGridEnv):
#     #     super().__init__(env)
#     #     action_mask_space = spaces.Box(low=0, high=1, shape=(len(Action),), dtype=bool)
#     #     for agent in self.env.agents:
#     #         agent.observation_space['action_mask'] = action_mask_space

#     # def observation(self, obs):
#     #     for agent_id in obs:
#     #         obs[agent_id]['action_mask'] = self.action_mask(self.env.agents[agent_id])
#     #     return obs

#     def step(self, actions):
#         obs, reward, terminated, truncated, info = super().step(actions)
#         for agent_id, action in actions.items():
#             action_mask = self.action_mask(self.env.agents[agent_id])
#             if not action_mask[action] or action in {Action.left, Action.right}:
#                 reward[agent_id] = -1e-5

#         return obs, reward, terminated, truncated, info

#     def action_mask(self, agent: Agent):
#         fwd_pos = agent.front_pos
#         fwd_obj = self.grid.get(*fwd_pos)
#         mask = np.zeros(len(Action), dtype=bool)
#         mask[Action.left] = True
#         mask[Action.right] = True
#         mask[Action.forward] = can_overlap(fwd_obj)
#         mask[Action.pickup] = can_pickup(fwd_obj)
#         mask[Action.drop] = (fwd_obj is None and agent.state.carrying is not None)

#         if fwd_obj and fwd_obj.type == Type.door:
#             if agent.state.carrying and agent.state.carrying.type == Type.key:
#                 if fwd_obj.is_locked:
#                     if fwd_obj.color == agent.state.carrying.color:
#                         mask[Action.toggle] = True
#                 else:
#                     mask[Action.toggle] = True
#         elif fwd_obj and fwd_obj.type == Type.box:
#             mask[Action.toggle] = True

#             mask[Action.toggle] = (
#                 fwd_obj is not None and fwd_obj.type in {Type.box, Type.door})
#         mask[Action.toggle] = (
#             fwd_obj is not None and fwd_obj.type in {Type.box, Type.door})
#         return mask




# class RewardShaping(Wrapper):
#     """
#     General reward shaping wrapper for MultiGrid environments.
#     """

#     def __init__(self, env: MultiGridEnv, bonus=1e-4):
#         """
#         Parameters
#         ----------
#         env : MultiGridEnv
#             Environment to wrap
#         bonus : float
#             Reward bonus
#         """
#         super().__init__(env)
#         self._bonus = bonus
#         #self.agents[0].color = 'purple'
#         #self.agents[1].color = 'yellow'

#     def reset(self, *args, **kwargs):
#         self._picked_up = [] # objects that have been picked up
#         self._opened = [] # doors that have been opened
#         return super().reset(*args, **kwargs)

#     def step(self, actions):
#         prev_pos = {agent_id: self.agents[agent_id].pos for agent_id in actions}
#         obs, reward, terminated, truncated, info = super().step(actions)

#         for agent_id, action in actions.items():
#             agent = self.agents[agent_id]

#             # Reward picking up new objects
#             if agent.carrying not in self._picked_up:
#                 self._picked_up.append(agent.carrying)
#                 reward[agent_id] += self._bonus

#             # Reward opening new doors
#             if agent_id in actions and action == Action.toggle:
#                 fwd_obj = self.env.grid.get(*agent.front_pos)
#                 if isinstance(fwd_obj, Door) and fwd_obj not in self._opened:
#                     self._opened.append(fwd_obj)
#                     reward[agent_id] += self._bonus

#         return obs, reward, terminated, truncated, info


# class RewardShaping2(Wrapper):
#     """
#     General reward shaping wrapper for MultiGrid environments.
#     """

#     def __init__(self, env: MultiGridEnv, scale: float = 1e-3):
#         """
#         Parameters
#         ----------
#         env : MultiGridEnv
#             Environment to wrap
#         scale : float
#             Reward scale
#         """
#         super().__init__(env)
#         self._reward_scale = scale

#     def reset(self, *args, **kwargs):
#         self._picked_up = defaultdict(list) # objects that have been picked up
#         self._opened = [] # doors that have been opened
#         self._seen = defaultdict(set)
#         return super().reset(*args, **kwargs)

#     def step(self, actions):
#         prev_pos = {agent_id: self.agents[agent_id].pos for agent_id in actions}
#         obs, reward, terminated, truncated, info = super().step(actions)

#         for agent_id, action in actions.items():
#             agent = self.env.agents[agent_id]

#             # # Reward new states directly in front of the agent
#             # fwd_pos = agent.front_pos
#             # fwd_obj = self.env.grid.get(*fwd_pos)
#             # fwd_obj = fwd_obj if fwd_obj is None else tuple(fwd_obj)
#             # if can_pickup(fwd_obj):
#             #     fwd_obj += fwd_pos

#             # if fwd_obj not in self._seen[agent_id]:
#             #     self._seen[agent_id].add(fwd_obj)
#             #     reward[agent_id] += self._reward_scale
            
#             # Reward moving in front of new objects
#             fwd_obj = self.env.grid.get(*agent.front_pos)
#             fwd_obj = tuple(fwd_obj) if fwd_obj else None
#             if fwd_obj not in self._seen[agent_id]:
#                 self._seen[agent_id].add(fwd_obj)
#                 reward[agent_id] += self._reward_scale

#             # Reward picking up new objects
#             if agent.carrying:
#                 if agent.carrying not in self._picked_up[agent_id]:
#                     self._picked_up[agent_id].append(agent.carrying)
#                     reward[agent_id] += self._reward_scale
#                     #print("Agent", agent_id, "picked up", agent.carrying)

#             # # Reward opening new doors
#             # if agent_id in actions and action == Action.toggle:
#             #     fwd_obj = self.env.grid.get(*agent.front_pos)
#             #     if isinstance(fwd_obj, Door) and fwd_obj not in self._opened:
#             #         self._opened.append(fwd_obj)
#             #         reward[agent_id] += self._reward_scale            

#             # # Penalize staying in the same position
#             # if agent.pos == prev_pos[agent_id]:
#             #     reward[agent_id] -= 0.01 * self._reward_scale

#         return obs, reward, terminated, truncated, info
    
#     @staticmethod
#     def weighted(weight: float):
#         class WeightedRewardShaping(RewardShaping2):
#             def __init__(self, env):
#                 super().__init__(env, bonus=weight)
#         return WeightedRewardShaping
