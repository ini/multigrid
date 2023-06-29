from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import pygame
import pygame.freetype

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from gymnasium import spaces
from itertools import repeat
from numpy.typing import NDArray as ndarray
from typing import Any, Callable, Iterable, Literal, SupportsFloat

from .core.actions import Action
from .core.agent import Agent, AgentState
from .core.constants import Type, TILE_PIXELS
from .core.grid import Grid
from .core.mission import MissionSpace
from .core.world_object import WorldObj
from .utils.obs import gen_obs_grid_encoding
from .utils.random import RandomMixin



### Typing

AgentID = int
ObsType = dict[str, Any]



### Environment

class MultiGridEnv(gym.Env, RandomMixin, ABC):
    """
    Base class for multi-agent 2D gridworld environments.

    :Agents:

        The environment can be configured with any fixed number of agents.
        Agents are represented by :class:`.Agent` instances, and are
        identified by their index, from ``0`` to ``len(env.agents) - 1``.

    :Observation Space:

        The multi-agent observation space is a Dict mapping from agent index to
        corresponding agent observation space.

        The standard agent observation is a dictionary with the following entries:

            * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
                Encoding of the agent's view of the environment,
                where each grid object is encoded as a 3 dimensional tuple:
                (:class:`.Type`, :class:`.Color`, :class:`.State`)
            * direction : int
                Agent's direction (0: right, 1: down, 2: left, 3: up)
            * mission : Mission
                Task string corresponding to the current environment configuration

    :Action Space:

        The multi-agent action space is a Dict mapping from agent index to
        corresponding agent action space.

        Agent actions are discrete integers, as enumerated in :class:`.Action`.

    Attributes
    ----------
    agents : list[Agent]
        List of agents in the environment
    grid : Grid
        Environment grid
    observation_space : spaces.Dict[AgentID, spaces.Space]
        Joint observation space of all agents
    action_space : spaces.Dict[AgentID, spaces.Space]
        Joint action space of all agents
    """
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 20,
    }

    def __init__(
        self,
        mission_space: MissionSpace | str = "maximize reward",
        agents: Iterable[Agent] | int = 1,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        allow_agent_overlap: bool = True,
        joint_reward: bool = False,
        success_termination_mode: Literal['any', 'all'] = 'any',
        failure_termination_mode: Literal['any', 'all'] = 'all',
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False):
        """
        Parameters
        ----------
        mission_space : MissionSpace
            Space of mission strings (i.e. agent instructions)
        agents : int or Iterable[Agent]
            Number of agents in the environment (or provide :class:`Agent` instances)
        grid_size : int
            Size of the environment grid (width and height)
        width : int
            Width of the environment grid (if `grid_size` is not provided)
        height : int
            Height of the environment grid (if `grid_size` is not provided)
        max_steps : int
            Maximum number of steps per episode
        see_through_walls : bool
            Whether agents can see through walls
        agent_view_size : int
            Size of agent view (must be odd)
        allow_agent_overlap : bool
            Whether agents are allowed to overlap
        joint_reward : bool
            Whether all agents receive the same joint reward
        success_termination_mode : 'any' or 'all'
            Whether to terminate when any agent completes its mission
            or when all agents complete their missions
        failure_termination_mode : 'any' or 'all'
            Whether to terminate when any agent fails its mission
            or when all agents fail their missions
        render_mode : str
            Rendering mode (human or rgb_array)
        screen_size : int
            Width and height of the rendering window (in pixels)
        highlight : bool
            Whether to highlight the view of each agent when rendering
        tile_size : int
            Width and height of each grid tiles (in pixels)
        """
        gym.Env.__init__(self)
        RandomMixin.__init__(self, self.np_random)

        # Initialize mission space
        if isinstance(mission_space, str):
            self.mission_space = MissionSpace.from_string(mission_space)
        else:
            self.mission_space = mission_space

        # Initialize grid
        width, height = (grid_size, grid_size) if grid_size else (width, height)
        assert width is not None and height is not None
        self.width, self.height = width, height
        self.grid: Grid = Grid(width, height)

        # Initialize agents
        if isinstance(agents, int):
            self.num_agents = agents
            self.agent_states = AgentState(agents) # joint agent state (vectorized)
            self.agents: list[Agent] = []
            for i in range(agents):
                agent = Agent(
                    index=i,
                    mission_space=self.mission_space,
                    view_size=agent_view_size,
                    see_through_walls=see_through_walls,
                )
                agent.state = self.agent_states[i]
                self.agents.append(agent)
        elif isinstance(agents, Iterable):
            assert {agent.index for agent in agents} == set(range(len(agents)))
            self.num_agents = len(agents)
            self.agent_states = AgentState(self.num_agents)
            self.agents: list[Agent] = sorted(agents, key=lambda agent: agent.index)
            for agent in self.agents:
                self.agent_states[agent.index] = agent.state # copy to joint agent state
                agent.state = self.agent_states[agent.index] # reference joint agent state
        else:
            raise ValueError(f"Invalid argument for agents: {agents}")

        # Action enumeration for this environment
        self.actions = Action

        # Range of possible rewards
        self.reward_range = (0, 1)

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov
        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Other
        self.allow_agent_overlap = allow_agent_overlap
        self.joint_reward = joint_reward
        self.success_termination_mode = success_termination_mode
        self.failure_termination_mode = failure_termination_mode

    @cached_property
    def observation_space(self) -> spaces.Dict[AgentID, spaces.Space]:
        """
        Return the joint observation space of all agents.
        """
        return spaces.Dict({
            agent.index: agent.observation_space
            for agent in self.agents
        })

    @cached_property
    def action_space(self) -> spaces.Dict[AgentID, spaces.Space]:
        """
        Return the joint action space of all agents.
        """
        return spaces.Dict({
            agent.index: agent.action_space
            for agent in self.agents
        })

    @abstractmethod
    def _gen_grid(self, width: int, height: int):
        """
        :meta public:

        Generate the grid for a new episode.

        This method should:

        * Set ``self.grid`` and populate it with :class:`.WorldObj` instances
        * Set the positions and directions of each agent

        Parameters
        ----------
        width : int
            Width of the grid
        height : int
            Height of the grid
        """
        pass

    def reset(
        self, seed: int | None = None, **kwargs) -> tuple[
            dict[AgentID, ObsType]:
            dict[AgentID, dict[str, Any]]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int or None
            Seed for random number generator

        Returns
        -------
        observations : dict[AgentID, ObsType]
            Observation for each agent
        infos : dict[AgentID, dict[str, Any]]
            Additional information for each agent
        """
        super().reset(seed=seed, **kwargs)

        # Reset agents
        self.mission_space.seed(seed)
        self.mission = self.mission_space.sample()
        for agent in self.agents:
            agent.reset(mission=self.mission)

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert np.all(self.agent_states.pos >= 0)
        assert np.all(self.agent_states.dir >= 0)

        # Check that agents don't overlap with other objects
        for agent in self.agents:
            start_cell = self.grid.get(*agent.state.pos)
            assert start_cell is None or start_cell.can_overlap()

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        observations = self.gen_obs()

        # Render environment
        if self.render_mode == 'human':
            self.render()

        return observations, defaultdict(dict)

    def step(
        self,
        actions: dict[AgentID, Action]) -> tuple[
            dict[AgentID, ObsType],
            dict[AgentID, SupportsFloat],
            dict[AgentID, bool],
            dict[AgentID, bool],
            dict[AgentID, dict[str, Any]]]:
        """
        Run one timestep of the environment’s dynamics
        using the provided agent actions.

        Parameters
        ----------
        actions : dict[AgentID, Action]
            Action for each agent acting at this timestep

        Returns
        -------
        observations : dict[AgentID, ObsType]
            Observation for each agent
        rewards : dict[AgentID, SupportsFloat]
            Reward for each agent
        terminations : dict[AgentID, bool]
            Whether the episode has been terminated for each agent (success or failure)
        truncations : dict[AgentID, bool]
            Whether the episode has been truncated for each agent (max steps reached)
        infos : dict[AgentID, dict[str, Any]]
            Additional information for each agent
        """
        self.step_count += 1
        rewards = self.handle_actions(actions)

        # Generate outputs
        observations = self.gen_obs()
        terminations = dict(enumerate(self.agent_states.terminated))
        truncated = self.step_count >= self.max_steps
        truncations = dict(enumerate(repeat(truncated, self.num_agents)))

        # Rendering
        if self.render_mode == 'human':
            self.render()

        return observations, rewards, terminations, truncations, defaultdict(dict)

    def gen_obs(self) -> dict[AgentID, ObsType]:
        """
        Generate observations for each agent (partially observable, low-res encoding).

        Returns
        -------
        observations : dict[AgentID, ObsType]
            Mapping from agent ID to observation dict, containing:
                * 'image': partially observable view of the environment
                * 'direction': agent's direction / orientation (acting as a compass)
                * 'mission': textual mission string (instructions for the agent)
        """
        direction = self.agent_states.dir
        image = gen_obs_grid_encoding(
            self.grid.state,
            self.agent_states,
            self.agents[0].view_size,
            self.agents[0].see_through_walls,
        )

        observations = {}
        for i in range(self.num_agents):
            observations[i] = {
                'image': image[i],
                'direction': direction[i],
                'mission': self.agents[i].mission,
            }

        return observations

    def handle_actions(
        self, actions: dict[AgentID, Action]) -> dict[AgentID, SupportsFloat]:
        """
        Handle actions taken by agents.

        Parameters
        ----------
        actions : dict[AgentID, Action]
            Action for each agent acting at this timestep

        Returns
        -------
        rewards : dict[AgentID, SupportsFloat]
            Reward for each agent
        """
        rewards = {agent_index: 0 for agent_index in range(self.num_agents)}

        # Randomize agent action order
        if self.num_agents == 1:
            order = (0,)
        else:
            order = self.np_random.random(size=self.num_agents).argsort()

        # Update agent states, grid states, and reward from actions
        for i in order:
            agent, action = self.agents[i], actions[i]

            if agent.state.terminated:
                continue

            # Rotate left
            if action == Action.left:
                agent.state.dir = (agent.state.dir - 1) % 4

            # Rotate right
            elif action == Action.right:
                agent.state.dir = (agent.state.dir + 1) % 4

            # Move forward
            elif action == Action.forward:
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)

                if fwd_obj is None or fwd_obj.can_overlap():
                    if not self.allow_agent_overlap:
                        agent_present = np.bitwise_and.reduce(
                            self.agent_states.pos == fwd_pos, axis=1).any()
                        if agent_present:
                            continue

                    agent.state.pos = fwd_pos
                    if fwd_obj is not None:
                        if fwd_obj.type == Type.goal:
                            self.on_success(agent, rewards, {})
                        if fwd_obj.type == Type.lava:
                            self.on_failure(agent, rewards, {})

            # Pick up an object
            elif action == Action.pickup:
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)

                if fwd_obj is not None and fwd_obj.can_pickup():
                    if agent.state.carrying is None:
                        agent.state.carrying = fwd_obj
                        self.grid.set(*fwd_pos, None)

            # Drop an object
            elif action == Action.drop:
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)

                if agent.state.carrying and fwd_obj is None:
                    agent_present = np.bitwise_and.reduce(
                        self.agent_states.pos == fwd_pos, axis=1).any()
                    if not agent_present:
                        self.grid.set(*fwd_pos, agent.state.carrying)
                        agent.state.carrying.cur_pos = fwd_pos
                        agent.state.carrying = None

            # Toggle/activate an object
            elif action == Action.toggle:
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)

                if fwd_obj is not None:
                    fwd_obj.toggle(self, agent, fwd_pos)

            # Done action (not used by default)
            elif action == Action.done:
                pass

            else:
                raise ValueError(f"Unknown action: {action}")

        return rewards

    def on_success(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool]):
        """
        Callback for when an agent completes its mission.

        Parameters
        ----------
        agent : Agent
            Agent that completed its mission
        rewards : dict[AgentID, SupportsFloat]
            Reward dictionary to be updated
        terminations : dict[AgentID, bool]
            Termination dictionary to be updated
        """
        if self.success_termination_mode == 'any':
            self.agent_states.terminated = True # terminate all agents
            for i in range(self.num_agents):
                terminations[i] = True
        else:
            agent.state.terminated = True # terminate this agent only
            terminations[agent.index] = True

        if self.joint_reward:
            for i in range(self.num_agents):
                rewards[i] = self._reward() # reward all agents
        else:
            rewards[agent.index] = self._reward() # reward this agent only

    def on_failure(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool]):
        """
        Callback for when an agent fails its mission prematurely.

        Parameters
        ----------
        agent : Agent
            Agent that failed its mission
        rewards : dict[AgentID, SupportsFloat]
            Reward dictionary to be updated
        terminations : dict[AgentID, bool]
            Termination dictionary to be updated
        """
        if self.failure_termination_mode == 'any':
            self.agent_states.terminated = True # terminate all agents
            for i in range(self.num_agents):
                terminations[i] = True
        else:
            agent.state.terminated = True # terminate this agent only
            terminations[agent.index] = True

    def is_done(self) -> bool:
        """
        Return whether the current episode is finished (for all agents).
        """
        truncated = self.step_count >= self.max_steps
        return truncated or all(self.agent_states.terminated)

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall': 'W',
            'floor': 'F',
            'door': 'D',
            'key': 'K',
            'ball': 'A',
            'box': 'B',
            'goal': 'G',
            'lava': 'V',
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: '>', 1: 'V', 2: '<', 3: '^'}

        # Get agent locations
        location_to_agent = {tuple(agent.pos): agent for agent in self.agents}

        output = ""
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                if (i, j) in location_to_agent:
                    output += 2 * AGENT_DIR_TO_STR[location_to_agent[i, j].dir]
                    continue

                tile = self.grid.get(i, j)

                if tile is None:
                    output += '  '
                    continue

                if tile.type == 'agent':
                    output += 2 * AGENT_DIR_TO_STR[tile.dir]
                    continue

                if tile.type == 'door':
                    if tile.is_open:
                        output += '__'
                    elif tile.is_locked:
                        output += 'L' + tile.color[0].upper()
                    else:
                        output += 'D' + tile.color[0].upper()
                    continue

                output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

            if j < self.grid.height - 1:
                output += '\n'

        return output

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success.
        """
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def place_obj(
        self,
        obj: WorldObj | None,
        top: tuple[int, int] = None,
        size: tuple[int, int] = None,
        reject_fn: Callable[[MultiGridEnv, tuple[int, int]], bool] | None = None,
        max_tries=math.inf) -> tuple[int, int]:
        """
        Place an object at an empty position in the grid.

        Parameters
        ----------
        obj: WorldObj
            Object to place in the grid
        top: tuple[int, int]
            Top-left position of the rectangular area where to place the object
        size: tuple[int, int]
            Width and height of the rectangular area where to place the object
        reject_fn: Callable(env, pos) -> bool
            Function to filter out potential positions
        max_tries: int
            Maximum number of attempts to place the object
        """
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where agents are
            if np.bitwise_and.reduce(self.agent_states.pos == pos, axis=1).any():
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """
        Put an object at a specific position in the grid.
        """
        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self,
        agent: Agent,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf) -> tuple[int, int]:
        """
        Set agent starting point at an empty position in the grid.
        """
        agent.state.pos = (-1, -1)
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        agent.state.pos = pos

        if rand_dir:
            agent.state.dir = self._rand_int(0, 4)

        return pos

    def get_pov_render(self, *args, **kwargs):
        """
        Render an agent's POV observation for visualization.
        """
        raise NotImplementedError(
            "POV rendering not supported for multiagent environments."
        )

    def get_full_render(self, highlight: bool, tile_size: int):
        """
        Render a non-partial observation for visualization.
        """
        # Compute agent visibility masks
        obs_shape = self.agents[0].observation_space['image'].shape[:-1]
        vis_masks = np.zeros((self.num_agents, *obs_shape), dtype=bool)
        for i, agent_obs in self.gen_obs().items():
            vis_masks[i] = (agent_obs['image'][..., 0] != Type.unseen.to_index())

        # Mask of which cells to highlight
        highlight_mask = np.zeros((self.width, self.height), dtype=bool)

        for agent in self.agents:
            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = agent.state.dir.to_vec()
            r_vec = np.array((-f_vec[1], f_vec[0]))
            top_left = (
                agent.state.pos
                + f_vec * (agent.view_size - 1)
                - r_vec * (agent.view_size // 2)
            )

            # For each cell in the visibility mask
            for vis_j in range(0, agent.view_size):
                for vis_i in range(0, agent.view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_masks[agent.index][vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            agents=self.agents,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False) -> ndarray[np.uint8]:
        """
        Returns an RGB image corresponding to the whole environment.

        Parameters
        ----------
        highlight: bool
            Whether to highlight agents' field of view (with a lighter gray color)
        tile_size: int
            How many pixels will form a tile from the NxM grid
        agent_pov: bool
            Whether to render agent's POV or the full environment

        Returns
        -------
        frame: ndarray of shape (H, W, 3)
            A frame representing RGB values for the HxW pixel image
        """
        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self):
        """
        Render the environment.
        """
        img = self.get_frame(self.highlight, self.tile_size)

        if self.render_mode == 'human':
            img = np.transpose(img, axes=(1, 0, 2))
            screen_size = (
                self.screen_size * min(img.shape[0] / img.shape[1], 1.0),
                self.screen_size * min(img.shape[1] / img.shape[0], 1.0),
            )
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                pygame.display.set_caption(f'multigrid - {self.__class__.__name__}')
                self.window = pygame.display.set_mode(screen_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, screen_size)

            font_size = 22
            text = str(self.mission)
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()

        elif self.render_mode == 'rgb_array':
            return img

    def close(self):
        """
        Close the rendering window.
        """
        if self.window:
            pygame.quit()
