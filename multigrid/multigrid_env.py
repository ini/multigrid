from __future__ import annotations

import gymnasium as gym
import hashlib
import math
import numpy as np
import pygame
import pygame.freetype

from abc import abstractmethod
from collections import defaultdict
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from typing import Any, Iterable, SupportsFloat, TypeVar

from .core.actions import Actions
from .core.agent import Agent, AgentState
from .core.constants import COLOR_NAMES, TILE_PIXELS
from .core.grid import Grid
from .core.mission import MissionSpace
from .core.world_object import WorldObj

from .utils.act import handle_actions
from .utils.obs import gen_obs_grid_encoding, gen_obs_grid_vis_mask

T = TypeVar('T')
AgentID = int



class MultiGridEnv(gym.Env):
    """
    Multi-agent 2D gridworld game environment.

    Attributes
    ----------
    agents : list[Agent]
        List of agents in the environment
    grid : Grid
        Environment grid
    action_space : gym.spaces.Dict
        Joint action space of all agents
    observation_space : gym.spaces.Dict
        Joint observation space of all agents
    """
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 20,
    }

    def __init__(
        self,
        mission_space: MissionSpace,
        agents: Iterable[Agent] | int = 1,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        allow_agent_overlap: bool = False,
        is_competitive: bool = True,
        render_mode: str | None = None,
        screen_size: int | None = 1,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False):
        """
        Parameters
        ----------
        agents : int or Iterable[Agent]
            Number of agents in the environment (or provide a list of agents)
        grid_size : int
            Size of the environment grid (width and height)
        width : int
            Width of the environment grid (overrides grid_size)
        height : int
            Height of the environment grid (overrides grid_size)
        max_steps : int
            Maximum number of steps per episode
        see_through_walls : bool
            Whether agents can see through walls
        agent_view_size : int
            Size of agent view (must be odd)
        allow_agent_overlap : bool
            Whether agents are allowed to overlap
        is_competitive : bool
            Whether the task is competitive (i.e. only one agent can get the reward)
        render_mode : str
            Rendering mode (human or rgb_array)
        screen_size : int
            Size of the screen (in tiles)
        highlight : bool
            Whether to highlight the view of each agent when rendering
        tile_size : int
            Width and height of each grid tiles (in pixels)
        """
        # Initialize mission
        self.mission = mission_space.sample()

        # Initialize grid
        width, height = (grid_size, grid_size) if grid_size else (width, height)
        assert width is not None and height is not None
        self.width, self.height = width, height
        self.grid = Grid(width, height)

        # Initialize agents
        if isinstance(agents, int):
            self.num_agents = agents
            self.agent_state = AgentState(agents)
            self.agents: list[Agent] = []
            for i in range(agents):
                agent = Agent(
                    i,
                    mission_space,
                    state=self.agent_state[i],
                    view_size=agent_view_size,
                    see_through_walls=see_through_walls,
                )
                self.agents.append(agent)
        elif isinstance(agents, Iterable):
            assert {agent.index for agent in agents} == set(range(len(agents)))
            self.num_agents = len(agents)
            self.agent_state = AgentState(self.num_agents)
            self.agents: list[Agent] = sorted(agents, key=lambda agent: agent.index)
            for agent in self.agents:
                self.agent_state[agent.index] = agent.state # copy to joint agent state
                agent.state = self.agent_state[agent.index] # reference joint agent state
        else:
            raise ValueError(f"Invalid argument for agents: {agents}")

        # Action enumeration for this environment
        self.actions = Actions
        self.null_actions = self.actions.done * np.ones(self.num_agents, dtype=int)

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
        self.is_competitive = is_competitive

    @property
    def action_space(self) -> spaces.Dict[AgentID, ActType]:
        """
        Return the joint action space of all agents.
        """
        return spaces.Dict({
            agent.index: agent.action_space
            for agent in self.agents
        })

    @property
    def observation_space(self) -> spaces.Dict[AgentID, ObsType]:
        """
        Return the joint observation space of all agents.
        """
        return spaces.Dict({
            agent.index: agent.observation_space
            for agent in self.agents
        })

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] = {}) -> tuple[
            dict[AgentID, ObsType]:
            dict[AgentID, dict[str, Any]]]:
        """
        Reset the environment.
        """
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        for agent in self.agents:
            agent.reset(self.mission)

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)
        self.grid.locations_to_update = -np.ones((self.num_agents, 2), dtype=int)
        self.grid.locations_to_remove = -np.ones((self.num_agents, 2), dtype=int)

        # These fields should be defined by _gen_grid
        assert np.all(self.agent_state.pos >= 0)
        assert np.all(self.agent_state.dir >= 0)

        # Check that agents don't overlap with other objects
        for agent in self.agents:
            start_cell = self.grid.get(*agent.state.pos)
            assert start_cell is None or start_cell.can_overlap()

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == 'human':
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, defaultdict(dict)

    def hash(self, size=16):
        """
        Compute a hash that uniquely identifies the current state of the environment.

        Parameters
        ----------
        size : int
            Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [
            self.grid.encode().tolist(),
            *[agent.state.pos for agent in self.agents],
            *[agent.state.dir for agent in self.agents],
        ]
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        """
        Number of steps remaining in the episode (until truncation).
        """
        return self.max_steps - self.step_count

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
                    output += 2 * AGENT_DIR_TO_STR[tile.dir]
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

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success.
        """
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in range [low, high].
        """
        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in range [low, high].
        """
        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value.
        """
        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list.
        """
        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list.
        """
        lst = list(iterable)
        assert num_elems <= len(lst)

        out: list[T] = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self) -> str:
        """
        Generate a random color name (string).
        """
        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(
        self, x_low: int, x_high: int, y_low: int, y_high: int) -> tuple[int, int]:
        """
        Generate a random (x, y) position tuple.
        """
        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )

    def place_obj(
        self,
        obj: WorldObj | None,
        top: tuple[int, int] = None,
        size: tuple[int, int] = None,
        reject_fn=None,
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
            if any(np.array_equal(pos, agent.state.pos) for agent in self.agents):
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
        pos = (-1, -1)
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        agent.state.pos = pos

        if rand_dir:
            agent.state.dir = self._rand_int(0, 4)

        return pos

    def step(
        self,
        actions: dict[AgentID, ActType]) -> tuple[
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
        actions : dict[AgentID, ActType]
            Action for each agent acting at this timestep

        Returns
        -------
        obs : dict[AgentID, ObsType]
            Observation for each agent
        reward : dict[AgentID, SupportsFloat]
            Reward for each agent
        terminated : dict[AgentID, bool]
            Whether the episode has been terminated for each agent (success or failure)
        truncated : dict[AgentID, bool]
            Whether the episode has been truncated for each agent (max steps reached)
        info : dict[AgentID, dict[str, Any]]
            Additional information for each agent
        """
        self.step_count += 1

        action_array = self.null_actions.copy()
        for i, action in actions.items():
            action_array[i] = action

        # Randomize agent action order
        if self.num_agents == 1:
            order = np.zeros(1, dtype=int)
        else:
            order = self.np_random.random(size=self.num_agents).argsort()

        # Update agent state and grid state from actions
        reward = handle_actions(
            action_array,
            order,
            self.agent_state,
            self.grid.state,
            self.grid.needs_update,
            self.grid.locations_to_update,
            self.grid.needs_remove,
            self.grid.locations_to_remove,
            allow_agent_overlap=self.allow_agent_overlap,
            is_competitive=self.is_competitive,
        )

        # Update world objects in grid
        if self.grid.needs_update:
            self.grid.update_world_objects()
        if self.grid.needs_remove:
            self.grid.remove_world_objects()

        # Rendering
        if self.render_mode == 'human':
            self.render()

        # Generate outputs
        obs = self.gen_obs()
        reward *= self._reward()
        reward = dict(enumerate(reward))
        truncated = self.step_count >= self.max_steps
        truncated = dict(enumerate([truncated] * self.num_agents))
        terminated = dict(enumerate(self.agent_state.terminated))

        return obs, reward, terminated, truncated, defaultdict(dict)

    def gen_obs(self) -> dict[AgentID, ObsType]:
        """
        Generate observations for each agent (partially observable, low-res encoding).

        Returns
        -------
        obs : dict[AgentID, dict]
            Mapping from agent ID to observation dict, containing:
                - 'image': partially observable view of the environment
                - 'direction': agent's direction/orientation (acting as a compass)
                - 'mission': textual mission string (instructions for the agent)
        """
        direction = self.agent_state.dir
        image = gen_obs_grid_encoding(
            self.grid.state,
            self.agent_state,
            self.agents[0].view_size,
            self.agents[0].see_through_walls,
        )

        obs = {}
        for i in range(self.num_agents):
            obs[i] = {
                'image': image[i],
                'direction': direction[i],
                'mission': self.mission,
            }
        return obs

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
        vis_masks = gen_obs_grid_vis_mask(
            self.grid.state, self.agent_state, self.agents[0].view_size)

        # Mask of which cells to highlight
        highlight_mask = np.zeros((self.width, self.height), dtype=bool)

        for agent in self.agents:
            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = agent.dir_vec
            r_vec = agent.right_vec
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
        agent_pov: bool = False):
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
        frame: np.ndarray of shape (H, W, 3)
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
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption('multigrid')
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

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            text = self.mission
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
