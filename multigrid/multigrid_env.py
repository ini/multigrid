from __future__ import annotations

import gymnasium as gym
import hashlib
import math
import numpy as np
import pygame
import pygame.freetype

from abc import abstractmethod
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from typing import Any, Iterable, SupportsFloat, TypeVar

from .core.actions import Actions
from .core.agent import Agent
from .core.constants import COLOR_NAMES, TILE_PIXELS
from .core.grid import Grid
from .core.mission import MissionSpace
from .core.world_object import Point, WorldObj

T = TypeVar("T")



class MultiGridEnv(gym.Env):
    """
    Multi-agent 2D gridworld game environment.

    Attributes
    ----------
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        mission_space: MissionSpace,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 1,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        # Initialize mission
        self.mission = mission_space.sample()

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None

        # Initialize agents
        self.agents = []
        for i in range(2):
            agent = Agent(
                i,
                mission_space,
                view_size=agent_view_size,
                see_through_walls=see_through_walls,
            )
            self.agents.append(agent)

        # Action enumeration for this environment
        self.actions = Actions

        self.action_space = self.agents[0].action_space
        self.observation_space = spaces.Dict({
            agent.index: agent.observation_space
            for agent in self.agents
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Environment configuration
        self.width = width
        self.height = height
        self.grid = Grid(width, height)

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        for agent in self.agents:
            agent.reset(self.mission)

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for agent in self.agents:
            assert (
                agent.pos >= (0, 0)
                if isinstance(agent.pos, tuple)
                else all(agent.pos >= 0) and agent.dir >= 0
            )

        # Check that agents don't overlap with other objects
        for agent in self.agents:
            start_cell = self.grid.get(*agent.pos)
            assert start_cell is None or start_cell.can_overlap()

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    # def hash(self, size=16):
    #     """Compute a hash that uniquely identifies the current state of the environment.
    #     :param size: Size of the hashing
    #     """
    #     sample_hash = hashlib.sha256()

    #     to_encode = [self.grid.encode().tolist(), self.agent.pos, self.agent.dir]
    #     for item in to_encode:
    #         sample_hash.update(str(item).encode("utf8"))

    #     return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    # def __str__(self):
    #     """
    #     Produce a pretty string of the environment's grid along with the agent.
    #     A grid cell is represented by 2-character string, the first one for
    #     the object and the second one for the color.
    #     """

    #     # Map of object types to short string
    #     OBJECT_TO_STR = {
    #         "wall": "W",
    #         "floor": "F",
    #         "door": "D",
    #         "key": "K",
    #         "ball": "A",
    #         "box": "B",
    #         "goal": "G",
    #         "lava": "V",
    #     }

    #     # Map agent's direction to short string
    #     AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

    #     output = ""

    #     for j in range(self.grid.height):
    #         for i in range(self.grid.width):
    #             if i == self.agent.pos[0] and j == self.agent.pos[1]:
    #                 output += 2 * AGENT_DIR_TO_STR[self.agent.dir]
    #                 continue

    #             tile = self.grid.get(i, j)

    #             if tile is None:
    #                 output += "  "
    #                 continue

    #             if tile.type == "door":
    #                 if tile.is_open:
    #                     output += "__"
    #                 elif tile.is_locked:
    #                     output += "L" + tile.color[0].upper()
    #                 else:
    #                     output += "D" + tile.color[0].upper()
    #                 continue

    #             output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

    #         if j < self.grid.height - 1:
    #             output += "\n"

    #     return output

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
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )

    def place_obj(
        self,
        obj: WorldObj | None,
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
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

            # Don't place the object where agent are
            if any(np.array_equal(pos, agent.pos) for agent in self.agents):
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
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """
        for agent in self.agents:
            agent.pos = (-1, -1)
            pos = self.place_obj(None, top, size, max_tries=max_tries)
            agent.pos = pos

            if rand_dir:
                agent.dir = self._rand_int(0, 4)

            #return pos

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = {agent.index: False for agent in self.agents}
        truncated = {agent.index: False for agent in self.agents}

        for i, action in actions.items():
            agent = self.agents[i]

            # Get the position in front of the agent
            fwd_pos = agent.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Rotate left
            if action == self.actions.left:
                agent.dir = (agent.dir - 1) % 4

            # Rotate right
            elif action == self.actions.right:
                agent.dir = (agent.dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    agent.pos = fwd_pos
                if fwd_cell is not None and fwd_cell.type == "goal":
                    terminated[i] = True
                    reward = self._reward()
                if fwd_cell is not None and fwd_cell.type == "lava":
                    terminated[i] = True

            # Pick up an object
            elif action == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if agent.carrying is None:
                        agent.carrying = fwd_cell
                        agent.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)

            # Drop an object
            elif action == self.actions.drop:
                if not fwd_cell and agent.carrying:
                    self.grid.set(fwd_pos[0], fwd_pos[1], agent.carrying)
                    agent.carrying.cur_pos = fwd_pos
                    agent.carrying = None

            # Toggle/activate an object
            elif action == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, agent, fwd_pos)

            # Done action (not used by default)
            elif action == self.actions.done:
                pass

            else:
                raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = {agent.index: True for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
    
    def thing(self):
        grid = Grid.from_grid_array(self.grid.array.copy())
        for agent in self.agents:
            grid.set(*agent.pos, agent)
        return grid

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        return {agent.index: agent.gen_obs(self.grid) for agent in self.agents}

    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        raise NotImplementedError

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-partial observation for visualization
        """
        # Compute which cells are visible to the agent
        griddy = self.thing()
        vis_masks = {agent.index: agent.gen_obs_grid(griddy)[1] for agent in self.agents}
        #_, vis_mask = self.agent.gen_obs_grid(self.thing())

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        for agent in self.agents:
            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = agent.dir_vec
            r_vec = agent.right_vec
            top_left = (
                agent.pos
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
        grid = self.thing()
        img = grid.render(
            tile_size,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
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

            print('one')
            font_size = 22
            text = self.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)
            print('two')

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            pygame.quit()
