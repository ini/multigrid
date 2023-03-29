import gymnasium as gym
import math
import numpy as np
import pygame
import pygame.freetype

from abc import abstractmethod
from typing import Iterable, TypeVar

from .core.constants import COLOR_NAMES, TILE_PIXELS
from .core.grid import Grid
from .core.mission import MissionSpace
from .core.world_object import Point, WorldObj

T = TypeVar('T')



class BaseEnv(gym.Env):
    """
    Multi-agent 2D gridworld game environment.

    Attributes
    ----------
    agents : dict[AgentID, Agent]
        Dictionary of agents in the environment
    grid : Grid
        Environment grid
    action_space : gym.spaces.Dict
        Joint action space of all agents
    observation_space : gym.spaces.Dict
        Joint observation space of all agents
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }

    def __init__(
        self,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        render_mode: str | None = None,
        screen_size: int | None = 1,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS):
        """

        """
        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width, height = grid_size, grid_size
        assert width is not None and height is not None

        # Initialize grid
        self.width, self.height = width, height
        self.grid = Grid(width, height)

        # Rendering attributes
        self.render_mode = render_mode
        self.screen_size = screen_size
        self.highlight = highlight
        self.tile_size = tile_size
        self.render_size = None
        self.window = None
        self.clock = None

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
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf):
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
        top = (0, 0) if top is None else (max(top[0], 0), max(top[1], 0))
        size = (self.grid.width, self.grid.height) if size is None else size

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

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos, obj.cur_pos = pos, pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """
        Put an object at a specific position in the grid.
        """
        self.grid.set(i, j, obj)
        obj.init_pos, obj.cur_pos = (i, j), (i, j)

    def get_frame(self, highlight: bool = True, tile_size: int = TILE_PIXELS):
        """
        Returns an RGB image corresponding to the whole environment.

        Parameters
        ----------
        highlight: bool
            Whether to highlight agents' field of view (with a lighter gray color)
        tile_size: int
            How many pixels will form a tile from the NxM grid
        
        Returns
        -------
        frame: np.ndarray of shape (H, W, 3)
            A frame representing RGB values for the HxW pixel image
        """
        raise NotImplementedError

    def render(self):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

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
        if self.window:
            pygame.quit()
