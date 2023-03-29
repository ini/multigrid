import numpy as np

from gymnasium import spaces
from typing import Optional

from .actions import Actions
from .array import EMPTY
from .constants import COLORS, DIR_TO_VEC, IDX_TO_COLOR, OBJECT_TO_IDX
from .grid import Grid
from .mission import MissionSpace
from .world_object import WorldObj

from ..utils.numba import gen_obs_grid, gen_obs_grid_encoding
from ..utils.misc import get_view_exts, front_pos
from ..utils.rendering import (
    fill_coords,
    point_in_triangle,
    rotate_fn,
)



class Agent(WorldObj):
    """
    Class representing an agent in the environment.
    """

    def __init__(
        self,
        id: int,
        mission_space: MissionSpace,
        view_size: int = 7,
        see_through_walls: bool = False):
        """
        Parameters
        ----------
        id : int
            Unique ID for the agent in the environment
        mission_space : MissionSpace
            The mission space for the agent
        view_size : int
            The size of the agent's view (must be odd)
        see_through_walls : bool
            Whether the agent can see through walls
        """
        color = IDX_TO_COLOR[id % (max(IDX_TO_COLOR) + 1)]
        super().__init__('agent', color)
        self.id = id

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(Actions))

        # Number of cells (width and height) in the agent view
        assert view_size % 2 == 1
        assert view_size >= 3
        self.view_size = view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(view_size, view_size, 3),
                dtype='uint8',
            ),
            'direction': spaces.Discrete(len(DIR_TO_VEC)),
            'mission_space': mission_space,
        })
        self.see_through_walls = see_through_walls

        # Current agent state
        self.mission: str = None
        self.terminated: bool = False
        self.carrying: Optional[WorldObj] = None
        self.pos: np.ndarray | tuple[int, int] = None
        self.dir: int = None

    def reset(self, mission: str = 'maximize reward'):
        """
        Reset the agent before environment episode.
        """
        self.mission = mission
        self.terminated = False
        self.carrying = None
        self.pos = (-1, -1)
        self.dir = -1

    @property
    def dir_vec(self) -> np.ndarray[int]:
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert (
            0 <= self.dir < 4
        ), f"Invalid direction: {self.dir} is not within range(0, 4)"
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self) -> np.ndarray[int]:
        """
        Get the vector pointing to the right of the agent.
        """
        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self) -> tuple[int, int]:
        """
        Get the position of the cell that is right in front of the agent.
        """
        return front_pos(self.pos, self.dir)

    def get_view_coords(self, i, j) -> tuple[int, int]:
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """
        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def relative_coords(self, x: int, y: int) -> Optional[tuple[int, int]]:
        """
        Check if a grid position belongs to the agent's field of view,
        and returns the corresponding coordinates.
        """
        vx, vy = self.get_view_coords(x, y)

        if not (0 <= vx < self.view_size) or not (0 <= vy < self.view_size):
            return None

        return vx, vy

    def in_view(self, x: int, y: int) -> bool:
        """
        Check if a grid position is visible to the agent.
        """
        return self.relative_coords(x, y) is not None

    def sees(self, x: int, y: int, grid: Grid) -> bool:
        """
        Check if a non-empty grid position is visible to the agent.
        """
        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs_grid, _ = self.gen_obs_grid(grid)
        obs_cell = obs_grid.get(vx, vy)
        world_cell = grid.get(x, y)

        assert world_cell is not None
        return obs_cell is not None and obs_cell.type == world_cell.type

    def gen_obs_grid(self, grid_array: np.ndarray[int]) -> tuple[Grid, np.ndarray[bool]]:
        """
        Generate the sub-grid observed by the agent.

        Returns
        -------
        grid : Grid
            Grid of partially observable view of the environment
        vis_mask : np.ndarray[bool]
            Mask indicating which grid cells are visible to the agent
        """
        topX, topY, _, _ = get_view_exts(self.dir, self.pos, self.view_size)
        obs_grid_result = gen_obs_grid(
            grid_array,
            EMPTY if self.carrying is None else self.carrying.array,
            self.dir,
            topX, topY,
            self.view_size, self.view_size,
            self.see_through_walls,
        )
        grid_array = obs_grid_result[..., :-1].astype(int)
        vis_mask = obs_grid_result[..., -1].astype(bool)
        return Grid.from_grid_array(grid_array), vis_mask

    def gen_obs(self, grid_array: np.ndarray[int]) -> dict:
        """
        Generate the agent's view (partially observable, low-resolution encoding).

        Returns
        -------
        obs : dict
            - 'image': partially observable view of the environment
            - 'direction': agent's direction/orientation (acting as a compass)
            - 'mission': textual mission string (instructions for the agent)
        """
        topX, topY, _, _ = get_view_exts(self.dir, self.pos, self.view_size)
        image = gen_obs_grid_encoding(
            grid_array,
            EMPTY if self.carrying is None else self.carrying.array,
            self.dir,
            topX, topY,
            self.view_size, self.view_size,
            self.see_through_walls,
        )
        obs = {'image': image, 'direction': self.dir, 'mission': self.mission}
        return obs

    def encode(self):
        """
        Encode the agent's state as a numpy array.
        """
        return (OBJECT_TO_IDX[self.type], self.id, self.dir)

    def render(self, img: np.ndarray[int]):
        """
        Draw the agent.
        """
        c = COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * self.dir)
        fill_coords(img, tri_fn, c)
