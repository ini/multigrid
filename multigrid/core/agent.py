import numpy as np

from gymnasium import spaces
from typing import Optional

from .actions import Actions
from .constants import COLORS, DIR_TO_VEC, COLOR_TO_IDX, IDX_TO_COLOR, OBJECT_TO_IDX
from .grid import Grid
from .mission import MissionSpace
from .world_object import WorldObj, WorldObjState

from ..utils.numba import gen_obs_grid, gen_obs_grid_encoding
from ..utils.misc import get_view_exts, front_pos
from ..utils.rendering import (
    fill_coords,
    point_in_triangle,
    rotate_fn,
)



class AgentState(np.ndarray):
    """
    Array representation of an agent state.

    The state is a 6-dimensional array, indexed as follows:
        * 0: x position
        * 1: y position
        * 2: direction
        * 3: color
        * 4: terminated
        * 5: carried object

    AgentState objects can also be vectorized, in which case the first
    (n - 1) dimensions index a batch of states, and the last dimension
    represents each index in a state array.

    Attributes
    ----------
    pos : np.ndarray[int] of shape (2,)
        Agent (x, y) position
    dir : int
        Agent direction
    color : str
        Agent color
    terminated : bool
        Whether the agent has terminated
    carrying : WorldObj or None
        The object the agent is carrying
    """
    dim = 6
    color_to_idx = np.vectorize(COLOR_TO_IDX.get)
    idx_to_color = np.vectorize(IDX_TO_COLOR.get)

    def __new__(cls, *dims: int):
        obj = super().__new__(cls, shape=dims+(cls.dim,), dtype=object)
        obj[..., :3] = -1 # pos & dir
        obj[..., 3] = np.arange(np.prod(dims), dtype=int).reshape(*dims) # color
        obj[..., 3] %= len(COLOR_TO_IDX)
        obj[..., 4] = False # terminated
        obj[..., 5] = None # carrying
        return obj

    def to_world_state(self) -> WorldObjState:
        out = WorldObjState.empty(*self.shape[:-1])
        out[..., :3] = (OBJECT_TO_IDX['agent'], COLOR_TO_IDX[self.color], self.dir)
        return out

    @property
    def pos(self) -> np.ndarray[int]:
        return self[..., :2]

    @pos.setter
    def pos(self, value: np.ndarray[int]):
        self[..., :2] = value

    @property
    def dir(self) -> int:
        out = self[..., 2]
        return out.item() if out.ndim == 0 else out

    @dir.setter
    def dir(self, value: int):
        self[..., 2] = value

    @property
    def color(self) -> str:
        out = self.idx_to_color(self[..., 3])
        return out.item() if out.ndim == 0 else out

    @color.setter
    def color(self, value: str):
        self[..., 3] = self.color_to_idx(value)

    @property
    def terminated(self) -> bool:
        out = self[..., 4].astype(bool)
        return out.item() if out.ndim == 0 else out

    @terminated.setter
    def terminated(self, value: bool):
        self[..., 4] = value

    @property
    def carrying(self) -> Optional[WorldObj]:
        obj = self[..., 5]
        return obj.item() if obj.ndim == 0 else obj

    @carrying.setter
    def carrying(self, obj: Optional[WorldObj]):
        self[..., 5] = obj



class Agent:
    """
    Class representing an agent in the environment.
    """

    def __init__(
        self,
        id: int,
        state: AgentState,
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
        self.id = id
        self.state = state

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

    def reset(self, mission: str = 'maximize reward'):
        """
        Reset the agent before environment episode.
        """
        self.mission = mission
        self.state.pos = (-1, -1)
        self.state.dir = -1
        self.state.terminated = False
        self.state.carrying = None

    @property
    def dir_vec(self) -> np.ndarray[int]:
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert (
            0 <= self.state.dir < 4
        ), f"Invalid direction: {self.state.dir} is not within range(0, 4)"
        return DIR_TO_VEC[self.state.dir]

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
        return front_pos(tuple(self.state.pos), self.state.dir)

    def get_view_coords(self, i, j) -> tuple[int, int]:
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """
        ax, ay = self.state.pos
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

    def gen_obs_grid(self, grid_state: WorldObjState) -> tuple[Grid, np.ndarray[bool]]:
        """
        Generate the sub-grid observed by the agent.

        Returns
        -------
        grid : Grid
            Grid of partially observable view of the environment
        vis_mask : np.ndarray[bool]
            Mask indicating which grid cells are visible to the agent
        """
        topX, topY, _, _ = get_view_exts(
            self.state.dir, tuple(self.state.pos), self.view_size)
        carrying = self.state.carrying
        obs_grid_result = gen_obs_grid(
            grid_state.encode(),
            WorldObjState.empty().encode() if carrying is None else carrying.encode(),
            self.state.dir,
            topX, topY,
            self.view_size, self.view_size,
            self.see_through_walls,
        )
        grid_state = obs_grid_result[..., :-1].astype(int)
        vis_mask = obs_grid_result[..., -1].astype(bool)
        return Grid.from_grid_state(grid_state), vis_mask

    def gen_obs(self, grid_state: WorldObjState) -> dict:
        """
        Generate the agent's view (partially observable, low-resolution encoding).

        Returns
        -------
        obs : dict
            - 'image': partially observable view of the environment
            - 'direction': agent's direction/orientation (acting as a compass)
            - 'mission': textual mission string (instructions for the agent)
        """
        topX, topY, _, _ = get_view_exts(
            self.state.dir, tuple(self.state.pos), self.view_size)
        carrying = self.state.carrying
        image = gen_obs_grid_encoding(
            grid_state.encode(),
            WorldObjState.empty().encode() if carrying is None else carrying.encode(),
            self.state.dir,
            topX, topY,
            self.view_size, self.view_size,
            self.see_through_walls,
        )
        obs = {'image': image, 'direction': self.state.dir, 'mission': self.mission}
        return obs

    def encode(self):
        """
        Encode a description of this agent as a 3-tuple of integers.
        """
        return (OBJECT_TO_IDX['agent'], COLOR_TO_IDX[self.state.color], self.state.dir)

    def render(self, img: np.ndarray[int]):
        """
        Draw the agent.
        """
        c = COLORS[self.state.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * self.state.dir)
        fill_coords(img, tri_fn, c)
