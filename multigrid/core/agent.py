import numpy as np

from gymnasium import spaces
from typing import Optional, Sequence, TYPE_CHECKING

from .actions import Actions
from .constants import COLORS, COLOR_TO_IDX, OBJECT_TO_IDX, DIR_TO_VEC
from .mission import MissionSpace
from .world_object import WorldObj, WorldObjState

from ..utils.misc import front_pos
from ..utils.rendering import (
    fill_coords,
    point_in_triangle,
    rotate_fn,
)

if TYPE_CHECKING:
    from .grid import Grid



class AgentState(np.ndarray):
    """
    State for an `Agent` object.

    The state is a 10-dimensional array, indexed as follows:
        * 0: object type (i.e. 'agent')
        * 1: agent color
        * 2: agent direction
        * 3: x position
        * 4: y position
        * 5: whether the agent has terminated
        * 6-9: WorldObjState of carried object (if any)

    AgentState objects also support vectorized operations,
    in which case the first (n - 1) dimensions represent a "batch" of states,
    and the last dimension represents each index in a state array.

    Attributes
    ----------
    color : str
        Agent color
    dir : int
        Agent direction (integer from 0 to 3)
    pos : np.ndarray[int]
        Agent (x, y) position
    terminated : bool
        Whether the agent has terminated
    carrying : WorldObjState
        WorldObjState of object the agent is carrying

    Examples
    --------
    Create a vectorized agent state for 3 agents:

    >>> agent_state = AgentState(3)
    >>> a = agent_state[0]
    >>> b = agent_state[1]
    >>> c = agent_state[2]
    >>> agent_state
    AgentState([[10,  0, -1, -1, -1,  0,  0,  0,  0,  0],
                [10,  1, -1, -1, -1,  0,  0,  0,  0,  0],
                [10,  2, -1, -1, -1,  0,  0,  0,  0,  0]])

    Access and set state attributes one at a time:

    >>> a.color
    'red'
    >>> a.color = 'yellow'
    >>> a
    AgentState([10,  4, -1, -1, -1,  0,  0,  0,  0,  0])
    >>> agent_state[1].pos = (23, 45)
    >>> b
    AgentState([10,  1, -1, 23, 45,  0,  0,  0,  0,  0])

    The underlying vectorized state is updated as well:

    >>> agent_state
    AgentState([[10,  4, -1, -1, -1,  0,  0,  0,  0,  0],
                [10,  1, -1, 23, 45,  0,  0,  0,  0,  0],
                [10,  2, -1, -1, -1,  0,  0,  0,  0,  0]])

    Access and set state attributes all at once:

    >>> agent_state.dir
    array([-1, -1, -1])
    >>> agent_state.dir = np.random.randint(4, size=(len(agent_state)))
    >>> agent_state.dir
    array([2, 3, 0])
    >>> a.dir
    2
    >>> b.dir
    3
    >>> c.dir
    0
    """
    dim = 6 + WorldObjState.dim
    _colors = np.array(list(COLOR_TO_IDX.keys()))
    _color_to_idx = np.vectorize(COLOR_TO_IDX.__getitem__)
    _dir_to_vec = np.array(DIR_TO_VEC)

    def __new__(cls, *dims: int):
        obj = super().__new__(cls, shape=dims+(cls.dim,), dtype=int)
        obj[..., 0] = OBJECT_TO_IDX['agent'] # type
        obj[..., 1] = np.arange(np.prod(dims), dtype=int).reshape(*dims) # color
        obj[..., 1] %= len(COLOR_TO_IDX)
        obj[..., 2] = -1 # dir
        obj[..., 3:5] = -1 # pos
        obj[..., 5] = False # terminated
        obj[..., 6:] = 0 # carrying
        return obj

    @property
    def color(self) -> str:
        """
        Return the name of the agent color.
        """
        out = self._colors[self[..., 1]]
        return out.item() if out.ndim == 0 else out

    @color.setter
    def color(self, value: str):
        """
        Set the agent color.
        """
        self[..., 1] = self._color_to_idx(value)

    @property
    def dir(self) -> int:
        """
        Return the agent direction.
        """
        out = self[..., 2].view(np.ndarray)
        return out.item() if out.ndim == 0 else out

    @dir.setter
    def dir(self, value: int):
        """
        Set the agent direction.
        """
        self[..., 2] = value

    @property
    def pos(self) -> np.ndarray[int]:
        """
        Return the agent's (x, y) position.
        """
        return self[..., 3:5].view(np.ndarray)

    @pos.setter
    def pos(self, value: np.ndarray[int]):
        """
        Set the agent's (x, y) position.
        """
        self[..., 3:5] = value

    @property
    def terminated(self) -> bool:
        """
        Return whether the agent has terminated.
        """
        out = self[..., 5].view(np.ndarray)
        return bool(out.item()) if out.ndim == 0 else out

    @terminated.setter
    def terminated(self, value: bool):
        """
        Set whether the agent has terminated.
        """
        self[..., 5] = value

    @property
    def carrying(self) -> WorldObjState:
        """
        Return the WorldObjState of the object the agent is carrying.
        """
        arr = self[..., 6:6+WorldObjState.dim]
        return WorldObjState.from_array(arr)

    @carrying.setter
    def carrying(self, obj_state: WorldObjState):
        """
        Set the WorldObjState of the object the agent is carrying.
        """
        self[..., 6:6+WorldObjState.dim] = obj_state

    def world_obj_state_encoding(self) -> np.ndarray[int]:
        """
        Encode a description of this agent as a 3-tuple of integers
        (i.e. type, color, direction).
        """
        return self[..., :WorldObjState.encode_dim].view(np.ndarray)


class Agent:
    """
    Class representing an agent in the environment.

    Attributes
    ----------
    id : int
        Index of the agent in the environment
    state : AgentState
        State of the agent
    action_space : gym.spaces.Discrete
        Action space for the agent
    observation_space : gym.spaces.Dict
        Observation space for the agent
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
        self.id: int = id
        self.state: AgentState = state

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
        Reset the agent to an initial state.
        """
        self.mission = mission
        self.state.pos = (-1, -1)
        self.state.dir = -1
        self.state.terminated = False
        self.state.carrying = WorldObjState.empty()

    @property
    def pos(self) -> np.ndarray[int]:
        """
        Return the agent's (x, y) position.
        """
        return self.state.pos

    @pos.setter
    def pos(self, value: Sequence[int]):
        """
        Set the agent's (x, y) position.
        """
        self.state.pos = value

    @property
    def dir(self) -> int:
        """
        Return the agent direction.
        """
        return self.state.dir

    @dir.setter
    def dir(self, value: int):
        """
        Set the agent direction.
        """
        self.state.dir = value

    @property
    def carrying(self) -> Optional[WorldObj]:
        """
        Return the object the agent is carrying.
        """
        return WorldObj.from_state(self.state.carrying)

    @carrying.setter
    def carrying(self, obj: Optional[WorldObj]):
        """
        Set the object the agent is carrying.
        """
        if obj is None:
            self.state.carrying = WorldObjState.empty()
        else:
            self.state.carrying = obj.state

    @property
    def dir_vec(self) -> np.ndarray[int]:
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        return DIR_TO_VEC[self.state.dir % 4]

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
        return front_pos(*self.state.pos, self.state.dir)

    def world_state(self) -> WorldObjState:
        arr = np.array([*self.state.world_obj_state_encoding(), 0])
        return WorldObjState.from_array(arr)

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

    def sees(self, x: int, y: int, grid: 'Grid') -> bool:
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

    def encode(self) -> tuple[int, int, int]:
        """
        Encode a description of this agent as a 3-tuple of integers.
        """
        return tuple(self.state.world_obj_state_encoding())

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
