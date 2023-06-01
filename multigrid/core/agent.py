from __future__ import annotations

import numpy as np

from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

from .actions import Action
from .constants import COLORS, COLOR_TO_IDX, OBJECT_TO_IDX, DIR_TO_VEC
from .mission import MissionSpace
from .world_object import WorldObj

from ..utils.misc import front_pos, PropertyAlias
from ..utils.rendering import (
    fill_coords,
    point_in_triangle,
    rotate_fn,
)



class AgentState(np.ndarray):
    """
    State for an :class:`.Agent` object.

    `AgentState` objects also support vectorized operations,
    in which case the `AgentState` object represents a "batch" of states
    over multiple agents.

    Attributes
    ----------
    color : str or ndarray[str]
        Agent color
    dir : int or ndarray[int]
        Agent direction (0: right, 1: down, 2: left, 3: up)
    pos : ndarray[int]
        Agent (x, y) position
    terminated : bool or ndarray[bool]
        Whether the agent has terminated
    carrying : WorldObj or None or ndarray
        Object the agent is carrying

    Examples
    --------
    Create a vectorized agent state for 3 agents:

    >>> agent_state = AgentState(3)
    >>> agent_state
    AgentState(3)

    Access and set state attributes for one agent at a time:

    >>> a = agent_state[0]
    >>> a
    AgentState()
    >>> a.color
    'red'
    >>> a.color = 'yellow'
    >>> b = agent_state[1]
    >>> b.pos
    array([-1, -1])
    >>> b.pos = (23, 45)

    The underlying vectorized state is automatically updated as well:

    >>> agent_state.color
    array(['yellow', 'green', 'blue'])
    >>> agent_state.pos
    array([[-1, -1],
       [23, 45],
       [-1, -1]])

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
    dim = 9
    _colors = np.array(list(COLOR_TO_IDX.keys()))
    _color_to_idx = np.vectorize(COLOR_TO_IDX.__getitem__)

    def __new__(cls, *dims: int):
        obj = np.zeros(dims + (cls.dim,), dtype=int).view(cls)
        obj._carried_obj = np.empty(dims, dtype=object) # object references
        obj[..., 0] = OBJECT_TO_IDX['agent'] # type
        obj[..., 1].flat = np.arange(np.prod(dims), dtype=int) # color
        obj[..., 1] %= len(COLOR_TO_IDX)
        obj.dir = -1
        obj.pos = (-1, -1)
        obj.terminated = False
        return obj

    def __repr__(self):
        shape = str(self.shape[:-1]).replace(",)", ")")
        return f'{self.__class__.__name__}{shape}'

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if out.shape and out.shape[-1] == self.dim:
            out._carried_obj = self._carried_obj[idx, ...] # set carried object reference
        return out

    @property
    def color(self) -> str | NDArray[np.str_]:
        """
        Return the name of the agent color.
        """
        out = super().__getitem__((..., 1))
        out = self._colors[out]
        return out.item() if out.ndim == 0 else out

    @color.setter
    def color(self, value: str | ArrayLike[str]):
        """
        Set the agent color.
        """
        self[..., 1] = self._color_to_idx(value)

    @property
    def dir(self) -> int | NDArray[np.int_]:
        """
        Return the agent direction (0: right, 1: down, 2: left, 3: up).
        """
        out = super().__getitem__((..., 2))
        out = out.view(np.ndarray)
        return out.item() if out.ndim == 0 else out

    @dir.setter
    def dir(self, value: int | ArrayLike[int]):
        """
        Set the agent direction.
        """
        self[..., 2] = value

    @property
    def pos(self) -> NDArray[np.int_]:
        """
        Return the agent's (x, y) position.
        """
        out = super().__getitem__((..., slice(3, 5)))
        return out.view(np.ndarray)

    @pos.setter
    def pos(self, value: ArrayLike[int]):
        """
        Set the agent's (x, y) position.
        """
        self[..., 3:5] = value

    @property
    def terminated(self) -> bool | NDArray[np.bool_]:
        """
        Return whether the agent has terminated.
        """
        out = super().__getitem__((..., 5))
        out = out.view(np.ndarray)
        return bool(out.item()) if out.ndim == 0 else out

    @terminated.setter
    def terminated(self, value: bool | ArrayLike[bool]):
        """
        Set whether the agent has terminated.
        """
        self[..., 5] = value

    @property
    def carrying(self) -> WorldObj | None | NDArray[np.object_]:
        """
        Return the object the agent is carrying.
        """
        out = self._carried_obj
        return out.item() if out.ndim == 0 else out

    @carrying.setter
    def carrying(self, obj: WorldObj | None | ArrayLike[WorldObj | None]):
        """
        Set the object the agent is carrying.
        """
        self[..., 6:6+WorldObj.dim] = WorldObj.empty() if obj is None else obj
        self._carried_obj[...].fill(obj)


class Agent:
    """
    Class representing an agent in the environment.

    **Observation Space**

    Observations are dictionaries with the following entries:

    * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
        Encoding of the agent's view of the environment
    * direction : int
        Agent's direction (0: right, 1: down, 2: left, 3: up)
    * mission : str
        Task string corresponding to the current environment configuration

    **Action Space**

    Actions are discrete integers, as enumerated in :class:`.Action`.

    Attributes
    ----------
    index : int
        Index of the agent in the environment
    state : AgentState
        State of the agent
    action_space : gym.spaces.Discrete
        Action space for the agent
    observation_space : gym.spaces.Dict
        Observation space for the agent
    dir_vec : ndarray[int]
        Direction vector for the agent, pointing in the direction of forward movement
    front_pos : tuple[int, int]
        Position of the cell that is directly in front of the agent
    """
    # Properties
    color = PropertyAlias(
        'state', AgentState.color, doc='Alias for :attr:`AgentState.color`.')
    dir = PropertyAlias(
        'state', AgentState.dir, doc='Alias for :attr:`AgentState.dir`.')
    pos = PropertyAlias(
        'state', AgentState.pos, doc='Alias for :attr:`AgentState.pos`.')
    terminated = PropertyAlias(
        'state', AgentState.terminated, doc='Alias for :attr:`AgentState.terminated`.')
    carrying = PropertyAlias(
        'state', AgentState.carrying, doc='Alias for :attr:`AgentState.carrying`.')

    def __init__(
        self,
        index: int,
        mission_space: MissionSpace,
        state: AgentState | None = None,
        view_size: int = 7,
        see_through_walls: bool = False):
        """
        Parameters
        ----------
        index : int
            Index of the agent in the environment
        mission_space : MissionSpace
            The mission space for the agent
        state : AgentState or None
            AgentState object to use for the agent
        view_size : int
            The size of the agent's view (must be odd)
        see_through_walls : bool
            Whether the agent can see through walls
        """
        self.index: int = index
        self.state: AgentState = AgentState() if state is None else state

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(Action))

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
            'mission': mission_space,
        })
        self.see_through_walls = see_through_walls

        # Current agent state
        self.mission: str = None

    @property
    def dir_vec(self) -> NDArray[np.int_]:
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        return DIR_TO_VEC[self.state.dir % 4]

    @property
    def front_pos(self) -> tuple[int, int]:
        """
        Get the position of the cell that is directly in front of the agent.
        """
        return front_pos(*self.state.pos, self.state.dir)

    def reset(self, mission: str = 'maximize reward'):
        """
        Reset the agent to an initial state.

        Parameters
        ----------
        mission : str
            Mission string to use for the new episode
        """
        self.mission = mission
        self.state.pos = (-1, -1)
        self.state.dir = -1
        self.state.terminated = False
        self.state.carrying = None

    def encode(self) -> tuple[int, int, int]:
        """
        Encode a description of this agent as a 3-tuple of integers.

        Returns
        -------
        type_idx : int
            The index of the agent type in `OBJECT_TO_IDX`
        color_idx : int
            The index of the agent color in `COLOR_TO_IDX`
        state_idx : int
            The direction of the agent (0: right, 1: down, 2: left, 3: up)
        """
        return (OBJECT_TO_IDX['agent'], COLOR_TO_IDX[self.state.color], self.state.dir)

    def render(self, img: NDArray[np.uint8]):
        """
        Draw the agent.

        Parameters
        ----------
        img : ndarray[int] of shape (width, height, 3)
            RGB image array to render agent on
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
