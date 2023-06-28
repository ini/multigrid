from __future__ import annotations

import numpy as np

from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray as ndarray

from .actions import Action
from .constants import Color, Direction, Type
from .mission import Mission, MissionSpace
from .world_object import WorldObj

from ..utils.misc import front_pos, PropertyAlias
from ..utils.rendering import (
    fill_coords,
    point_in_triangle,
    rotate_fn,
)



class Agent:
    """
    Class representing an agent in the environment.

    :Observation Space:

        Observations are dictionaries with the following entries:

            * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
                Encoding of the agent's view of the environment
            * direction : int
                Agent's direction (0: right, 1: down, 2: left, 3: up)
            * mission : Mission
                Task string corresponding to the current environment configuration

    :Action Space:

        Actions are discrete integers, as enumerated in :class:`.Action`.

    Attributes
    ----------
    index : int
        Index of the agent in the environment
    state : AgentState
        State of the agent
    mission : Mission
        Current mission string for the agent
    action_space : gym.spaces.Discrete
        Action space for the agent
    observation_space : gym.spaces.Dict
        Observation space for the agent
    """

    def __init__(
        self,
        index: int,
        mission_space: MissionSpace = MissionSpace.from_string('maximize reward'),
        view_size: int = 7,
        see_through_walls: bool = False):
        """
        Parameters
        ----------
        index : int
            Index of the agent in the environment
        mission_space : MissionSpace
            The mission space for the agent
        view_size : int
            The size of the agent's view (must be odd)
        see_through_walls : bool
            Whether the agent can see through walls
        """
        self.index: int = index
        self.state: AgentState = AgentState()
        self.mission: Mission = None

        # Number of cells (width and height) in the agent view
        assert view_size % 2 == 1
        assert view_size >= 3
        self.view_size = view_size
        self.see_through_walls = see_through_walls

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(view_size, view_size, WorldObj.dim),
                dtype=int,
            ),
            'direction': spaces.Discrete(len(Direction)),
            'mission': mission_space,
        })

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(Action))

    # AgentState Properties
    color = PropertyAlias(
        'state', 'color', doc='Alias for :attr:`AgentState.color`.')
    dir = PropertyAlias(
        'state', 'dir', doc='Alias for :attr:`AgentState.dir`.')
    pos = PropertyAlias(
        'state', 'pos', doc='Alias for :attr:`AgentState.pos`.')
    terminated = PropertyAlias(
        'state', 'terminated', doc='Alias for :attr:`AgentState.terminated`.')
    carrying = PropertyAlias(
        'state', 'carrying', doc='Alias for :attr:`AgentState.carrying`.')

    @property
    def front_pos(self) -> tuple[int, int]:
        """
        Get the position of the cell that is directly in front of the agent.
        """
        agent_dir = self.state._view[AgentState.DIR]
        agent_pos = self.state._view[AgentState.POS]
        return front_pos(*agent_pos, agent_dir)

    def reset(self, mission: Mission = Mission('maximize reward')):
        """
        Reset the agent to an initial state.

        Parameters
        ----------
        mission : Mission
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
            The index of the agent type
        color_idx : int
            The index of the agent color
        agent_dir : int
            The direction of the agent (0: right, 1: down, 2: left, 3: up)
        """
        return (Type.agent.to_index(), self.state.color.to_index(), self.state.dir)

    def render(self, img: ndarray[np.uint8]):
        """
        Draw the agent.

        Parameters
        ----------
        img : ndarray[int] of shape (width, height, 3)
            RGB image array to render agent on
        """
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * self.state.dir)
        fill_coords(img, tri_fn, self.state.color.rgb())


class AgentState(np.ndarray):
    """
    State for an :class:`.Agent` object.

    ``AgentState`` objects also support vectorized operations,
    in which case the ``AgentState`` object represents the states of multiple agents.

    Attributes
    ----------
    color : Color or ndarray[str]
        Agent color
    dir : Direction or ndarray[int]
        Agent direction (0: right, 1: down, 2: left, 3: up)
    pos : tuple[int, int] or ndarray[int]
        Agent (x, y) position
    terminated : bool or ndarray[bool]
        Whether the agent has terminated
    carrying : WorldObj or None or ndarray[object]
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

    The underlying vectorized state is automatically updated as well:

    >>> agent_state.color
    array(['yellow', 'green', 'blue'])

    Access and set state attributes all at once:

    >>> agent_state.dir
    array([-1, -1, -1])
    >>> agent_state.dir = np.random.randint(4, size=(len(agent_state)))
    >>> agent_state.dir
    array([2, 3, 0])
    >>> a.dir
    2
    """
    # State vector indices
    TYPE = 0
    COLOR = 1
    DIR = 2
    ENCODING = slice(0, 3)
    POS = slice(3, 5)
    TERMINATED = 5
    CARRYING = slice(6, 6 + WorldObj.dim)

    # State vector dimension
    dim = 6 + WorldObj.dim

    def __new__(cls, *dims: int):
        """
        Parameters
        ----------
        dims : int, optional
            Shape of vectorized agent state
        """
        obj = np.zeros(dims + (cls.dim,), dtype=int).view(cls)

        # Set default values
        obj[..., AgentState.TYPE] = Type.agent
        obj[..., AgentState.COLOR].flat = Color.cycle(np.prod(dims))
        obj[..., AgentState.DIR] = -1
        obj[..., AgentState.POS] = (-1, -1)

        # Other attributes
        obj._carried_obj = np.empty(dims, dtype=object) # object references
        obj._terminated = np.zeros(dims, dtype=bool) # cache for faster access
        obj._view = obj.view(np.ndarray) # view of the underlying array (faster indexing)

        return obj

    def __repr__(self):
        shape = str(self.shape[:-1]).replace(",)", ")")
        return f'{self.__class__.__name__}{shape}'

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if out.shape and out.shape[-1] == self.dim:
            out._view = self._view[idx, ...]
            out._carried_obj = self._carried_obj[idx, ...] # set carried object reference
            out._terminated = self._terminated[idx, ...] # set terminated cache

        return out

    @property
    def color(self) -> Color | ndarray[np.str]:
        """
        Return the agent color.
        """
        return Color.from_index(self._view[..., AgentState.COLOR])

    @color.setter
    def color(self, value: str | ArrayLike[str]):
        """
        Set the agent color.
        """
        self[..., AgentState.COLOR] = np.vectorize(lambda c: Color(c).to_index())(value)

    @property
    def dir(self) -> Direction | ndarray[np.int]:
        """
        Return the agent direction.
        """
        out = self._view[..., AgentState.DIR]
        return Direction(out.item()) if out.ndim == 0 else out

    @dir.setter
    def dir(self, value: int | ArrayLike[int]):
        """
        Set the agent direction.
        """
        self[..., AgentState.DIR] = value

    @property
    def pos(self) -> tuple[int, int] | ndarray[np.int]:
        """
        Return the agent's (x, y) position.
        """
        out = self._view[..., AgentState.POS]
        return tuple(out) if out.ndim == 1 else out

    @pos.setter
    def pos(self, value: ArrayLike[int] | ArrayLike[ArrayLike[int]]):
        """
        Set the agent's (x, y) position.
        """
        self[..., AgentState.POS] = value

    @property
    def terminated(self) -> bool | ndarray[np.bool]:
        """
        Return whether the agent has terminated.
        """
        out = self._terminated
        return out.item() if out.ndim == 0 else out

    @terminated.setter
    def terminated(self, value: bool | ArrayLike[bool]):
        """
        Set whether the agent has terminated.
        """
        self[..., AgentState.TERMINATED] = value
        self._terminated[...] = value

    @property
    def carrying(self) -> WorldObj | None | ndarray[np.object]:
        """
        Return the object the agent is carrying.
        """
        out = self._carried_obj
        return out.item() if out.ndim == 0 else out

    @carrying.setter
    def carrying(self, obj: WorldObj | None | ArrayLike[object]):
        """
        Set the object the agent is carrying.
        """
        self[..., AgentState.CARRYING] = WorldObj.empty() if obj is None else obj
        if isinstance(obj, (WorldObj, type(None))):
            self._carried_obj[...].fill(obj)
        else:
            self._carried_obj[...] = obj
