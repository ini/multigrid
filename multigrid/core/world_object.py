from __future__ import annotations

import functools
import numpy as np
import numba as nb

from numpy.typing import ArrayLike, NDArray as ndarray
from typing import TYPE_CHECKING

from .constants import Color, State, Type
from ..utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)

if TYPE_CHECKING:
    from .agent import Agent
    from ..multigrid_env import MultiGridEnv



# WorldObj indices
TYPE = 0
COLOR = 1
STATE = 2

# Object type indices
EMPTY = Type.empty.to_index()
WALL = Type.wall.to_index()
FLOOR = Type.floor.to_index()
DOOR = Type.door.to_index()
KEY = Type.key.to_index()
BALL = Type.ball.to_index()
BOX = Type.box.to_index()
GOAL = Type.goal.to_index()
LAVA = Type.lava.to_index()

# Object state indices
OPEN = State.open.to_index()
CLOSED = State.closed.to_index()
LOCKED = State.locked.to_index()



### World Object Qualities

@nb.njit(cache=True)
def can_overlap(obj: WorldObj | None) -> bool:
    """
    Can an agent overlap with this?

    Parameters
    ----------
    obj : WorldObj or None
        Object to check
    """
    if obj is None:
        return True
    if obj[TYPE] in {EMPTY, FLOOR, GOAL, LAVA}:
        return True
    elif obj[TYPE] == DOOR and obj[STATE] == OPEN:
        return True

    return False

@nb.njit(cache=True)
def can_pickup(obj: WorldObj | None) -> bool:
    """
    Can an agent pick this up?

    Parameters
    ----------
    obj : WorldObj or None
        Object to check
    """
    return obj is not None and obj[TYPE] in {KEY, BALL, BOX}

@nb.njit(cache=True)
def can_contain(obj: WorldObj | None) -> bool:
    """
    Can this contain another object?

    Parameters
    ----------
    obj : WorldObj
        Object to check
    """
    return obj is not None and obj[TYPE] == BOX

@nb.njit(cache=True)
def see_behind(obj: WorldObj | None) -> bool:
    """
    Can an agent see behind this object?

    Parameters
    ----------
    obj : WorldObj or None
        Object to check
    """
    if obj is None:
        return True
    if obj[TYPE] == WALL:
        return False
    elif obj[TYPE] == DOOR and obj[STATE] != OPEN:
        return False

    return True



### World Object Classes

class WorldObj(np.ndarray):
    """
    Base class for grid world objects.

    Attributes
    ----------
    type : Type
        The object type
    color : Color
        The object color
    state : State
        The object state
    contains : WorldObj or None
        The object contained by this object, if any
    init_pos : tuple[int, int] or None
        The initial position of the object
    cur_pos : tuple[int, int] or None
        The current position of the object
    """
    dim = len([TYPE, COLOR, STATE])

    def __new__(cls, type: str, color: str):
        """
        Parameters
        ----------
        type : str
            Object type
        color : str
            Object color
        """
        obj = np.zeros(cls.dim, dtype=int).view(cls)
        obj[TYPE] = Type(type).to_index()
        obj[COLOR] = Color(color).to_index()
        obj.contains: WorldObj | None = None # object contained by this object
        obj.init_pos = None # initial position of the object
        obj.cur_pos = None # current position of the object
        return obj

    def __bool__(self) -> bool:
        return self.type != 'empty'

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(color={self.color})"

    @staticmethod
    @functools.cache
    def empty() -> 'WorldObj':
        """
        Return an empty WorldObj instance.
        """
        return WorldObj(type='empty', color=Color.from_index(0))

    @staticmethod
    def from_array(arr: ArrayLike[int]) -> 'WorldObj' | None:
        """
        Convert an array to a WorldObj instance.

        Parameters
        ----------
        arr : ArrayLike[int]
            Array encoding the object type, color, and state
        """
        if arr[TYPE] == EMPTY:
            return None

        object_idx_to_class = {
            WALL: Wall,
            FLOOR: Floor,
            DOOR: Door,
            KEY: Key,
            BALL: Ball,
            BOX: Box,
            GOAL: Goal,
            LAVA: Lava,
        }

        if arr[TYPE] in object_idx_to_class:
            cls = object_idx_to_class[arr[TYPE]]
            obj = cls.__new__(cls)
            obj[...] = arr
            return obj

        raise ValueError(f'Unknown object type: {arr[TYPE]}')

    @property
    def type(self) -> Type:
        """
        Return the object type.
        """
        return Type.from_index(self[TYPE])

    @type.setter
    def type(self, value: str):
        """
        Set the object type.
        """
        self[TYPE] = Type(value)

    @property
    def color(self) -> Color:
        """
        Return the object color.
        """
        return Color.from_index(self[COLOR])

    @color.setter
    def color(self, value: str):
        """
        Set the object color.
        """
        self[COLOR] = Color(value)

    @property
    def state(self) -> str:
        """
        Return the name of the object state.
        """
        return State.from_index(self[STATE])

    @state.setter
    def state(self, value: str):
        """
        Set the name of the object state.
        """
        self[STATE] = State(value)

    def toggle(self, env: MultiGridEnv, agent: Agent, pos: tuple[int, int]) -> bool:
        """
        Toggle the state of this object or trigger an action this object performs.

        Parameters
        ----------
        env : MultiGridEnv
            The environment this object is contained in
        agent : Agent
            The agent performing the toggle action
        pos : tuple[int, int]
            The (x, y) position of this object in the environment grid

        Returns
        -------
        success : bool
            Whether the toggle action was successful
        """
        return False

    def encode(self) -> tuple[int, int, int]:
        """
        Encode a 3-tuple description of this object.

        Returns
        -------
        type_idx : int
            The index of the object type
        color_idx : int
            The index of the object color
        state_idx : int
            The index of the object state
        """
        return tuple(self)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state_idx: int) -> 'WorldObj' | None:
        """
        Create an object from a 3-tuple description.

        Parameters
        ----------
        type_idx : int
            The index of the object type
        color_idx : int
            The index of the object color
        state_idx : int
            The index of the object state
        """
        arr = np.array([type_idx, color_idx, state_idx])
        return WorldObj.from_array(arr)

    def render(self, img: ndarray[np.uint8]):
        """
        Draw the world object.

        Parameters
        ----------
        img : ndarray[int] of shape (width, height, 3)
            RGB image array to render object on
        """
        raise NotImplementedError


class Goal(WorldObj):
    """
    Goal object an agent may be searching for.
    """

    @functools.cache # reuse instances, since object is immutable
    def __new__(cls):
        """
        """
        return super().__new__(cls, type='goal', color='green')

    def render(self, img):
        """
        :meta private:
        """
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())


class Floor(WorldObj):
    """
    Colored floor tile an agent can walk over.
    """

    @functools.cache # reuse instances, since object is immutable
    def __new__(cls, color: str = 'blue'):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, type='floor', color=color)

    def render(self, img):
        """
        :meta private:
        """
        # Give the floor a pale color
        color = self.color.rgb() / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    """
    Lava object an agent can fall onto.
    """

    @functools.cache # reuse instances, since object is immutable
    def __new__(cls):
        """
        """
        return super().__new__(cls, type='lava', color='red')

    def render(self, img):
        """
        :meta private:
        """
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    """
    Wall object that agents cannot move through.
    """

    @functools.cache # reuse instances, since object is immutable
    def __new__(cls, color: str = 'grey'):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, type='wall', color=color)

    def render(self, img):
        """
        :meta private:
        """
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())


class Door(WorldObj):
    """
    Door object that may be opened or closed. Locked doors require a key to open.

    Attributes
    ----------
    is_open: bool
        Whether the door is open
    is_locked: bool
        Whether the door is locked
    """

    def __new__(
        cls, color: str = 'blue', is_open: bool = False, is_locked: bool = False):
        """
        Parameters
        ----------
        color : str
            Object color
        is_open : bool
            Whether the door is open
        is_locked : bool
            Whether the door is locked
        """
        door = super().__new__(cls, type='door', color=color)
        door.is_open = is_open
        door.is_locked = is_locked
        return door

    def __str__(self):
        return f"{self.__class__.__name__}(color={self.color},state={self.state})"

    @property
    def is_open(self) -> bool:
        """
        Whether the door is open.
        """
        return self.state == 'open'

    @is_open.setter
    def is_open(self, value: bool):
        """
        Set the door to be open or closed.
        """
        if value:
            self.state = 'open' # set state to open
        elif not self.is_locked:
            self.state = 'closed' # set state to closed (unless already locked)

    @property
    def is_locked(self) -> bool:
        """
        Whether the door is locked.
        """
        return self.state == 'locked'

    @is_locked.setter
    def is_locked(self, value: bool):
        """
        Set the door to be locked or unlocked.
        """
        if value:
            self.state = 'locked' # set state to locked
        elif not self.is_open:
            self.state = 'closed' # set state to closed (unless already open)

    def toggle(self, env, agent, pos):
        """
        :meta private:
        """
        if self.is_locked:
            # Check if the player has the right key to unlock the door
            carried_obj = agent.state.carrying
            if isinstance(carried_obj, Key) and carried_obj.color == self.color:
                self.is_locked = False
                self.is_open = True
                env.grid.update(*pos)
                return True
            return False

        self.is_open = not self.is_open
        env.grid.update(*pos)
        return True

    def render(self, img):
        """
        :meta private:
        """
        c = self.color.rgb()

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(
                img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    """
    Key object that can be picked up and used to unlock doors.
    """

    def __new__(cls, color: str = 'blue'):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, type='key', color=color)

    def render(self, img):
        """
        :meta private:
        """
        c = self.color.rgb()

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    """
    Ball object that can be picked up by agents.
    """

    def __new__(cls, color: str = 'blue'):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, type='ball', color=color)

    def render(self, img):
        """
        :meta private:
        """
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.color.rgb())


class Box(WorldObj):
    """
    Box object that may contain other objects.
    """

    def __new__(cls, color: str = 'yellow', contains: WorldObj | None = None):
        """
        Parameters
        ----------
        color : str
            Object color
        contains : WorldObj or None
            Object contents
        """
        box = super().__new__(cls, type='box', color=color)
        box.contains = contains
        return box

    def toggle(self, env, agent, pos):
        """
        :meta private:
        """
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

    def render(self, img):
        """
        :meta private:
        """
        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), self.color.rgb())
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), self.color.rgb())
