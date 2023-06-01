from __future__ import annotations

import numpy as np
import numba as nb

from .constants import (
    OBJECT_TO_IDX, IDX_TO_OBJECT,
    COLOR_TO_IDX, IDX_TO_COLOR, COLORS,
    STATE_TO_IDX, IDX_TO_STATE,
)

from ..utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..multigrid_env import MultiGridEnv
    from .agent import Agent



# WorldObj indices
TYPE = 0
COLOR = 1
STATE = 2

# Object type indices
EMPTY = OBJECT_TO_IDX['empty']
WALL = OBJECT_TO_IDX['wall']
FLOOR = OBJECT_TO_IDX['floor']
DOOR = OBJECT_TO_IDX['door']
KEY = OBJECT_TO_IDX['key']
BALL = OBJECT_TO_IDX['ball']
BOX = OBJECT_TO_IDX['box']
GOAL = OBJECT_TO_IDX['goal']
LAVA = OBJECT_TO_IDX['lava']

# Object state indices
OPEN = STATE_TO_IDX['open']
CLOSED = STATE_TO_IDX['closed']
LOCKED = STATE_TO_IDX['locked']



### World Object Qualities

@nb.njit(cache=True)
def can_overlap(obj: WorldObj) -> bool:
    """
    Can an agent overlap with this?
    """
    if obj[TYPE] in {EMPTY, FLOOR, GOAL, LAVA}:
        return True
    elif obj[TYPE] == DOOR and obj[STATE] == OPEN:
        return True

    return False

@nb.njit(cache=True)
def can_pickup(obj: WorldObj) -> bool:
    """
    Can an agent pick this up?
    """
    return obj in {KEY, BALL, BOX}

@nb.njit(cache=True)
def can_contain(obj: WorldObj) -> bool:
    """
    Can this contain another object?
    """
    return obj[TYPE] == BOX

@nb.njit(cache=True)
def see_behind(obj: WorldObj) -> bool:
    """
    Can an agent see behind this object?
    """
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
    type : str
        The name of the object type
    color : str
        The name of the object color
    state : str
        The name of the object state
    contains : WorldObj or None
        The object contained by this object, if any
    init_pos : tuple[int, int] or None
        The initial position of the object
    cur_pos : tuple[int, int] or None
        The current position of the object
    """
    dim = 3 # (type, color, state)
    _empty = None

    def __new__(cls, type: str, color: str):
        """
        Parameters
        ----------
        type : str
            The name of the object type
        color : str
            The name of the object color
        """
        obj = np.zeros(cls.dim, dtype=int).view(cls)
        obj[TYPE] = OBJECT_TO_IDX[type]
        obj[COLOR] = COLOR_TO_IDX[color]
        obj.contains: WorldObj | None = None # object contained by this object
        obj.init_pos = None # initial position of the object
        obj.cur_pos = None # current position of the object
        return obj

    def __bool__(self) -> bool:
        return self.type != 'empty'

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(color={self.color})"

    @staticmethod
    def empty() -> 'WorldObj':
        """
        Return fixed reference to an empty WorldObj instance.
        """
        if WorldObj._empty is None:
            WorldObj._empty = WorldObj(type='empty', color='red')

        return WorldObj._empty

    @staticmethod
    def from_array(arr: np.ndarray) -> 'WorldObj' | None:
        """
        Convert an array to a WorldObj instance.
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
    def type(self) -> str:
        """
        Return the name of the object type.
        """
        return IDX_TO_OBJECT[self[TYPE]]

    @type.setter
    def type(self, value: str):
        """
        Set the name of the object type.
        """
        self[TYPE] = OBJECT_TO_IDX[value]

    @property
    def color(self) -> str:
        """
        Return the name of the object color.
        """
        return IDX_TO_COLOR[self[COLOR]]

    @color.setter
    def color(self, value: str):
        """
        Set the name of the object color.
        """
        self[COLOR] = COLOR_TO_IDX[value]

    @property
    def state(self) -> str:
        """
        Return the name of the object state.
        """
        return IDX_TO_STATE[self[STATE]]

    @state.setter
    def state(self, value: str):
        """
        Set the name of the object state.
        """
        self[STATE] = STATE_TO_IDX[value]

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
            The index of the object type in `OBJECT_TO_IDX`
        color_idx : int
            The index of the object color in `COLOR_TO_IDX`
        state_idx : int
            The index of the object state in `STATE_TO_IDX`
        """
        return tuple(self)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state_idx: int) -> 'WorldObj' | None:
        """
        Create an object from a 3-tuple description.

        Parameters
        ----------
        type_idx : int
            The index of the object type in `OBJECT_TO_IDX`
        color_idx : int
            The index of the object color in `COLOR_TO_IDX`
        state_idx : int
            The index of the object state in `STATE_TO_IDX`
        """
        arr = np.array([type_idx, color_idx, state_idx])
        return WorldObj.from_array(arr)

    def render(self, img: np.ndarray[int]):
        """
        Draw this object with the given renderer.

        Parameters
        ----------
        img : np.ndarray[int] of shape (width, height, 3)
            RGB image array to render object on
        """
        raise NotImplementedError


class Goal(WorldObj):
    """
    Goal object an agent may be searching for.
    """

    def __new__(cls):
        return super().__new__(cls, type='goal', color='green')

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile an agent can walk over.
    """

    def __new__(cls, color: str = 'blue'):
        return super().__new__(cls, type='floor', color=color)

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    """
    Lava object an agent can fall onto.
    """

    def __new__(cls):
        return super().__new__(cls, type='lava', color='red')

    def render(self, img):
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

    def __new__(cls, color: str = 'grey'):
        return super().__new__(cls, type='wall', color=color)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


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
        c = COLORS[self.color]

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
    Key object that can be picked up and used to open locked doors.
    """

    def __new__(cls, color: str = 'blue'):
        return super().__new__(cls, type='key', color=color)

    def render(self, img):
        c = COLORS[self.color]

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
        return super().__new__(cls, type='ball', color=color)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    """
    Box object that may contain other objects.
    """

    def __new__(cls, color: str = 'yellow', contains: WorldObj | None = None):
        box = super().__new__(cls, type='box', color=color)
        box.contains = contains
        return box

    def toggle(self, env, agent, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)
