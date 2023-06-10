from __future__ import annotations

import functools
import numpy as np

from numpy.typing import ArrayLike, NDArray as ndarray
from typing import Any, TYPE_CHECKING

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
    # WorldObj vector indices
    TYPE = 0
    COLOR = 1
    STATE = 2

    # WorldObj vector dimension
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
        obj[WorldObj.TYPE] = Type(type).to_index()
        obj[WorldObj.COLOR] = Color(color).to_index()
        obj.contains: WorldObj | None = None # object contained by this object
        obj.init_pos: tuple[int, int] | None = None # initial position of the object
        obj.cur_pos: tuple[int, int] | None = None # current position of the object
        return obj

    def __bool__(self) -> bool:
        return self.type != 'empty'

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(color={self.color})"

    def __eq__(self, other: Any):
        return self is other

    @staticmethod
    @functools.cache
    def empty() -> 'WorldObj':
        """
        Return an empty WorldObj instance.
        """
        return WorldObj(type=Type.empty, color=Color.from_index(0))

    @staticmethod
    def from_array(arr: ArrayLike[int]) -> 'WorldObj' | None:
        """
        Convert an array to a WorldObj instance.

        Parameters
        ----------
        arr : ArrayLike[int]
            Array encoding the object type, color, and state
        """
        if arr[WorldObj.TYPE] == Type.empty.to_index():
            return None

        object_idx_to_class = {
            Type.wall.to_index(): Wall,
            Type.floor.to_index(): Floor,
            Type.door.to_index(): Door,
            Type.key.to_index(): Key,
            Type.ball.to_index(): Ball,
            Type.box.to_index(): Box,
            Type.goal.to_index(): Goal,
            Type.lava.to_index(): Lava,
        }

        if arr[WorldObj.TYPE] in object_idx_to_class:
            cls = object_idx_to_class[arr[WorldObj.TYPE]]
            obj = cls.__new__(cls)
            obj[...] = arr
            return obj

        raise ValueError(f'Unknown object type: {arr[WorldObj.TYPE]}')

    @property
    def type(self) -> Type:
        """
        Return the object type.
        """
        return Type.from_index(self[WorldObj.TYPE])

    @type.setter
    def type(self, value: str):
        """
        Set the object type.
        """
        self[WorldObj.TYPE] = Type(value).to_index()

    @property
    def color(self) -> Color:
        """
        Return the object color.
        """
        return Color.from_index(self[WorldObj.COLOR])

    @color.setter
    def color(self, value: str):
        """
        Set the object color.
        """
        self[WorldObj.OLOR] = Color(value).to_index()

    @property
    def state(self) -> str:
        """
        Return the name of the object state.
        """
        return State.from_index(self[WorldObj.STATE])

    @state.setter
    def state(self, value: str):
        """
        Set the name of the object state.
        """
        self[WorldObj.STATE] = State(value).to_index()

    def can_overlap(self) -> bool:
        """
        Can an agent overlap with this?
        """
        return self.type == Type.empty

    def can_pickup(self) -> bool:
        """
        Can an agent pick this up?
        """
        return False

    def can_contain(self) -> bool:
        """
        Can this contain another object?
        """
        return False

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

    @functools.cache # reuse instances, since object is effectively immutable
    def __new__(cls):
        """
        """
        return super().__new__(cls, type=Type.goal, color=Color.green)

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        return True

    def render(self, img):
        """
        :meta private:
        """
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())


class Floor(WorldObj):
    """
    Colored floor tile an agent can walk over.
    """

    @functools.cache # reuse instances, since object is effectively immutable
    def __new__(cls, color: str = Color.blue):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, type=Type.floor, color=color)

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        return True

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

    @functools.cache # reuse instances, since object is effectively immutable
    def __new__(cls):
        """
        """
        return super().__new__(cls, type=Type.lava, color=Color.red)

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        return True

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

    @functools.cache # reuse instances, since object is effectively immutable
    def __new__(cls, color: str = Color.grey):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, type=Type.wall, color=color)

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
        cls, color: str = Color.blue, is_open: bool = False, is_locked: bool = False):
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
        door = super().__new__(cls, type=Type.door, color=color)
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

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        return self.is_open

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

    def __new__(cls, color: str = Color.blue):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, type=Type.key, color=color)

    def can_pickup(self) -> bool:
        """
        :meta private:
        """
        return True

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

    def __new__(cls, color: str = Color.blue):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, type=Type.ball, color=color)

    def can_pickup(self) -> bool:
        """
        :meta private:
        """
        return True

    def render(self, img):
        """
        :meta private:
        """
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.color.rgb())


class Box(WorldObj):
    """
    Box object that may contain other objects.
    """

    def __new__(cls, color: str = Color.yellow, contains: WorldObj | None = None):
        """
        Parameters
        ----------
        color : str
            Object color
        contains : WorldObj or None
            Object contents
        """
        box = super().__new__(cls, type=Type.box, color=color)
        box.contains = contains
        return box

    def can_pickup(self) -> bool:
        """
        :meta private:
        """
        return True

    def can_contain(self) -> bool:
        """
        :meta private:
        """
        return True

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
