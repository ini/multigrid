from __future__ import annotations

import numpy as np

from .constants import (
    OBJECT_TO_IDX, IDX_TO_OBJECT,
    COLOR_TO_IDX, IDX_TO_COLOR, COLORS,
    STATE_TO_IDX, IDX_TO_STATE,
)

from ..utils.misc import can_overlap, can_pickup
from ..utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)



class WorldObj(np.ndarray):
    """
    Base class for grid world objects.

    The state is a 4-dimensional integer array, indexed as follows:
    * 0: object type
    * 1: object color
    * 2: object state (i.e. open/closed/locked)
    * 3: mixed-radix integer encoding of contained object

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
    """
    dim = 4
    encode_dim = 3
    max_contain_depth = 2
    _bases = [max(IDX_TO_OBJECT) + 1, max(IDX_TO_COLOR) + 1, len(IDX_TO_STATE) + 1]
    _bases = [*_bases, np.prod(_bases) ** max_contain_depth]
    _empty = None

    def __new__(cls, type: str = 'empty', color: str | None = None):
        """
        Parameters
        ----------
        type : str, default='empty'
            The name of the object type
        color : str, optional
            The name of the object color
        """
        obj = np.zeros(cls.dim, dtype=int).view(cls)
        obj[0] = OBJECT_TO_IDX[type]
        if color is not None:
            obj[1] = COLOR_TO_IDX[color]

        return obj

    def __eq__(self, other: 'WorldObj') -> bool:
        return np.array_equal(self, other)

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
            WorldObj._empty = WorldObj(type='empty')

        return WorldObj._empty

    @staticmethod
    def from_array(arr: np.ndarray) -> 'WorldObj':
        """
        Convert an array to a WorldObj instance.
        """
        object_idx_to_class = {
            OBJECT_TO_IDX['empty']: type(None),
            OBJECT_TO_IDX['wall']: Wall,
            OBJECT_TO_IDX['floor']: Floor,
            OBJECT_TO_IDX['door']: Door,
            OBJECT_TO_IDX['key']: Key,
            OBJECT_TO_IDX['ball']: Ball,
            OBJECT_TO_IDX['box']: Box,
            OBJECT_TO_IDX['goal']: Goal,
            OBJECT_TO_IDX['lava']: Lava,
        }

        if arr[0] in object_idx_to_class:
            cls = object_idx_to_class[arr[0]]
            obj = cls.__new__(cls)
            if obj is not None:
                obj[...] = arr
            return obj

        raise ValueError(f'Unknown object type: {arr[0]}')

    @classmethod
    def from_int(cls, n: int) -> 'WorldObj' | None:
        """
        Convert a mixed-radix integer encoding to a WorldObj instance.
        """
        if n == 0 or n == 1:
            return None

        arr = np.empty(cls.dim, dtype=int)
        for i in range(cls.dim):
            arr[..., i] = n % cls._bases[i]
            n //= cls._bases[i]

        return cls.from_array(arr)

    @property
    def type(self) -> str:
        """
        Return the name of the object type.
        """
        return IDX_TO_OBJECT[self[0]]

    @type.setter
    def type(self, value: str):
        """
        Set the name of the object type.
        """
        self[0] = OBJECT_TO_IDX[value]

    @property
    def color(self) -> str:
        """
        Return the name of the object color.
        """
        return IDX_TO_COLOR[self[1]]

    @color.setter
    def color(self, value: str):
        """
        Set the name of the object color.
        """
        self[1] = COLOR_TO_IDX[value]

    @property
    def state(self) -> str:
        """
        Return the name of the object state.
        """
        return IDX_TO_STATE[self[2]]

    @state.setter
    def state(self, value: str):
        """
        Set the name of the object state.
        """
        self[2] = STATE_TO_IDX[value]

    @property
    def contains(self) -> 'WorldObj' | None:
        """
        Return the object contained by this object.
        """
        return self.from_int(self[3])

    @contains.setter
    def contains(self, world_obj: 'WorldObj' | None):
        """
        Set the object state contained by this object.
        """
        self[3] = 0 if world_obj is None else world_obj.to_int()

    def can_overlap(self) -> bool:
        """
        Can an agent overlap with this?
        """
        return can_overlap(*self)

    def can_pickup(self) -> bool:
        """
        Can an agent pick this up?
        """
        return can_pickup(*self)

    def can_contain(self) -> bool:
        """
        Can this contain another object?
        """
        return self.type == 'box'

    def see_behind(self) -> bool:
        """
        Can an agent see behind this object?
        """
        if self.type == 'wall':
            return False
        elif self.type == 'door' and self.state != 'open':
            return False
        return True

    def encode(self) -> tuple[int, int, int]:
        """
        Encode a description of this object state.
        """
        return tuple(self[:self.encode_dim])

    @staticmethod
    def decode(type_idx: int, color_idx: int, state_idx: int) -> 'WorldObj' | None:
        """
        Create an object from a 3-tuple state description.
        """
        arr = np.array([type_idx, color_idx, state_idx, 0])
        return WorldObj.from_array(arr)

    def render(self, img: np.ndarray[int]):
        """
        Draw this object with the given renderer.
        """
        raise NotImplementedError

    def to_int(self) -> int:
        """
        Encode this object state as a mixed-radix integer.
        """
        if self.type == 'empty':
            return 0

        base, n = 1, 0
        for i in range(self.dim):
            n += (self[i] * base)
            base *= self._bases[i]

        return n


class Goal(WorldObj):
    """
    Goal object an agent may be searching for.
    """

    def __new__(cls):
        return super().__new__(cls, type='goal', color='green')

    def render(self, img: np.ndarray[int]):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile an agent can walk over.
    """

    def __new__(cls, color: str = 'blue'):
        return super().__new__(cls, type='floor', color=color)

    def render(self, img: np.ndarray[int]):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    """
    Lava object an agent can fall onto.
    """

    def __new__(cls):
        return super().__new__(cls, type='lava', color='red')

    def render(self, img: np.ndarray[int]):
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

    def render(self, img: np.ndarray[int]):
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

    def render(self, img: np.ndarray[int]):
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

    def render(self, img: np.ndarray[int]):
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

    def render(self, img: np.ndarray[int]):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    """
    Box object that may contain other objects.
    """

    def __new__(cls, color: str = 'yellow', contains: WorldObj | None = None):
        box = super().__new__(cls, type='box', color=color)
        box.contains = contains
        return box

    def render(self, img: np.ndarray[int]):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)
