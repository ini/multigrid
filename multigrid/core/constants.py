from __future__ import annotations

import enum
import functools
import numpy as np

from numpy.typing import ArrayLike, NDArray as ndarray



#: Tile size for rendering grid cell
TILE_PIXELS = 32

COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
}

DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]



### Helper Functions

@functools.cache
def _enum_array(enum_cls: enum.EnumMeta):
    """
    Return an array of all values of the given enum.
    """
    return np.array([item.value for item in enum_cls])

@functools.cache
def _enum_index(enum_item: enum.Enum):
    """
    Return the index of the given enum item.
    """
    return list(enum_item.__class__).index(enum_item)



### Enumerations

class StrEnum(str, enum.Enum):
    """
    Enum where each member is a string with a corresponding integer index.

    :meta private:
    """

    def __int__(self):
        return self.to_index()

    @classmethod
    def from_index(cls, index: int | ArrayLike[int]) -> enum.Enum | ndarray[np.str]:
        """
        Return the enum item corresponding to the given index.
        Also supports vector inputs.

        Parameters
        ----------
        index : int or ArrayLike[int]
            Enum index (or array of indices)

        Returns
        -------
        enum.Enum or ndarray[str]
            Enum item (or array of enum item values)
        """
        out = _enum_array(cls)[index]
        return cls(out) if out.ndim == 0 else out

    def to_index(self) -> int:
        """
        Return the integer index of this enum item.
        """
        return _enum_index(self)


class Type(StrEnum):
    """
    Enumeration of object types.
    """
    unseen = 'unseen'
    empty = 'empty'
    wall = 'wall'
    floor = 'floor'
    door = 'door'
    key = 'key'
    ball = 'ball'
    box = 'box'
    goal = 'goal'
    lava = 'lava'
    agent = 'agent'


class Color(StrEnum):
    """
    Enumeration of object colors.
    """
    red = 'red'
    green = 'green'
    blue = 'blue'
    purple = 'purple'
    yellow = 'yellow'
    grey = 'grey'

    def rgb(self) -> ndarray[np.uint8]:
        """
        Return the RGB value of this ``Color``.
        """
        return COLORS[self]


class State(StrEnum):
    """
    Enumeration of object states.
    """
    open = 'open'
    closed = 'closed'
    locked = 'locked'


class Direction(enum.IntEnum):
    """
    Enumeration of agent directions.
    """
    right = 0
    down = 1
    left = 2
    up = 3

    def to_vec(self) -> ndarray[np.int]:
        """
        Return the vector corresponding to this ``Direction``.
        """
        return DIR_TO_VEC[self]



### Minigrid Compatibility

OBJECT_TO_IDX = {t: t.to_index() for t in Type}
IDX_TO_OBJECT = {t.to_index(): t for t in Type}
COLOR_TO_IDX = {c: c.to_index() for c in Color}
IDX_TO_COLOR = {c.to_index(): c for c in Color}
STATE_TO_IDX = {s: s.to_index() for s in State}
COLOR_NAMES = sorted(list(Color))
