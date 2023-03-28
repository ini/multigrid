import numpy as np

from typing import Optional, TYPE_CHECKING

from .array import (
    empty,
    contents,
    can_overlap,
    can_pickup,
    can_contain,
    see_behind,
)
from .constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    STATE_TO_IDX,
)
Point = tuple[int, int] #from ..utils.typing import Point
from ..utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)

if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv



def world_obj_from_array(array: np.ndarray) -> Optional['WorldObj']:
    """
    Create a world object from array representation.
    """
    OBJECT_TO_CLS = {
        'wall': Wall,
        'floor': Floor,
        'door': Door,
        'key': Key,
        'ball': Ball,
        'box': Box,
        'goal': Goal,
        'lava': Lava,
    }

    if IDX_TO_OBJECT[array[0]] == 'empty':
        return None

    if IDX_TO_OBJECT[array[0]] in OBJECT_TO_CLS:
        cls = OBJECT_TO_CLS[IDX_TO_OBJECT[array[0]]]
        return cls.from_array(array)

    raise ValueError(f'Unknown object index: {array[0]}')


class WorldObj:
    """
    Base class for grid world objects.

    Attributes
    ----------
    array : np.ndarray[int] of shape (ARRAY_DIM,)
        Underlying array-encoding representation
    type : str
        The name of the object type
    color : str
        The color of the object
    state : int
        The state of the object
    contains : WorldObj or None
        The object contained by this object, if any
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.array = empty()
        self.array[0] = OBJECT_TO_IDX[type]
        self.array[1] = COLOR_TO_IDX[color]

    def __eq__(self, other: 'WorldObj') -> bool:
        return np.array_equal(self.array, other.array)

    @classmethod
    def from_array(cls, array: np.ndarray):
        """
        Create a world object from array representation.
        """
        obj = cls.__new__(cls)
        obj.array = array
        return obj

    @property
    def type(self) -> str:
        """
        The name of the object type.
        """
        return IDX_TO_OBJECT[self.array[0]]

    @property
    def color(self) -> str:
        """
        The color of the object.
        """
        return IDX_TO_COLOR[self.array[1]]

    @color.setter
    def color(self, value: str):
        """
        Set the color of the object.
        """
        self.array[1] = COLOR_TO_IDX[value]

    @property
    def state(self) -> int:
        """
        The state of the object.
        """
        return self.array[2]

    @state.setter
    def state(self, value: int):
        """
        Set the state of the object.
        """
        self.array[2] = value

    @property
    def contains(self) -> Optional['WorldObj']:
        """
        The object contained by this object, if any.
        """
        array = contents(self.array)
        if IDX_TO_OBJECT[array[0]] != 'empty':
            return world_obj_from_array(array)

    @contains.setter
    def contains(self, value):
        """
        Set the contents of this object.
        """
        if value is None:
            self.array[4:] = empty()[:4]
        else:
            self.array[4:] = value.array[:4]

    def can_overlap(self) -> bool:
        """
        Can an agent overlap with this?
        """
        return can_overlap(self.array)

    def can_pickup(self) -> bool:
        """
        Can an agent pick this up?
        """
        return can_pickup(self.array)

    def can_contain(self) -> bool:
        """
        Can this contain another object?
        """
        return can_contain(self.array)

    def see_behind(self) -> bool:
        """
        Can an agent see behind this object?
        """
        return see_behind(self.array)

    def toggle(self, env: 'MiniGridEnv', pos: tuple[int, int]) -> bool:
        """
        Method to trigger/toggle an action this object performs.
        """
        return False

    def encode(self) -> tuple[int, int, int]:
        """
        Encode the a description of this object as a 3-tuple of integers.
        """
        return tuple(self.array[:3])

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> Optional['WorldObj']:
        """
        Create an object from a 3-tuple state description.
        """
        array = empty()
        array[:3] = type_idx, color_idx, state
        return world_obj_from_array(array)

    def render(self, img: np.ndarray[int]):
        """
        Draw this object with the given renderer.
        """
        raise NotImplementedError


class Goal(WorldObj):
    """
    Goal object an agent may be searching for.
    """

    def __init__(self):
        super().__init__('goal', 'green')

    def render(self, img: np.ndarray[int]):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile an agent can walk over.
    """

    def __init__(self, color: str = 'blue'):
        super().__init__('floor', color)

    def render(self, img: np.ndarray[int]):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    """
    Lava object an agent can fall onto.
    """

    def __init__(self):
        super().__init__('lava', 'red')

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

    def __init__(self, color: str = 'grey'):
        super().__init__('wall', color)

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

    def __init__(self, color: str, is_open: bool = False, is_locked: bool = False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    @property
    def is_open(self) -> bool:
        """
        Whether the door is open.
        """
        return self.state == STATE_TO_IDX['open']

    @is_open.setter
    def is_open(self, value: bool):
        """
        Set the door to be open or closed.
        """
        if value:
            self.state = STATE_TO_IDX['open'] # set state to open
        elif not self.is_locked:
            self.state = STATE_TO_IDX['closed'] # closed (unless already locked)

    @property
    def is_locked(self) -> bool:
        """
        Whether the door is locked.
        """
        return self.state == STATE_TO_IDX['locked']

    @is_locked.setter
    def is_locked(self, value: bool):
        """
        Set the door to be locked or unlocked.
        """
        if value:
            self.state = STATE_TO_IDX['locked'] # set state to locked
        elif not self.is_open:
            self.state = STATE_TO_IDX['closed'] # closed (unless already open)

    def toggle(self, env: 'MiniGridEnv', pos: tuple[int, int]) -> bool:
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

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

    def __init__(self, color: str = 'blue'):
        super().__init__('key', color)

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

    def __init__(self, color: str = 'blue'):
        super().__init__('ball', color)

    def render(self, img: np.ndarray[int]):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    """
    Box object that may contain other objects.
    """

    def __init__(self, color: str, contains: Optional[WorldObj] = None):
        super().__init__('box', color)
        self.contains = contains

    def toggle(self, env: 'MiniGridEnv', pos: tuple[int, int]) -> bool:
        # Replace the box by its contents
        env.grid.set(pos[0], pos[1], self.contains)
        return True

    def render(self, img: np.ndarray[int]):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)
