import numpy as np

from typing import Optional, TYPE_CHECKING

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

if TYPE_CHECKING:
    from .agent import Agent
    from ..multigrid_env import MultiGridEnv



class WorldObjState(np.ndarray):
    """
    State for a `WorldObj` grid object.

    The state is a 4-dimensional array, indexed as follows:
        * 0: object type
        * 1: object color
        * 2: object state (i.e. open/closed/locked)
        * 3: WorldObjState of contained object (if any)

    WorldObjState objects also support vectorized operations,
    in which case the first (n - 1) dimensions represent a "batch" of states,
    and the last dimension represents each index in a state array.

    Attributes
    ----------
    type : str
        The name of the object type
    color : str
        The name of the object color
    state : str
        The name of the object state (i.e. 'open', 'closed', or 'locked')
    contains : WorldObjState
        The WorldObjState of the object contained by this object

    Examples
    --------
    Create a WorldObjState:

    >>> state = WorldObjState(type='ball', color='blue')
    >>> state
    WorldObjState([6, 2, 0, 0])
    >>> state.type
    'ball'
    >>> state.color
    'blue'

    Change attributes of a WorldObjState:
    >>> state.color = 'green'
    >>> state
    WorldObjState([6, 1, 0, 0])
    >>> state.color
    'green'

    Create a vectorized WorldObjState:

    >>> grid_state = WorldObjState(2, 2)
    >>> grid_state.type
    array([['empty', 'empty'],
           ['empty', 'empty']], dtype='<U6')

    Vector operations on a WorldObjState:

    >>> grid_state[:, 1].type = 'wall'
    >>> grid_state.type
    array([['empty', 'wall'],
           ['empty', 'wall']], dtype='<U6')
    """
    dim = 4
    encode_dim = 3
    max_contain_depth = 2
    _bases = [max(IDX_TO_OBJECT) + 1, max(IDX_TO_COLOR) + 1, len(IDX_TO_STATE) + 1]
    _bases = [*_bases, np.prod(_bases) ** max_contain_depth]
    _types = np.array(list(OBJECT_TO_IDX.keys()))
    _colors = np.array(list(COLOR_TO_IDX.keys()))
    _states = np.array(list(STATE_TO_IDX.keys()))
    _object_to_idx = np.vectorize(OBJECT_TO_IDX.__getitem__)
    _color_to_idx = np.vectorize(COLOR_TO_IDX.__getitem__)
    _state_to_idx = np.vectorize(STATE_TO_IDX.__getitem__)
    _empty = None

    def __new__(cls, *dims: int, type: str = 'empty', color: Optional[str] = None):
        """
        Parameters
        ----------
        dims : int
            Batch dimensions for vectorized WorldObjState
        type : str, default='empty'
            The name of the object type
        color : str, optional
            The name of the object color
        """
        obj = np.zeros(shape=dims+(cls.dim,), dtype=int).view(cls)
        obj[..., 0] = OBJECT_TO_IDX[type]
        if color is not None:
            obj[..., 1] = COLOR_TO_IDX[color]

        return obj

    @staticmethod
    def empty() -> 'WorldObjState':
        """
        Return reference to a fixed empty WorldObjState object.
        """
        if WorldObjState._empty is None:
            WorldObjState._empty = WorldObjState(type='empty')

        return WorldObjState._empty

    @classmethod
    def from_int(cls, n: int):
        """
        Convert a mixed radix integer-encoding to a WorldObjState object.
        """
        n_arr = np.array(n)
        x = cls(*n.shape)
        for i in range(cls.dim):
            x[..., i] = n_arr % cls._bases[i]
            n_arr //= cls._bases[i]

        x[np.array(n) == 0] = WorldObjState(type='empty')
        return x

    @classmethod
    def from_array(cls, arr: np.ndarray[int]):
        """
        Convert a numpy array to a WorldObjState object.
        """
        assert arr.shape[-1] == cls.dim
        return arr.view(cls)

    @property
    def type(self) -> str:
        """
        Return the name of the object type.
        """
        out = self._types[self[..., 0]]
        return out.item() if out.ndim == 0 else out

    @type.setter
    def type(self, value: str):
        """
        Set the name of the object type.
        """
        self[..., 0] = self._object_to_idx(value)

    @property
    def color(self) -> str:
        """
        Return the name of the object color.
        """
        out = self._colors[self[..., 1]]
        return out.item() if out.ndim == 0 else out

    @color.setter
    def color(self, value: str):
        """
        Set the name of the object color.
        """
        self[..., 1] = self._color_to_idx(value)

    @property
    def state(self) -> str:
        """
        Return the name of the object state (i.e. 'open', 'closed', or 'locked').
        """
        out = self._states[self[..., 2]]
        return out.item() if out.ndim == 0 else out

    @state.setter
    def state(self, value: str):
        """
        Set the name of the object state (i.e. open/closed/locked).
        """
        self[..., 2] = self._state_to_idx(value)

    @property
    def contains(self) -> 'WorldObjState':
        """
        Return the state object contained by this object.
        """
        return self.from_int(self[..., 3])

    @contains.setter
    def contains(self, obj_state: 'WorldObjState'):
        """
        Set the object state contained by this object.
        """
        self[..., 3] = obj_state.to_int()

    def can_overlap(self) -> bool:
        """
        Can an agent overlap with this?
        """
        if self.ndim == 1:
            return can_overlap(*self)

        mask = (self[..., 0] == OBJECT_TO_IDX['empty']) # can overlap empty
        mask |= (self[..., 0] == OBJECT_TO_IDX['goal']) # can overlap goal
        mask |= (self[..., 0] == OBJECT_TO_IDX['floor']) # can overlap floor
        mask |= (self[..., 0] == OBJECT_TO_IDX['lava']) # can overlap lava
        mask |= ( # can overlap open door
            (self[..., 0] == OBJECT_TO_IDX['door']) # door
            & (self[..., 2] == STATE_TO_IDX['open']) # and open
        )
        return mask

    def can_pickup(self) -> bool:
        """
        Can an agent pick this up?
        """
        if self.ndim == 1:
            return can_pickup(*self)

        mask = (self[..., 0] == OBJECT_TO_IDX['key']) # can pickup key
        mask |= (self[..., 0] == OBJECT_TO_IDX['ball']) # can pickup ball
        mask |= (self[..., 0] == OBJECT_TO_IDX['box']) # can pickup box
        return mask

    def can_contain(self) -> bool:
        """
        Can this contain another object?
        """
        return (self[..., 0] == OBJECT_TO_IDX['box']) # only boxes can contain objects

    def see_behind(self) -> bool:
        """
        Can an agent see behind this object?
        """
        neg_mask = (self[..., 0] == OBJECT_TO_IDX['wall']) # CANNOT see behind wall
        neg_mask |= ( # CANNOT see behind non-open door
            (self[..., 0] == OBJECT_TO_IDX['door']) # door
            & (self[..., 2] != STATE_TO_IDX['open']) # and not open
        )
        return ~neg_mask

    def encode(self) -> np.ndarray[int]:
        """
        Encode a description of this object state.
        """
        return self[..., :self.encode_dim]

    def to_int(self):
        """
        Encode this object state as a mixed radix integer.
        """
        base, n = 1, np.zeros(self.shape[:-1])
        for i in range(self.dim):
            n += (self[..., i] * base)
            base *= self._bases[i]

        n[n == OBJECT_TO_IDX['empty']] = 0
        return n


class WorldObj:
    """
    Base class for grid world objects.

    Attributes
    ----------
    state : WorldObjState
        State for this object
    type : str
        The name of the object type
    color : str
        The color of the object
    contains : WorldObj or None
        The object contained by this object, if any
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.state = WorldObjState(type=type, color=color)

    def __eq__(self, other: 'WorldObj') -> bool:
        return (
            np.array_equal(self.state, other.state)
            and np.may_share_memory(self.state, other.state)
        )

    @staticmethod
    def from_state(state: WorldObjState) -> 'WorldObj':
        """
        Create a world object from the given state.
        """
        assert state.ndim == 1
        object_idx_to_class = {
            2: Wall,
            3: Floor,
            4: Door,
            5: Key,
            6: Ball,
            7: Box,
            8: Goal,
            9: Lava,
        }

        if state[0] == OBJECT_TO_IDX['empty']:
            return None

        elif state[0] in object_idx_to_class:
            cls = object_idx_to_class[state[0]]
            obj = cls.__new__(cls)
            obj.state = state
            return obj

        raise ValueError(f'Unknown object type: {state.type}')

    @property
    def type(self) -> str:
        """
        The name of the object type.
        """
        return self.state.type

    @property
    def color(self) -> str:
        """
        The color of the object.
        """
        return self.state.color

    @color.setter
    def color(self, value: str):
        """
        Set the color of the object.
        """
        self.state.color = value

    @property
    def contains(self) -> Optional['WorldObj']:
        """
        The object contained by this object, if any.
        """
        return WorldObj.from_state(self.state.contains)

    @contains.setter
    def contains(self, value: Optional['WorldObj']):
        """
        Set the contents of this object.
        """
        self.state.contains = WorldObjState() if value is None else value.state

    def can_overlap(self) -> bool:
        """
        Can an agent overlap with this?
        """
        return self.state.can_overlap()

    def can_pickup(self) -> bool:
        """
        Can an agent pick this up?
        """
        return self.state.can_pickup()

    def can_contain(self) -> bool:
        """
        Can this contain another object?
        """
        return self.state.can_contain()

    def see_behind(self) -> bool:
        """
        Can an agent see behind this object?
        """
        return self.state.see_behind()

    def toggle(self, env: 'MultiGridEnv', agent: 'Agent', pos: tuple[int, int]) -> bool:
        """
        Method to trigger/toggle an action this object performs.
        """
        return False

    def encode(self) -> tuple[int, int, int]:
        """
        Encode a description of this object as a 3-tuple of integers.
        """
        return tuple(self.state.encode())

    @staticmethod
    def decode(type_idx: int, color_idx: int, state_idx: int) -> Optional['WorldObj']:
        """
        Create an object from a 3-tuple state description.
        """
        arr = np.array([type_idx, color_idx, state_idx, 0])
        return WorldObj.from_state(WorldObjState.from_array(arr))

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
        return self.state.state == 'open'

    @is_open.setter
    def is_open(self, value: bool):
        """
        Set the door to be open or closed.
        """
        if value:
            self.state.state = 'open' # set state to open
        elif not self.is_locked:
            self.state.state = 'closed' # set state to closed (unless already locked)

    @property
    def is_locked(self) -> bool:
        """
        Whether the door is locked.
        """
        return self.state.state == 'locked'

    @is_locked.setter
    def is_locked(self, value: bool):
        """
        Set the door to be locked or unlocked.
        """
        if value:
            self.state.state = 'locked' # set state to locked
        elif not self.is_open:
            self.state.state = 'closed' # set state to closed (unless already open)

    def toggle(self, env: 'MultiGridEnv', agent: 'Agent', pos: tuple[int, int]) -> bool:
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(agent.state.carrying, Key):
                if agent.state.carrying.color == self.color:
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

    def toggle(self, env: 'MultiGridEnv', agent: 'Agent', pos: tuple[int, int]) -> bool:
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
