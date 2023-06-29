from __future__ import annotations

import numpy as np

from collections import deque
from typing import Callable, Iterable, TypeVar

from .agent import Agent
from .constants import Color, Direction, Type
from .grid import Grid
from .world_object import Door, WorldObj
from ..base import MultiGridEnv



T = TypeVar('T')



def bfs(start_node: T, neighbor_fn: Callable[[T], Iterable[T]]) -> set[T]:
    """
    Run a breadth-first search from a starting node.

    Parameters
    ----------
    start_node : T
        Start node
    neighbor_fn : Callable(T) -> Iterable[T]
        Function that returns the neighbors of a node

    Returns
    -------
    visited : set[T]
        Set of nodes reachable from the start node
    """
    visited, queue = set(), deque([start_node])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(neighbor_fn(node))

    return visited

def reject_next_to(env: MultiGridEnv, pos: tuple[int, int]):
    """
    Function to filter out object positions that are right next to
    the agent's starting point
    """
    return any(np.linalg.norm(pos - env.agent_states.pos, axis=-1) <= 1)


class Room:
    """
    Room as an area inside a grid.
    """

    def __init__(self, top: tuple[int, int], size: tuple[int, int]):
        """
        Parameters
        ----------
        top : tuple[int, int]
            Top-left position of the room
        size : tuple[int, int]
            Room size as (width, height)
        """
        self.top, self.size = top, size
        Point = tuple[int, int] # typing alias

        # Mapping of door objects and door positions
        self.doors: dict[Direction, Door | None] = {d: None for d in Direction}
        self.door_pos: dict[Direction, Point | None] = {d: None for d in Direction}

        # Mapping of rooms adjacent to this one
        self.neighbors: dict[Direction, Room | None] = {d: None for d in Direction}

        # List of objects contained in this room
        self.objs = []

    @property
    def locked(self) -> bool:
        """
        Return whether this room is behind a locked door.
        """
        return any(door and door.is_locked for door in self.doors.values())

    def set_door_pos(
        self,
        dir: Direction,
        random: np.random.Generator | None = None) -> tuple[int, int]:
        """
        Set door position in the given direction.

        Parameters
        ----------
        dir : Direction
            Direction of wall to place door
        random : np.random.Generator, optional
            Random number generator (if provided, door position will be random)
        """
        left, top = self.top
        right, bottom = self.top[0] + self.size[0] - 1, self.top[1] + self.size[1] - 1,

        if dir == Direction.right:
            if random:
                self.door_pos[dir] = (right, random.integers(top + 1, bottom))
            else:
                self.door_pos[dir] = (right, (top + bottom) // 2)

        elif dir == Direction.down:
            if random:
                self.door_pos[dir] = (random.integers(left + 1, right), bottom)
            else:
                self.door_pos[dir] = ((left + right) // 2, bottom)

        elif dir == Direction.left:
            if random:
                self.door_pos[dir] = (left, random.integers(top + 1, bottom))
            else:
                self.door_pos[dir] = (left, (top + bottom) // 2)

        elif dir == Direction.up:
            if random:
                self.door_pos[dir] = (random.integers(left + 1, right), top)
            else:
                self.door_pos[dir] = ((left + right) // 2, top)

        return self.door_pos[dir]

    def pos_inside(self, x: int, y: int) -> bool:
        """
        Check if a position is within the bounds of this room.
        """
        left_x, top_y = self.top
        width, height = self.size
        return left_x <= x < left_x + width and top_y <= y < top_y + height


class RoomGrid(MultiGridEnv):
    """
    Environment with multiple rooms and random objects.
    This is meant to serve as a base class for other environments.
    """

    def __init__(
        self,
        room_size: int = 7,
        num_rows: int = 3,
        num_cols: int = 3,
        **kwargs):
        """
        Parameters
        ----------
        room_size : int, default=7
            Width and height for each of the rooms
        num_rows : int, default=3
            Number of rows of rooms
        num_cols : int, default=3
            Number of columns of rooms
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """
        assert room_size >= 3
        assert num_rows > 0
        assert num_cols > 0
        self.room_size = room_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        height = (room_size - 1) * num_rows + 1
        width = (room_size - 1) * num_cols + 1
        super().__init__(width=width, height=height, **kwargs)

    def get_room(self, col: int, row: int) -> Room:
        """
        Get the room at the given column and row.

        Parameters
        ----------
        col : int
            Column of the room
        row : int
            Row of the room
        """
        assert 0 <= col < self.num_cols
        assert 0 <= row < self.num_rows
        return self.room_grid[row][col]

    def room_from_pos(self, x: int, y: int) -> Room:
        """
        Get the room a given position maps to.

        Parameters
        ----------
        x : int
            Grid x-coordinate
        y : int
            Grid y-coordinate
        """
        col = x // (self.room_size - 1)
        row = y // (self.room_size - 1)
        return self.get_room(col, row)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)
        self.room_grid = [[None] * self.num_cols for _ in range(self.num_rows)]

        # Create rooms
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                room = Room(
                    (col * (self.room_size - 1), row * (self.room_size - 1)),
                    (self.room_size, self.room_size),
                )
                self.room_grid[row][col] = room
                self.grid.wall_rect(*room.top, *room.size) # generate walls

        # Create connections between rooms
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                room = self.room_grid[row][col]
                if col < self.num_cols - 1:
                    room.neighbors[Direction.right] = self.room_grid[row][col + 1]
                if row < self.num_rows - 1:
                    room.neighbors[Direction.down] = self.room_grid[row + 1][col]
                if col > 0:
                    room.neighbors[Direction.left] = self.room_grid[row][col - 1]
                if row > 0:
                    room.neighbors[Direction.up] = self.room_grid[row - 1][col]

        # Agents start in the middle, facing right
        self.agent_states.dir = Direction.right
        self.agent_states.pos = (
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
        )

    def place_in_room(
        self, col: int, row: int, obj: WorldObj) -> tuple[WorldObj, tuple[int, int]]:
        """
        Add an existing object to the given room.

        Parameters
        ----------
        col : int
            Room column
        row : int
            Room row
        obj : WorldObj
            Object to add
        """
        room = self.get_room(col, row)
        pos = self.place_obj(
            obj, room.top, room.size, reject_fn=reject_next_to, max_tries=1000)
        room.objs.append(obj)
        return obj, pos

    def add_object(
        self,
        col: int,
        row: int,
        kind: Type | None = None,
        color: Color | None = None) -> tuple[WorldObj, tuple[int, int]]:
        """
        Create a new object in the given room.

        Parameters
        ----------
        col : int
            Room column
        row : int
            Room row
        kind : str, optional
            Type of object to add (random if not specified)
        color : str, optional
            Color of the object to add (random if not specified)
        """
        kind = kind or self._rand_elem([Type.key, Type.ball, Type.box])
        color = color or self._rand_color()
        obj = WorldObj(type=kind, color=color)
        return self.place_in_room(col, row, obj)

    def add_door(
        self,
        col: int,
        row: int,
        dir: Direction | None = None,
        color: Color | None = None,
        locked: bool | None = None,
        rand_pos: bool = True) -> tuple[Door, tuple[int, int]]:
        """
        Add a door to a room, connecting it to a neighbor.

        Parameters
        ----------
        col : int
            Room column
        row : int
            Room row
        dir : Direction, optional
            Which wall to put the door on (random if not specified)
        color : Color, optional
            Color of the door (random if not specified)
        locked : bool, optional
            Whether the door is locked (random if not specified)
        rand_pos : bool, default=True
            Whether to place the door at a random position on the room wall
        """
        room = self.get_room(col, row)

        # Need to make sure that there is a neighbor along this wall
        # and that there is not already a door
        if dir is None:
            while room.neighbors[dir] is None or room.doors[dir] is not None:
                dir = self._rand_elem(Direction)
        else:
            assert room.neighbors[dir] is not None, "no neighbor in this direction"
            assert room.doors[dir] is None, "door already exists"

        # Create the door
        color = color if color is not None else self._rand_color()
        locked = locked if locked is not None else self._rand_bool()
        door = Door(color, is_locked=locked)
        pos = room.set_door_pos(dir, random=self.np_random if rand_pos else None)
        self.put_obj(door, *pos)

        # Connect the door to the neighboring room
        room.doors[dir] = door
        room.neighbors[dir].doors[(dir + 2) % 4] = door

        return door, pos

    def remove_wall(self, col: int, row: int, dir: Direction):
        """
        Remove a wall between two rooms.

        Parameters
        ----------
        col : int
            Room column
        row : int
            Room row
        dir : Direction
            Direction of the wall to remove
        """
        room = self.get_room(col, row)
        assert room.doors[dir] is None, "door exists on this wall"
        assert room.neighbors[dir], "invalid wall"

        tx, ty = room.top
        w, h = room.size

        # Remove the wall
        if dir == Direction.right:
            for i in range(1, h - 1):
                self.grid.set(tx + w - 1, ty + i, None)
        elif dir == Direction.down:
            for i in range(1, w - 1):
                self.grid.set(tx + i, ty + h - 1, None)
        elif dir == Direction.left:
            for i in range(1, h - 1):
                self.grid.set(tx, ty + i, None)
        elif dir == Direction.up:
            for i in range(1, w - 1):
                self.grid.set(tx + i, ty, None)
        else:
            assert False, "invalid wall index"

        # Mark the rooms as connected
        room.doors[dir] = True
        room.neighbors[dir].doors[(dir + 2) % 4] = True

    def place_agent(
        self,
        agent: Agent,
        col: int | None = None,
        row: int | None = None,
        rand_dir: bool = True) -> tuple[int, int]:
        """
        Place an agent in a room.

        Parameters
        ----------
        agent : Agent
            Agent to place
        col : int, optional
            Room column to place the agent in (random if not specified)
        row : int, optional
            Room row to place the agent in (random if not specified)
        rand_dir : bool, default=True
            Whether to select a random agent direction
        """
        col = col if col is not None else self._rand_int(0, self.num_cols)
        row = row if row is not None else self._rand_int(0, self.num_rows)
        room = self.get_room(col, row)

        # Find a position that is not right in front of an object
        while True:
            super().place_agent(agent, room.top, room.size, rand_dir, max_tries=1000)
            front_cell = self.grid.get(*agent.front_pos)
            if front_cell is None or front_cell.type == Type.wall:
                break

        return agent.state.pos

    def connect_all(
        self,
        door_colors: list[Color] = list(Color),
        max_itrs: int = 5000) -> list[Door]:
        """
        Make sure that all rooms are reachable by the agent from its
        starting position.

        Parameters
        ----------
        door_colors : list[Color], default=list(Color)
            Color options for creating doors
        max_itrs : int, default=5000
            Maximum number of iterations to try to connect all rooms
        """
        added_doors = []
        neighbor_fn = lambda room: [
            room.neighbors[dir] for dir in Direction if room.doors[dir] is not None]
        start_room = self.get_room(0, 0)

        for i in range(max_itrs):
            # If all rooms are reachable, stop
            reachable_rooms = bfs(start_room, neighbor_fn)
            if len(reachable_rooms) == self.num_rows * self.num_cols:
                return added_doors

            # Pick a random room and door position
            col = self._rand_int(0, self.num_cols)
            row = self._rand_int(0, self.num_rows)
            dir = self._rand_elem(Direction)
            room = self.get_room(col, row)

            # If there is already a door there, skip
            if not room.neighbors[dir] or room.doors[dir]:
                continue

            neighbor_room = room.neighbors[dir]
            assert neighbor_room is not None
            if room.locked or neighbor_room.locked:
                continue

            # Add a new door
            color = self._rand_elem(door_colors)
            door, _ = self.add_door(col, row, dir=dir, color=color, locked=False)
            added_doors.append(door)

        raise RecursionError('connect_all() failed')

    def add_distractors(
        self,
        col: int | None = None,
        row: int | None = None,
        num_distractors: int = 10,
        all_unique: bool = True) -> list[WorldObj]:
        """
        Add random objects that can potentially distract / confuse the agent.

        Parameters
        ----------
        col : int, optional
            Room column to place the objects in (random if not specified)
        row : int, optional
            Room row to place the objects in (random if not specified)
        num_distractors : int, default=10
            Number of distractor objects to add
        all_unique : bool, default=True
            Whether all distractor objects should be unique with respect to (type, color)
        """
        # Collect keys for existing room objects
        room_objs = (obj for row in self.room_grid for room in row for obj in room.objs)
        room_obj_keys = {(obj.type, obj.color) for obj in room_objs}  

        # Add distractors
        distractors = []
        while len(distractors) < num_distractors:
            color = self._rand_color()
            type = self._rand_elem([Type.key, Type.ball, Type.box])

            if all_unique and (type, color) in room_obj_keys:
                continue

            # Add the object to a random room if no room specified
            col = col if col is not None else self._rand_int(0, self.num_cols)
            row = row if row is not None else self._rand_int(0, self.num_rows)
            distractor, _ = self.add_object(col, row, kind=type, color=color)

            room_obj_keys.append((type, color))
            distractors.append(distractor)

        return distractors
