from __future__ import annotations
from typing import Any

import numpy as np

from multigrid import MultiGridEnv

from ..core import Agent, Door, Grid, Key, Wall, WorldObj
from ..core.actions import Action
from ..core.constants import Color, Type





class LockedHallwayEnv(MultiGridEnv):
    """
    ***********
    Description
    ***********

    This environment consists of a hallway with multiple locked rooms on either side.
    To unlock each door, agents must first find the corresponding key,
    which may be in a different room. Agents are rewarded for each door they open.

    *************
    Mission Space
    *************

    "open all the doors"

    *****************
    Observation Space
    *****************

    The multi-agent observation space is a Dict mapping from agent index to
    corresponding agent observation space.

    Each agent observation is a dictionary with the following entries:

    * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
        Encoding of the agent's partially observable view of the environment,
        where each grid cell is encoded as a 3 dimensional tuple:
        (:class:`.Type`, :class:`.Color`, :class:`.State`)
    * direction : int
        Agent's direction (0: right, 1: down, 2: left, 3: up)
    * mission : Mission
        Task string corresponding to the current environment configuration

    ************
    Action Space
    ************

    The multi-agent action space is a Dict mapping from agent index to
    corresponding agent action space.

    Agent actions are discrete integer values, given by:

    +-----+--------------+-----------------------------+
    | Num | Name         | Action                      |
    +=====+==============+=============================+
    | 0   | left         | Turn left                   |
    +-----+--------------+-----------------------------+
    | 1   | right        | Turn right                  |
    +-----+--------------+-----------------------------+
    | 2   | forward      | Move forward                |
    +-----+--------------+-----------------------------+
    | 3   | pickup       | Pick up an object           |
    +-----+--------------+-----------------------------+
    | 4   | drop         | Drop an object              |
    +-----+--------------+-----------------------------+
    | 5   | toggle       | Toggle / activate an object |
    +-----+--------------+-----------------------------+
    | 6   | done         | Done completing task        |
    +-----+--------------+-----------------------------+

    *******
    Rewards
    *******

    A reward of ``1 - 0.9 * (step_count / max_steps)`` is given
    when each door is opened.

    ***********
    Termination
    ***********

    The episode ends if any one of the following conditions is met:

    * All doors are opened
    * Timeout (see ``max_steps``)

    *************************
    Registered Configurations
    *************************

    * ``MultiGrid-LockedHallway-4Rooms-v0``
    * ``MultiGrid-LockedHallway-6Rooms-v0``
    """

    def __init__(
        self,
        num_rooms=4,
        max_hallway_keys=1,
        max_keys_per_room=2,
        grid_size=13,
        max_steps=None,
        joint_reward=True,
        **kwargs):

        self.num_rooms = num_rooms
        self.max_hallway_keys = max_hallway_keys
        self.max_keys_per_room = max_keys_per_room

        if max_steps is None:
            max_steps = 16 * grid_size**2

        super().__init__(
            grid_size=grid_size,
            max_steps=max_steps,
            joint_reward=joint_reward,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Hallway walls
        hallway_left = width // 2 - 2
        hallway_right = width // 2 + 2

        # Create rooms
        num_rooms_per_side = self.num_rooms // 2
        room_size = (hallway_left + 1, height // num_rooms_per_side + 1)
        colors = [Color.red, Color.green, Color.blue, Color.yellow]
        colors = list(self.np_random.permutation([c.value for c in colors]))

        self.rooms = {}
        for n in range(num_rooms_per_side):
            
            j = n * (height // num_rooms_per_side)
            k = height // num_rooms_per_side // 2

            # Left side
            color = colors.pop()
            self.rooms[color] = Room(
                self, (0, j), room_size, (hallway_left, j + k), color)

            # Right side
            color = colors.pop()
            self.rooms[color] = Room(
                self, (hallway_right, j), room_size, (hallway_right, j + k), color)

        # Get doors
        self.doors = {color: self.rooms[color].door for color in self.rooms}

        # color_to_room = {room.color: room for room in self.rooms}
        color_sequence = self.np_random.permutation([c for c in self.rooms])

        # Randomly place keys in hallway
        index = self.np_random.integers(self.max_hallway_keys) + 1
        top, size = (hallway_left, 0), (hallway_right - hallway_left, height)
        for key_color in color_sequence[:index]:
            self.place_obj(Key(color=key_color), top=top, size=size)

        # Randomly place keys in rooms
        while index < len(color_sequence):
            k = self.np_random.integers(self.max_keys_per_room) + 1
            room_color = color_sequence[index - 1]
            key_colors = color_sequence[index : index + k]
            for key_color in key_colors:
                self.rooms[room_color].place_obj(Key(color=key_color))
                index += 1

        # Randomize the player start position and orientation
        for agent in self.agents:
            self.place_agent(
                agent,
                top=(hallway_left, 0),
                size=(hallway_right - hallway_left, height)
            )

    def reset(self, **kwargs):
        self.opened_doors = []
        return super().reset(**kwargs)
    
    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)

        # Reward for opening new doors
        for agent_id, action in actions.items():
            if action == Action.toggle:
                fwd_obj = self.grid.get(*self.agents[agent_id].front_pos)
                if isinstance(fwd_obj, Door) and fwd_obj.is_open:
                    if fwd_obj not in self.opened_doors:
                        self.opened_doors.append(fwd_obj)
                        reward[agent_id] += self._reward()

        # Check if all doors are opened
        if len(self.opened_doors) == len(self.doors):
            for agent in self.agents:
                terminated[agent.index] = True

        return obs, reward, terminated, truncated, info


class Room:

    def __init__(
        self,
        env: MultiGridEnv,
        top: tuple[int, int],
        size: tuple[int, int],
        door_pos: tuple[int, int],
        color: str,
        open: bool = False,
        locked: bool = True,
    ):
        """
        Parameters
        ----------
        env : MultiGridEnv
            Environment
        top : tuple[int, int]
            Top-left position of the room
        size : tuple[int, int]
            Tuple of room size as (width, height)
        door_pos : tuple[int, int]
            Position of room door
        color : str
            Color of room door
        open : bool, default=False
            Whether room door is open
        locked : bool, default=True
            Whether room door is locked
        """
        self.env = env
        self.top = top
        self.size = size

        # Get bounds
        left, top = top
        width = min(size[0], env.grid.width - left)
        height = min(size[1], env.grid.height - top)

        # Create walls
        env.grid.wall_rect(left, top, width, height)

        # Create door
        self.door_pos = door_pos
        self.door = Door(color=color, is_open=open, is_locked=locked)
        env.put_obj(self.door, *door_pos)

    @property
    def color(self) -> str:
        """
        Color of room door.
        """
        return self.door.color

    @property
    def is_open(self) -> bool:
        """
        Whether the room door is open.
        """
        return self.door.is_open

    @property
    def is_locked(self) -> bool:
        """
        Whether the room door is locked.
        """
        return self.door.is_locked

    def place_obj(self, obj: WorldObj):
        """
        Place object at random position within room.

        Parameters
        ----------
        obj : WorldObj
            Object to place
        """
        self.env.place_obj(obj, top=self.top, size=self.size)
