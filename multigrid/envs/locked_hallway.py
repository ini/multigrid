from __future__ import annotations

from math import ceil
from multigrid import MultiGridEnv
from multigrid.core.actions import Action
from multigrid.core.constants import Color, Direction
from multigrid.core.mission import MissionSpace
from multigrid.core.roomgrid import Room, RoomGrid
from multigrid.core.world_object import Door, Key



class LockedHallwayEnv(RoomGrid):
    """
    .. image:: https://i.imgur.com/VylPtnn.gif
        :width: 325

    ***********
    Description
    ***********

    This environment consists of a hallway with multiple locked rooms on either side.
    To unlock each door, agents must first find the corresponding key,
    which may be in another locked room. Agents are rewarded for each door they unlock.

    *************
    Mission Space
    *************

    "unlock all the doors"

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
    when a door is unlocked.

    ***********
    Termination
    ***********

    The episode ends if any one of the following conditions is met:

    * All doors are unlocked
    * Timeout (see ``max_steps``)

    *************************
    Registered Configurations
    *************************

    * ``MultiGrid-LockedHallway-2Rooms-v0``
    * ``MultiGrid-LockedHallway-4Rooms-v0``
    * ``MultiGrid-LockedHallway-6Rooms-v0``
    """

    def __init__(
        self,
        num_rooms: int = 6,
        room_size: int = 5,
        max_hallway_keys: int = 1,
        max_keys_per_room: int = 2,
        max_steps: int | None = None,
        joint_reward: bool = True,
        **kwargs):
        """
        Parameters
        ----------
        num_rooms : int, default=6
            Number of rooms in the environment
        room_size : int, default=5
            Width and height for each of the rooms
        max_hallway_keys : int, default=1
            Maximum number of keys in the hallway
        max_keys_per_room : int, default=2
            Maximum number of keys in each room
        max_steps : int, optional
            Maximum number of steps per episode
        joint_reward : bool, default=True
            Whether all agents receive the same reward
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """
        assert room_size >= 4
        assert num_rooms % 2 == 0

        self.num_rooms = num_rooms
        self.max_hallway_keys = max_hallway_keys
        self.max_keys_per_room = max_keys_per_room

        if max_steps is None:
            max_steps = 8 * num_rooms * room_size**2

        super().__init__(
            mission_space=MissionSpace.from_string("unlock all the doors"),
            room_size=room_size,
            num_rows=(num_rooms // 2),
            num_cols=3,
            max_steps=max_steps,
            joint_reward=joint_reward,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        LEFT, HALLWAY, RIGHT = range(3) # columns
        color_sequence = list(Color) * ceil(self.num_rooms / len(Color))
        color_sequence = self._rand_perm(color_sequence)[:self.num_rooms]

        # Create hallway
        for row in range(self.num_rows - 1):
            self.remove_wall(HALLWAY, row, Direction.down)

        # Add doors
        self.rooms: dict[Color, Room] = {}
        door_colors = self._rand_perm(color_sequence)
        for row in range(self.num_rows):
            for col, dir in ((LEFT, Direction.right), (RIGHT, Direction.left)):
                color = door_colors.pop()
                self.rooms[color] = self.get_room(col, row)
                self.add_door(
                    col, row, dir=dir, color=color, locked=True, rand_pos=False)

        # Place keys in hallway
        num_hallway_keys = self._rand_int(1, self.max_hallway_keys + 1)
        hallway_top = self.get_room(HALLWAY, 0).top
        hallway_size = (self.get_room(HALLWAY, 0).size[0], self.height)
        for key_color in color_sequence[:num_hallway_keys]:
            self.place_obj(Key(color=key_color), top=hallway_top, size=hallway_size)

        # Place keys in rooms
        key_index = num_hallway_keys
        while key_index < len(color_sequence):
            room = self.rooms[color_sequence[key_index - 1]]
            num_room_keys = self._rand_int(1, self.max_keys_per_room + 1)
            for key_color in color_sequence[key_index : key_index + num_room_keys]:
                self.place_obj(Key(color=key_color), top=room.top, size=room.size)
                key_index += 1

        # Place agents in hallway
        for agent in self.agents:
            MultiGridEnv.place_agent(self, agent, top=hallway_top, size=hallway_size)

    def reset(self, **kwargs):
        """
        :meta private:
        """
        self.unlocked_doors = []
        return super().reset(**kwargs)
    
    def step(self, actions):
        """
        :meta private:
        """
        observations, rewards, terminations, truncations, infos = super().step(actions)

        # Reward for unlocking doors
        for agent_id, action in actions.items():
            if action == Action.toggle:
                fwd_obj = self.grid.get(*self.agents[agent_id].front_pos)
                if isinstance(fwd_obj, Door) and not fwd_obj.is_locked:
                    if fwd_obj not in self.unlocked_doors:
                        self.unlocked_doors.append(fwd_obj)
                        if self.joint_reward:
                            for k in rewards:
                                rewards[k] += self._reward()
                        else:
                            rewards[agent_id] += self._reward()

        # Check if all doors are unlocked
        if len(self.unlocked_doors) == len(self.rooms):
            for agent in self.agents:
                terminations[agent.index] = True

        return observations, rewards, terminations, truncations, infos
