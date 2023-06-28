from __future__ import annotations

from multigrid.core.constants import Color, Direction, Type
from multigrid.core.mission import MissionSpace
from multigrid.core.roomgrid import RoomGrid
from multigrid.core.world_object import Ball



class BlockedUnlockPickupEnv(RoomGrid):
    """
    .. image:: https://i.imgur.com/uSFi059.gif
        :width: 275

    ***********
    Description
    ***********

    The objective is to pick up a box which is placed in another room, behind a
    locked door. The door is also blocked by a ball which must be moved before
    the door can be unlocked. Hence, agents must learn to move the ball,
    pick up the key, open the door and pick up the object in the other
    room.

    The standard setting is cooperative, where all agents receive the reward
    when the task is completed.

    *************
    Mission Space
    *************

    "pick up the ``{color}`` box"

    ``{color}`` is the color of the box. Can be any :class:`.Color`.

    *****************
    Observation Space
    *****************

    The multi-agent observation space is a Dict mapping from agent index to
    corresponding agent observation space.

    Each agent observation is a dictionary with the following entries:

    * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
        Encoding of the agent's partially observable view of the environment,
        where the object at each grid cell is encoded as a vector:
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

    A reward of ``1 - 0.9 * (step_count / max_steps)`` is given for success,
    and ``0`` for failure.

    ***********
    Termination
    ***********

    The episode ends if any one of the following conditions is met:

    * Any agent picks up the correct box
    * Timeout (see ``max_steps``)

    *************************
    Registered Configurations
    *************************

    * ``MultiGrid-BlockedUnlockPickup-v0``
    """

    def __init__(
        self,
        room_size: int = 6,
        max_steps: int | None = None,
        joint_reward: bool = True,
        **kwargs):
        """
        Parameters
        ----------
        room_size : int, default=6
            Width and height for each of the two rooms
        max_steps : int, optional
            Maximum number of steps per episode
        joint_reward : bool, default=True
            Whether all agents receive the reward when the task is completed
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """
        assert room_size >= 4
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[list(Color), [Type.box, Type.key]],
        )
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps or (16 * room_size**2),
            joint_reward=joint_reward,
            success_termination_mode='any',
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        """
        :meta private:
        """
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        self.obj, _ = self.add_object(1, 0, kind=Type.box)

        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, Direction.right, locked=True)

        # Block the door with a ball
        self.grid.set(pos[0] - 1, pos[1], Ball(color=self._rand_color()))

        # Add a key to unlock the door
        self.add_object(0, 0, Type.key, door.color)

        # Place agents in the left room
        for agent in self.agents:
            self.place_agent(agent, 0, 0)

        self.mission = f"pick up the {self.obj.color} {self.obj.type}"

    def step(self, actions):
        """
        :meta private:
        """
        obs, reward, terminated, truncated, info = super().step(actions)
        for agent in self.agents:
            if agent.state.carrying == self.obj:
                self.on_success(agent, reward, terminated)

        return obs, reward, terminated, truncated, info
