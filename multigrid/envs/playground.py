from __future__ import annotations

from multigrid.core.mission import MissionSpace
from multigrid.core.roomgrid import RoomGrid



class PlaygroundEnv(RoomGrid):
    """
    .. image:: https://i.imgur.com/QBz99Vh.gif
        :width: 380

    ***********
    Description
    ***********

    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.

    *************
    Mission Space
    *************

    None

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

    None

    ***********
    Termination
    ***********

    The episode ends when the following condition is met:

    * Timeout (see ``max_steps``)

    *************************
    Registered Configurations
    *************************

    * ``MultiGrid-Playground-v0``
    """

    def __init__(
        self,
        room_size: int = 7,
        num_rows: int = 3,
        num_cols: int = 3,
        max_steps: int = 100,
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
        max_steps : int, default=100
            Maximum number of steps per episode
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """
        super().__init__(
            mission_space=MissionSpace.from_string(""),
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        """
        :meta private:
        """
        super()._gen_grid(width, height)
        self.connect_all()

        # Place random objects in the world
        for i in range(0, 12):
            col = self._rand_int(0, self.num_cols)
            row = self._rand_int(0, self.num_rows)
            self.add_object(col, row)

        # Place agents
        for agent in self.agents:
            self.place_agent(agent)
