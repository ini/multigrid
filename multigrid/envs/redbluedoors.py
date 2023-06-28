from __future__ import annotations

from multigrid import MultiGridEnv
from multigrid.core import Action, Grid, MissionSpace
from multigrid.core.constants import Color
from multigrid.core.world_object import Door



class RedBlueDoorsEnv(MultiGridEnv):
    """
    .. image:: https://i.imgur.com/usbavAh.gif
        :width: 400

    ***********
    Description
    ***********

    This environment is a room with one red and one blue door facing
    opposite directions. Agents must open the red door and then open the blue door,
    in that order.
    
    The standard setting is cooperative, where all agents receive the reward
    upon completion of the task.

    *************
    Mission Space
    *************

    "open the red door then the blue door"

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

    * Any agent opens the blue door while the red door is open (success)
    * Any agent opens the blue door while the red door is not open (failure)
    * Timeout (see ``max_steps``)

    *************************
    Registered Configurations
    *************************

    * ``MultiGrid-RedBlueDoors-6x6-v0``
    * ``MultiGrid-RedBlueDoors-8x8-v0``
    """

    def __init__(
        self,
        size: int = 8,
        max_steps: int | None = None,
        joint_reward: bool = True,
        success_termination_mode: str = 'any',
        failure_termination_mode: str = 'any',
        **kwargs):
        """
        Parameters
        ----------
        size : int, default=8
            Width and height of the grid
        max_steps : int, optional
            Maximum number of steps per episode
        joint_reward : bool, default=True
            Whether all agents receive the reward when the task is completed
        success_termination_mode : 'any' or 'all', default='any'
            Whether to terminate the environment when any agent fails the task
            or after all agents fail the task
        failure_termination_mode : 'any' or 'all', default='any'
            Whether to terminate the environment when any agent fails the task
            or after all agents fail the task
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """
        self.size = size
        mission_space = MissionSpace.from_string("open the red door then the blue door")
        super().__init__(
            mission_space=mission_space,
            width=(2 * size),
            height=size,
            max_steps=max_steps or (20 * size**2),
            joint_reward=joint_reward,
            success_termination_mode=success_termination_mode,
            failure_termination_mode=failure_termination_mode,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        """
        :meta private:
        """
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the grid walls
        room_top = (width // 4, 0)
        room_size = (width // 2, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(*room_top, *room_size)

        # Place agents in the top-left corner
        for agent in self.agents:
            self.place_agent(agent, top=room_top, size=room_size)

        # Add a red door at a random position in the left wall
        x = room_top[0]
        y = self._rand_int(1, height - 1)
        self.red_door = Door(Color.red)
        self.grid.set(x, y, self.red_door)

        # Add a blue door at a random position in the right wall
        x = room_top[0] + room_size[0] - 1
        y = self._rand_int(1, height - 1)
        self.blue_door = Door(Color.blue)
        self.grid.set(x, y, self.blue_door)

    def step(self, actions):
        """
        :meta private:
        """
        obs, reward, terminated, truncated, info = super().step(actions)

        for agent_id, action in actions.items():
            if action == Action.toggle:
                agent = self.agents[agent_id]
                fwd_obj = self.grid.get(*agent.front_pos)
                if fwd_obj == self.blue_door and self.blue_door.is_open:
                    if self.red_door.is_open:
                        self.on_success(agent, reward, terminated)
                    else:
                        self.on_failure(agent, reward, terminated)
                        self.blue_door.is_open = False # close the door again

        return obs, reward, terminated, truncated, info
