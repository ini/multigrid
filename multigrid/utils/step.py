import numba as nb
import numpy as np

from ..core.constants import DIR_TO_VEC, OBJECT_TO_IDX, STATE_TO_IDX
from ..core.world_object import WorldObjState



# WorldObjState indices
TYPE = 0
COLOR = 1
STATE = 2
CONTENTS = 3

# AgentState indices
DIR = 2
POS_X = 3
POS_Y = 4
TERMINATED = 5
CARRYING = [6, 7, 8, 9]

# Object type indices
EMPTY = OBJECT_TO_IDX['empty']
FLOOR = OBJECT_TO_IDX['floor']
DOOR = OBJECT_TO_IDX['door']
KEY = OBJECT_TO_IDX['key']
BALL = OBJECT_TO_IDX['ball']
BOX = OBJECT_TO_IDX['box']
GOAL = OBJECT_TO_IDX['goal']
LAVA = OBJECT_TO_IDX['lava']

# Object state indices (i.e. 'open', 'closed', 'locked')
OPEN = STATE_TO_IDX['open']
CLOSED = STATE_TO_IDX['closed']
LOCKED = STATE_TO_IDX['locked']

# Other constants
DIR_TO_VEC = np.array(DIR_TO_VEC)
BASES = np.array(WorldObjState._bases)



### Functions

@nb.njit(cache=True)
def handle_actions(
    action: np.ndarray[int],
    order: np.ndarray[int],
    grid_state: np.ndarray[int],
    agent_state: np.ndarray[int]):
    """
    Handle the actions taken by the agents.
    Update the grid state, agent state, and rewards.

    Parameters
    ----------
    action : np.ndarray[int] of shape (num_agents,)
        The action taken by each agent
    order : np.ndarray[int] of shape (num_agents,)
        The order in which the agents take their actions
    grid_state : np.ndarray[int] of shape (width, height, grid_state_dim)
        The state of the grid
    agent_state : np.ndarray[int] of shape (num_agents, agent_state_dim)
        The state of each agent

    Returns
    -------
    rewards : np.ndarray[float] of shape (num_agents,)
        The rewards received by each agent
    """
    rewards = np.zeros(len(agent_state), dtype=np.float_)

    # Get the agent locations
    agent_location_mask = np.zeros((len(order), 2), dtype=np.bool_)
    for agent in range(len(agent_state)):
        x, y = agent_state[agent, POS_X], agent_state[agent, POS_Y]
        agent_location_mask[x, y] = True

    for i in range(len(order)):
        agent = order[i]

        # Unpack agent state
        agent_dir = agent_state[agent, DIR]
        agent_x = agent_state[agent, POS_X]
        agent_y = agent_state[agent, POS_Y]

        # Get the cell in front of the agent
        dir_vec = DIR_TO_VEC[agent_dir]
        fwd_x = agent_state[agent, POS_X] + dir_vec[0] 
        fwd_y = agent_state[agent, POS_Y] + dir_vec[1]
        fwd_state = grid_state[fwd_x, fwd_y]

        # Check if the agent is terminated
        if agent_state[agent, TERMINATED]:
            continue

        # Rotate left
        if action[agent] == 0:
            agent_state[agent, DIR] = (agent_state[agent, DIR] - 1) % 4

        # Rotate right
        elif action[agent] == 1:
            agent_state[agent, DIR] = (agent_state[agent, DIR] + 1) % 4

        # Move forward
        elif action[agent] == 2:
            if can_overlap(fwd_state) and not agent_location_mask[fwd_x, fwd_y]:
                agent_location_mask[agent_x, agent_y] = False
                agent_location_mask[fwd_x, fwd_y] = True
                agent_state[agent, POS_X] = fwd_x
                agent_state[agent, POS_Y] = fwd_y

            if fwd_state[0] == LAVA:
                agent_state[agent, TERMINATED] = True # set terminated to True
            elif fwd_state[0] == GOAL:
                agent_state[agent, TERMINATED] = True # set terminated to True
                rewards[agent] += 1

        # Pick up an object
        elif action[agent] == 3:
            if can_pickup(fwd_state):
                if agent_state[agent, 6] == EMPTY:
                    agent_state[agent, 6:10] = fwd_state
                    grid_state[fwd_x, fwd_y] = (EMPTY, 0, 0, 0)

        # Drop an object
        elif action[agent] == 4:
            if fwd_state[0] == EMPTY:
                if agent_state[agent, 6] != EMPTY:
                    grid_state[fwd_x, fwd_y] = agent_state[agent, 6:10]
                    agent_state[agent, 6:10] = (EMPTY, 0, 0, 0)
        
        # Toggle an object
        elif action[agent] == 5:
            toggle(fwd_state, agent_state[agent, 6:10])

        # Done action (not used by default)
        elif action[agent] == 6:
            pass

    return rewards

@nb.njit(cache=True)
def can_overlap(world_obj_state: np.ndarray[int]) -> bool:
    """
    Can an agent overlap with this?

    Parameters
    ----------
    world_obj_state : np.ndarray[int]
        The state of the world object
    """
    if world_obj_state[..., TYPE] == EMPTY:
        return True
    elif world_obj_state[..., TYPE] == GOAL:
        return True
    elif world_obj_state[..., TYPE] == FLOOR:
        return True
    elif world_obj_state[..., TYPE] == LAVA:
        return True
    elif world_obj_state[..., TYPE] == DOOR:
        if world_obj_state[..., STATE] == OPEN:
            return True
    return False

@nb.njit(cache=True)
def can_pickup(world_obj_state: np.ndarray[int]) -> bool:
    """
    Can an agent pick this up?

    Parameters
    ----------
    world_obj_state : np.ndarray[int]
        The state of the world object
    """
    if world_obj_state[..., TYPE] == KEY:
        return True
    elif world_obj_state[..., TYPE] == BALL:
        return True
    elif world_obj_state[..., TYPE] == BOX:
        return True
    return False

@nb.njit(cache=True)
def toggle(
    world_obj_state: np.ndarray[int],
    carrying_obj_state: np.ndarray[int]):
    """
    Toggle the state of this object.

    Parameters
    ----------
    world_obj_state : np.ndarray[int]
        The state of the world object
    """
    # Handle doors
    if world_obj_state[TYPE] == DOOR:
        # Open -> Closed
        if world_obj_state[..., STATE] == OPEN:
            world_obj_state[..., STATE] = CLOSED

        # Closed -> Open
        elif world_obj_state[..., STATE] == CLOSED:
            world_obj_state[..., STATE] = OPEN

        # Locked -> Open
        elif world_obj_state[..., STATE] == LOCKED:
            if carrying_obj_state[TYPE] == KEY:
                # Check if key color matches door color
                if carrying_obj_state[..., COLOR] == world_obj_state[..., COLOR]:
                    world_obj_state[..., STATE] = OPEN

    # Handle boxes
    elif world_obj_state[TYPE] == BOX:
        # Return the contents of the box
        contents = from_mixed_radix_int(world_obj_state[CONTENTS])
        world_obj_state[...] = contents

@nb.njit(cache=True)
def from_mixed_radix_int(n: int) -> np.ndarray[int]:
    """
    Convert a mixed radix integer-encoding to a WorldObjState object.
    """
    x = np.zeros(len(BASES), dtype=np.int_)
    for i in range(len(BASES)):
        x[i] = n % BASES[i]
        n //= BASES[i]

    if x[0] == 0:
        x[0] = EMPTY

    return x
