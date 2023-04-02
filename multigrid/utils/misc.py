import functools
from ..core.constants import DIR_TO_VEC, OBJECT_TO_IDX, IDX_TO_OBJECT, IDX_TO_STATE



@functools.cache
def can_overlap(type_idx, color_idx, state_idx, *args) -> bool:
    """
    Can an agent overlap with this?
    """
    if IDX_TO_OBJECT[type_idx] in {'empty', 'goal', 'floor', 'lava'}:
        return True
    elif IDX_TO_OBJECT[type_idx] == 'door' and IDX_TO_STATE[state_idx] == 'open':
        return True

    return False

@functools.cache
def can_pickup(type_idx, color_idx, state_idx, *args) -> bool:
    """
    Can an agent pick this up?
    """
    return IDX_TO_OBJECT[type_idx] in {'key', 'ball', 'box'}

@functools.cache
def front_pos(agent_x: int, agent_y: int, agent_dir: int):
    """
    Get the position in front of an agent.
    """
    dx, dy = DIR_TO_VEC[agent_dir]
    return (agent_x + dx, agent_y + dy)
