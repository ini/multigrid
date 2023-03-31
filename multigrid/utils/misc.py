import functools
import numpy as np

from ..core.constants import DIR_TO_VEC, OBJECT_TO_IDX, IDX_TO_OBJECT, IDX_TO_STATE



@functools.cache
def can_overlap(type, color, state, *args) -> bool:
    """
    Can an agent overlap with this?
    """
    if IDX_TO_OBJECT[type] in {'empty', 'goal', 'floor', 'lava'}:
        return True
    elif IDX_TO_OBJECT[type] == 'door' and IDX_TO_STATE[state] == 'open':
        return True

    return False

@functools.cache
def can_pickup(type, color, state, *args) -> bool:
    """
    Can an agent pick this up?
    """
    return IDX_TO_OBJECT[type] in {'key', 'ball', 'box'}






@functools.cache
def from_mixed_radix_int(n: int, bases: tuple):
    """
    Convert a mixed radix integer-encoding to a WorldObjState object.
    """
    x = []
    for i in range(len(bases)):
        x.append(n % bases[i])
        n //= bases[i]

    if x[0] == 0:
        x[0] = OBJECT_TO_IDX['empty']

    return x

@functools.cache
def front_pos(agent_x, agent_y, agent_dir: int):
    """
    Get the position in front of an agent.
    """
    dx, dy = DIR_TO_VEC[agent_dir]
    return (agent_x + dx, agent_y + dy)
