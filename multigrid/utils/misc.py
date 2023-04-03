import functools
from ..core.constants import DIR_TO_VEC



@functools.cache
def front_pos(agent_x: int, agent_y: int, agent_dir: int):
    """
    Get the position in front of an agent.
    """
    dx, dy = DIR_TO_VEC[agent_dir]
    return (agent_x + dx, agent_y + dy)
