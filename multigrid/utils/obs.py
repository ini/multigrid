import numba as nb
import numpy as np

from ..core.world_object import WorldObj, Wall



# Constants
ENCODE_DIM = WorldObj.encode_dim
WALL_ENCODING = Wall().encode()

# WorldObj Functions
see_behind = WorldObj.see_behind



### Observation Functions

@nb.njit(cache=True)
def gen_obs_grid_encoding(
    grid_state: np.ndarray[int],
    agent_state: np.ndarray[int],
    agent_view_size: int,
    see_through_walls: bool) -> np.ndarray[int]:
    """
    Generate encoding for the sub-grid observed by an agent (including visibility mask).

    Parameters
    ----------
    grid_state : np.ndarray[int] of shape (width, height, grid_state_dim)
        Array representation for each grid object
    agent_state : np.ndarray[int] of shape (num_agents, agent_state_dim)
        Array representation for each agent
    agent_view_size : int
        Width and height of observation sub-grids
    see_through_walls : bool
        Whether the agent can see through walls

    Returns
    -------
    img : np.ndarray[int] of shape (num_agents, view_size, view_size, encode_dim)
        Encoding of observed sub-grid for each agent
    """
    obs_grid = gen_obs_grid(grid_state, agent_state, agent_view_size)

    # Generate and apply visibility masks
    vis_mask = get_vis_mask(obs_grid)
    num_agents = len(agent_state)
    for agent in range(num_agents):
        if not see_through_walls:
            for i in range(agent_view_size):
                for j in range(agent_view_size):
                    if not vis_mask[agent, i, j]:
                        obs_grid[agent, i, j] = 0

    return obs_grid

@nb.njit(cache=True)
def gen_obs_grid_vis_mask(
    grid_state: np.ndarray[int],
    agent_state: np.ndarray[int],
    agent_view_size: int) -> np.ndarray[int]:
    """
    Generate visibility mask for the sub-grid observed by an agent.

    Parameters
    ----------
    grid_state : np.ndarray[int] of shape (width, height, grid_state_dim)
        Array representation for each grid object
    agent_state : np.ndarray[int] of shape (num_agents, agent_state_dim)
        Array representation for each agent
    agent_view_size : int
        Width and height of observation sub-grids

    Returns
    -------
    mask : np.ndarray[int] of shape (num_agents, view_size, view_size)
        Encoding of observed sub-grid for each agent
    """
    obs_grid = gen_obs_grid(grid_state, agent_state, agent_view_size)
    return get_vis_mask(obs_grid)


@nb.njit(cache=True)
def gen_obs_grid(
    grid_state: np.ndarray[int],
    agent_state: np.ndarray[int],
    agent_view_size: int) -> np.ndarray[int]:
    """
    Generate the sub-grid observed by each agent (WITHOUT visibility mask).

    Parameters
    ----------
    grid_state : np.ndarray[int] of shape (width, height, grid_state_dim)
        Array representation for each grid object
    agent_state : np.ndarray[int] of shape (num_agents, agent_state_dim)
        Array representation for each agent
    agent_view_size : int
        Width and height of observation sub-grids

    Returns
    -------
    obs_grid : np.ndarray[int] of shape (num_agents, width, height, encode_dim)
        Observed sub-grid for each agent
    """
    num_agents = len(agent_state)
    obs_width, obs_height = agent_view_size, agent_view_size

    # Process agent states
    agent_grid_encoding = agent_state[..., :3]
    agent_dir = agent_state[..., 2]
    agent_pos = agent_state[..., 3:5]
    agent_carrying_encoding = agent_state[..., 6:6+ENCODE_DIM]

    # Get grid encoding
    if num_agents > 1:
        grid_encoding = np.empty(
            (grid_state.shape[0], grid_state.shape[1], ENCODE_DIM), dtype=np.int_)
        grid_encoding[...] = grid_state[..., :ENCODE_DIM]
        # Insert agent encodings
        for agent in range(num_agents):
            i, j = agent_pos[agent]
            grid_encoding[i, j, :ENCODE_DIM] = agent_grid_encoding[agent]
    else:
        grid_encoding = grid_state[..., :ENCODE_DIM]

    # Get top left corner of observation grids
    top_left = get_view_exts(agent_dir, agent_pos, agent_view_size)
    topX, topY = top_left[:, 0], top_left[:, 1]

    # Populate observation grids
    num_left_rotations = (agent_dir + 1) % 4
    obs_grid = np.empty((num_agents, obs_width, obs_height, ENCODE_DIM), dtype=np.int_)
    for agent in range(num_agents):
        for i in range(0, obs_width):
            for j in range(0, obs_height):
                # Absolute coordinates in world grid
                x, y = topX[agent] + i, topY[agent] + j

                # Rotated relative coordinates for observation grid
                if num_left_rotations[agent] == 0:
                    i_rot, j_rot = i, j
                elif num_left_rotations[agent] == 1:
                    i_rot, j_rot = j, obs_width - i - 1
                elif num_left_rotations[agent] == 2:
                    i_rot, j_rot = obs_width - i - 1, obs_height - j - 1
                elif num_left_rotations[agent] == 3:
                    i_rot, j_rot = obs_height - j - 1, i

                # Set observation grid
                if 0 <= x < grid_encoding.shape[0] and 0 <= y < grid_encoding.shape[1]:
                    obs_grid[agent, i_rot, j_rot] = grid_encoding[x, y]
                else:
                    obs_grid[agent, i_rot, j_rot] = WALL_ENCODING

    # Make it so each agent sees what it's carrying
    # We do this by placing the carried object at the agent position
    # in each agent's partially observable view
    obs_grid[:, obs_width // 2, obs_height - 1] = agent_carrying_encoding

    return obs_grid

@nb.njit(cache=True)
def get_see_behind_mask(grid_array: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Return boolean mask indicating which grid locations can be seen through.

    Parameters
    ----------
    grid_array : np.ndarray[int] of shape (num_agents, width, height, dim)
        Grid object array for each agent

    Returns
    -------
    see_behind_mask : np.ndarray[bool] of shape (width, height)
        Boolean transparency mask
    """
    num_agents, width, height = grid_array.shape[:3]
    see_behind_mask = np.zeros((num_agents, width, height), dtype=np.bool_)
    for agent in range(num_agents):
        for i in range(width):
            for j in range(height):
                see_behind_mask[agent, i, j] = see_behind(grid_array[agent, i, j])

    return see_behind_mask

@nb.njit(cache=True)
def get_vis_mask(obs_grid: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Generate a boolean mask indicating which grid locations are visible to the agent.

    Parameters
    ----------
    obs_grid : np.ndarray[int] of shape (num_agents, width, height, dim)
        Grid object array for each agent observation

    Returns
    -------
    vis_mask : np.ndarray[bool] of shape (num_agents, width, height)
        Boolean visibility mask for each agent
    """
    num_agents, width, height = obs_grid.shape[:3]
    see_behind_mask = get_see_behind_mask(obs_grid)
    vis_mask = np.zeros((num_agents, width, height), dtype=np.bool_)
    vis_mask[:, width // 2, height - 1] = True # agent relative position

    for agent in range(num_agents):
        for j in range(height - 1, -1, -1):
            # Forward pass
            for i in range(0, width - 1):
                if vis_mask[agent, i, j] and see_behind_mask[agent, i, j]:
                    vis_mask[agent, i + 1, j] = True
                    if j > 0:
                        vis_mask[agent, i + 1, j - 1] = True
                        vis_mask[agent, i, j - 1] = True

            # Backward pass
            for i in range(width - 1, 0, -1):
                if vis_mask[agent, i, j] and see_behind_mask[agent, i, j]:
                    vis_mask[agent, i - 1, j] = True
                    if j > 0:
                        vis_mask[agent, i - 1, j - 1] = True
                        vis_mask[agent, i, j - 1] = True

    return vis_mask

@nb.njit(cache=True)
def get_view_exts(
    agent_dir: np.ndarray[int],
    agent_pos: np.ndarray[int],
    agent_view_size: int):
    """
    Get the extents of the square set of tiles visible to agents.

    Parameters
    ----------
    agent_dir : np.ndarray[int] of shape (num_agents,)
        Direction of each agent
    agent_pos : np.ndarray[int] of shape (num_agents, 2)
        The (x, y) position of each agent
    agent_view_size : int
        Width and height of agent view

    Returns
    -------
    top_left : np.ndarray[int] of shape (num_agents, 2)
        The (x, y) coordinates of the top-left corner of each observable view
    """
    agent_x, agent_y = agent_pos[:, 0], agent_pos[:, 1]
    top_left = np.zeros((agent_dir.shape[0], 2), dtype=np.int_)

    # Facing right
    top_left[agent_dir == 0, 0] = agent_x[agent_dir == 0]
    top_left[agent_dir == 0, 1] = agent_y[agent_dir == 0] - agent_view_size // 2

    # Facing down
    top_left[agent_dir == 1, 0] = agent_x[agent_dir == 1] - agent_view_size // 2
    top_left[agent_dir == 1, 1] = agent_y[agent_dir == 1]

    # Facing left
    top_left[agent_dir == 2, 0] = agent_x[agent_dir == 2] - agent_view_size + 1
    top_left[agent_dir == 2, 1] = agent_y[agent_dir == 2] - agent_view_size // 2

    # Facing up
    top_left[agent_dir == 3, 0] = agent_x[agent_dir == 3] - agent_view_size // 2
    top_left[agent_dir == 3, 1] = agent_y[agent_dir == 3] - agent_view_size + 1

    return top_left
