"""
Module for fast operations on object grid arrays with numba.
"""
import numba as nb
import numpy as np

from ..core.array import EMPTY
from ..core.constants import OBJECT_TO_IDX, STATE_TO_IDX



### Constants

WALL_IDX = OBJECT_TO_IDX['wall']
DOOR_IDX = OBJECT_TO_IDX['door']
OPEN_IDX = STATE_TO_IDX['open']
ENCODE_DIM = 3
WALL = EMPTY.copy()
WALL[0] = WALL_IDX
WALL_ENCODING = WALL[:ENCODE_DIM]



### Functions

@nb.njit(cache=True)
def gen_obs_grid_encoding(
    grid_array: np.ndarray[int],
    carrying_array: np.ndarray[int],
    agent_dir: int,
    topX: int, topY: int,
    width: int, height: int,
    see_through_walls: bool) -> np.ndarray[int]:
    """
    Generate encoding for the sub-grid observed by an agent.

    Parameters
    ----------
    grid_array : np.ndarray[int] of shape (width, height, dim)
        Grid object array
    carrying_array : np.ndarray[int] of shape (dim,)
        Array representation for object being carried by agent
    agent_dir : int
        Direction the agent is facing
    topX : int
        Top-left x coordinate of the sub-grid
    topY : int
        Top-left y coordinate of the sub-grid
    width : int
        Width of the sub-grid
    height : int
        Height of the sub-grid
    see_through_walls : bool
        Whether the agent can see through walls

    Returns
    -------
    img : np.ndarray[int] of shape (width, height, 3)
        Encoding for observed sub-grid
    """
    # Get dimensions for observation grid
    num_left_rotations = (agent_dir + 1) % 4
    if num_left_rotations % 2 == 0:
        obs_grid = np.empty((width, height, ENCODE_DIM), dtype=np.int_)
    else:
        obs_grid = np.empty((height, width, ENCODE_DIM), dtype=np.int_)

    # Populate observation grid
    for i in range(0, width):
        for j in range(0, height):
            # Absolute coordinates in world grid
            x, y = topX + i, topY + j

            # Rotated relative coordinates for observation grid
            if num_left_rotations == 0:
                i_rot, j_rot = i, j
            elif num_left_rotations == 1:
                i_rot, j_rot = j, width - i - 1
            elif num_left_rotations == 2:
                i_rot, j_rot = width - i - 1, height - j - 1
            elif num_left_rotations == 3:
                i_rot, j_rot = height - j - 1, i

            # Set observation grid
            if 0 <= x < grid_array.shape[0] and 0 <= y < grid_array.shape[1]:
                obs_grid[i_rot, j_rot] = grid_array[x, y, :ENCODE_DIM]
            else:
                obs_grid[i_rot, j_rot] = WALL[:ENCODE_DIM]

    # Generate and apply visibility mask
    if not see_through_walls:
        vis_mask = generate_vis_mask(obs_grid)
        for i in range(vis_mask.shape[0]):
            for j in range(vis_mask.shape[1]):
                if not vis_mask[i, j]:
                    obs_grid[i, j] = 0

    # Make it so the agent sees what it's carrying
    # We do this by placing the carried object at the agent's position
    # in the agent's partially observable view
    obs_grid[width // 2, height - 1] = carrying_array[:ENCODE_DIM]

    return obs_grid

@nb.njit(cache=True)
def gen_obs_grid(
    grid_array: np.ndarray[int],
    carrying_array: np.ndarray[int],
    agent_dir: int,
    topX: int, topY: int,
    width: int, height: int,
    see_through_walls: bool) -> np.ndarray[int]:
    """
    Generate the sub-grid observed by an agent.

    Parameters
    ----------
    grid_array : np.ndarray[int] of shape (width, height, dim)
        Grid object array
    carrying_array : np.ndarray[int] of shape (dim,)
        Array representation for object being carried by agent
    agent_dir : int
        Direction the agent is facing
    topX : int
        Top-left x coordinate of the sub-grid
    topY : int
        Top-left y coordinate of the sub-grid
    width : int
        Width of the sub-grid
    height : int
        Height of the sub-grid
    see_through_walls : bool
        Whether the agent can see through walls

    Returns
    -------
    result : np.ndarray[int] of shape (width, height, dim + 1)
        - result[..., :-1] is the subgrid object array
        - result[..., -1] is the visibility mask
    """
    array_dim = grid_array.shape[2]
    result = np.empty((width, height, array_dim + 1), dtype=np.int_)

    # Get subgrid corresponding to the agent's view
    subgrid = np.empty((width, height, array_dim), dtype=np.int_)
    for i in range(0, width):
        for j in range(0, height):
            x = topX + i
            y = topY + j
            if 0 <= x < grid_array.shape[0] and 0 <= y < grid_array.shape[1]:
                subgrid[i, j] = grid_array[x, y]
            else:
                subgrid[i, j] = WALL

    # Rotate grid to match agent orientation
    k = -(agent_dir + 1) % 4
    result[:, :, :-1] = np.rot90(subgrid, k=k)

    # Generate and apply visibility mask
    if see_through_walls:
        result[:, :, -1] = np.ones(grid_array.shape[:2], dtype=np.bool_)
    else:
        result[:, :, -1] = generate_and_apply_vis_mask(result[:, :, :-1], EMPTY)

    # Make it so the agent sees what it's carrying
    # We do this by placing the carried object at the agent's position
    # in the agent's partially observable view
    result[width // 2, height - 1, :-1] = carrying_array

    return result

@nb.njit(cache=True)
def get_see_behind_mask(grid_array: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Return boolean mask indicating which grid locations can be seen through.

    Parameters
    ----------
    grid_array : np.ndarray[int] of shape (width, height, dim)
        Grid object array

    Returns
    -------
    see_behind_mask : np.ndarray[bool] of shape (width, height)
        Boolean transparency mask
    """
    width, height = grid_array.shape[:2]
    neg_mask = np.zeros((width, height), dtype=np.bool_)
    for i in range(width):
        for j in range(height):
            if grid_array[i, j, 0] == WALL_IDX:
                neg_mask[i, j] = True
            elif grid_array[i, j, 0] == DOOR_IDX and grid_array[i, j, 2] != OPEN_IDX:
                neg_mask[i, j] = True

    return ~neg_mask

@nb.njit(cache=True)
def generate_vis_mask(grid_array: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Generate a boolean mask indicating which grid locations are visible to the agent.

    Parameters
    ----------
    grid_array : np.ndarray[int] of shape (width, height, dim)
        Grid object array

    Returns
    -------
    vis_mask : np.ndarray[bool] of shape (width, height)
        Boolean visibility mask
    """
    width, height = grid_array.shape[:2]
    see_behind_mask = get_see_behind_mask(grid_array)
    vis_mask = np.zeros((width, height), dtype=np.bool_)
    vis_mask[width // 2, height - 1] = True # agent relative position

    for j in range(height - 1, -1, -1):
        # Forward pass
        for i in range(0, width - 1):
            if vis_mask[i, j] and see_behind_mask[i, j]:
                vis_mask[i + 1, j] = True
                if j > 0:
                    vis_mask[i + 1, j - 1] = True
                    vis_mask[i, j - 1] = True

        # Backward pass
        for i in range(width - 1, 0, -1):
            if vis_mask[i, j] and see_behind_mask[i, j]:
                vis_mask[i - 1, j] = True
                if j > 0:
                    vis_mask[i - 1, j - 1] = True
                    vis_mask[i, j - 1] = True

    return vis_mask

@nb.njit(cache=True)
def generate_and_apply_vis_mask(
    grid_array: np.ndarray[int],
    mask_object_array: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Generate a boolean mask indicating which grid locations are visible to the agent,
    and apply it to the given array.

    Parameters
    ----------
    grid_array : np.ndarray[int] of shape (width, height, dim)
        Grid object array
    mask_object_array : np.ndarray[int] of shape (dim,)
        Object array to replace masked grid locations

    Returns
    -------
    vis_mask : np.ndarray[bool] of shape (width, height)
        Boolean visibility mask
    """
    width, height = grid_array.shape[:2]
    see_behind_mask = get_see_behind_mask(grid_array)
    vis_mask = np.zeros((width, height), dtype=np.bool_)
    vis_mask[width // 2, height - 1] = True # agent relative position

    for j in range(height - 1, -1, -1):
        # Forward pass
        for i in range(0, width - 1):
            if vis_mask[i, j] and see_behind_mask[i, j]:
                vis_mask[i + 1, j] = True
                if j > 0:
                    vis_mask[i + 1, j - 1] = True
                    vis_mask[i, j - 1] = True

        # Backward pass
        for i in range(width - 1, 0, -1):
            if vis_mask[i, j] and see_behind_mask[i, j]:
                vis_mask[i - 1, j] = True
                if j > 0:
                    vis_mask[i - 1, j - 1] = True
                    vis_mask[i, j - 1] = True

        # Apply mask
        for i in range(width):
            if not vis_mask[i, j]:
                grid_array[i, j] = mask_object_array

    return vis_mask
