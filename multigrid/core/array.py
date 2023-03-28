"""
Module for performing operations directly with the underlying
array-encoded representations of gridworld objects.

Each world object is represented by an array,
with the values at each index specified as follows:

    * 0: object type
    * 1: object color
    * 2: object state
    * 3: object indicator variable (i.e. this agent is me)
    * 4: contained object type
    * 5: contained object color
    * 6: contained object state
    * 7: contained object indicator variable

This module allows for vectorized operations over world object arrays.
"""
import numpy as np
import numba as nb
from .constants import OBJECT_TO_IDX, STATE_TO_IDX



### Constants

ARRAY_DIM = 8
EMPTY = np.array([
    OBJECT_TO_IDX['empty'], 0, 0, 0,
    OBJECT_TO_IDX['empty'], 0, 0, 0,
])



### Functions

def empty() -> np.ndarray[int]:
    """
    Return the array corresponding to an "empty" grid object.
    """
    return EMPTY.copy()

def contents(array: np.ndarray[int]) -> np.ndarray[int]:
    """
    Return array corresponding to the contents of a grid object.

    Parameters
    ----------
    array : np.ndarray[int] of shape (*, ARRAY_DIM)
        Array corresponding to parent grid object(s)

    Returns
    -------
    contents : np.ndarray[int] of shape (*, ARRAY_DIM)
        Array corresponding to child grid object(s)
    """
    x = np.empty_like(array)
    n = ARRAY_DIM // 2
    x[..., :n] = array[..., n:] # contents are the object
    x[..., n:] = empty()[:n] # contents of the contents are empty
    return x

def can_overlap(array: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Return boolean mask indicating which grid objects can be overlapped.

    Parameters
    ----------
    array : np.ndarray[int] of shape (*, ARRAY_DIM)
        Array corresponding to grid object(s)

    Returns
    -------
    mask : np.ndarray[bool] of shape (*)
        Boolean mask
    """
    mask = (array[..., 0] == OBJECT_TO_IDX['empty'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['goal'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['floor'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['lava'])
    mask |= (
        (array[..., 0] == OBJECT_TO_IDX['door'])
        & (array[..., 2] == STATE_TO_IDX['open'])
    )
    return mask

def can_pickup(array: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Return boolean mask indicating which grid objects can be picked up.

    Parameters
    ----------
    array : np.ndarray[int] of shape (*, ARRAY_DIM)
        Array corresponding to grid object(s)

    Returns
    -------
    mask : np.ndarray[bool] of shape (*)
        Boolean mask
    """
    mask = (array[..., 0] == OBJECT_TO_IDX['key'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['ball'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['box'])
    return mask

def can_contain(array: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Return boolean mask indicating which grid objects can contain other objects.

    Parameters
    ----------
    array : np.ndarray[int] of shape (*, ARRAY_DIM)
        Array corresponding to grid object(s)

    Returns
    -------
    mask : np.ndarray[bool] of shape (*)
        Boolean mask
    """
    return (array[..., 0] == OBJECT_TO_IDX['box'])

def see_behind(array: np.ndarray[int]) -> np.ndarray[bool]:
    """
    Return boolean mask indicating which grid objects can be seen through.

    Parameters
    ----------
    array : np.ndarray[int] of shape (*, ARRAY_DIM)
        Array corresponding to grid object(s)

    Returns
    -------
    mask : np.ndarray[bool] of shape (*)
        Boolean mask
    """
    neg_mask = (array[..., 0] == OBJECT_TO_IDX['wall'])
    neg_mask |= (
        (array[..., 0] == OBJECT_TO_IDX['door'])
        & (array[..., 2] != STATE_TO_IDX['open'])
    )
    return ~neg_mask

def toggle(array: np.ndarray[int], carrying: np.ndarray):
    """
    Toggle the state of a grid object.

    Parameters
    ----------
    array : np.ndarray[int] of shape (*, ARRAY_DIM)
        Array corresponding to grid object(s)
    carrying : np.ndarray[int] of shape (*, ARRAY_DIM)
        Array corresponding to agent's currently carried object(s)
    """
    # Handle doors
    is_door = (array[..., 0] == OBJECT_TO_IDX['door'])
    is_open = is_door & (array[..., 2] == STATE_TO_IDX['open'])
    is_closed = is_door & (array[..., 2] == STATE_TO_IDX['closed'])
    is_locked = is_door & (array[..., 2] == STATE_TO_IDX['locked'])
    can_unlock = (
        is_locked
        & (carrying[..., 0] == OBJECT_TO_IDX['key'])
        & (carrying[..., 1] == array[..., 1])
    )
    array[is_open][..., 2] = STATE_TO_IDX['closed'] # open -> closed
    array[is_closed][..., 2] = STATE_TO_IDX['open'] # closed -> open
    array[can_unlock][..., 2] = STATE_TO_IDX['open'] # locked -> open

    # Handle boxes
    is_box = (array[..., 0] == OBJECT_TO_IDX['box'])
    array[is_box] = contents(array[is_box]) # replace the box by its contents

def get_vis_mask(array: np.ndarray[int], agent_x: int, agent_y: int):
    """
    Return boolean mask indicating which grid locations are visible to the agent.

    Parameters
    ----------
    array : np.ndarray[int] of shape (width, height, ARRAY_DIM)
        Array corresponding to grid object(s)
    agent_pos : tuple[int, int]
        Agent position

    Returns
    -------
    vis_mask : np.ndarray[bool] of shape (width, height)
        Boolean visibility mask
    """
    return _get_vis_mask(
        array, agent_x, agent_y,
        OBJECT_TO_IDX['wall'], OBJECT_TO_IDX['door'], STATE_TO_IDX['open'],
    )

@nb.njit(cache=True)
def rotate_left(array: np.ndarray, k: int):
    k %= 4
    width, height = array.shape[:2]

    if k % 2 == 0:
        rotated_array = np.empty((width, height, array.shape[2]))
    else:
        rotated_array = np.empty((height, width, array.shape[2]))

    for i in range(width):
        for j in range(height):
            if k == 0:
                rotated_array[i, j] = array[i, j]
            elif k == 1:
                rotated_array[j, width - i - 1] = array[i, j]
            elif k == 2:
                rotated_array[width - i - 1, height - j - 1] = array[i, j]
            elif k == 3:
                rotated_array[height - j - 1, i] = array[i, j]

    return rotated_array




@nb.njit(cache=True)
def _set_grid(array, mask, value):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if mask[i, j]:
                array[i, j] = value
    return array





@nb.njit(cache=True)
def _get_see_behind_mask(
    array: np.ndarray[int],
    wall_idx: int, door_idx: int, open_idx: int) -> np.ndarray[bool]:
    width, height = array.shape[:2]
    neg_mask = np.zeros((width, height), dtype=np.bool_)
    for i in range(width):
        for j in range(height):
            if array[i, j, 0] == wall_idx:
                neg_mask[i, j] = True
            elif array[i, j, 0] == door_idx and array[i, j, 2] != open_idx:
                neg_mask[i, j] = True

    return ~neg_mask

@nb.njit(cache=True)
def _get_vis_mask(
    array: np.ndarray, agent_x: int, agent_y: int,
    wall_idx: int, door_idx: int, open_idx: int) -> np.ndarray[bool]:
    width, height = array.shape[:2]
    see_behind_mask = _get_see_behind_mask(array, wall_idx, door_idx, open_idx)
    vis_mask = np.zeros((width, height), dtype=np.bool_)
    vis_mask[agent_x, agent_y] = True

    for j in range(height - 1, -1, -1):
        for i in range(0, width - 1):
            if vis_mask[i, j] and see_behind_mask[i, j]:
                vis_mask[i + 1, j] = True
                if j > 0:
                    vis_mask[i + 1, j - 1] = True
                    vis_mask[i, j - 1] = True

        for i in range(width - 1, 0, -1):
            if vis_mask[i, j] and see_behind_mask[i, j]:
                vis_mask[i - 1, j] = True
                if j > 0:
                    vis_mask[i - 1, j - 1] = True
                    vis_mask[i, j - 1] = True

    return vis_mask
