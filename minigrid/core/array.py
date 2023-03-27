"""
Module for underlying array-encoded representation of gridworld objects.

Each world object is represented by an array,
with the values at each index specified as follows:

    * 0: object type
    * 1: object color
    * 2: object state
    * 3: object ID
    * 4: contained object type
    * 5: contained object color
    * 6: contained object state
    * 7: contained object ID

This module allows for vectorized operations over world object arrays.
The underlying behavioral logic for each object type is contained here.
"""
import numpy as np
from .constants import OBJECT_TO_IDX, STATE_TO_IDX



ARRAY_DIM = 8
EMPTY = np.array([
    OBJECT_TO_IDX['empty'], 0, 0, 0,
    OBJECT_TO_IDX['empty'], 0, 0, 0,
])

def empty() -> np.ndarray[int]:
    return EMPTY.copy()

def contents(array: np.ndarray[int]) -> np.ndarray[int]:
    x = np.zeros_like(array)
    n = ARRAY_DIM // 2
    x[..., :n] = array[..., n:]
    x[..., n:] = empty()[:n]
    return x

def can_overlap(array: np.ndarray[int]) -> np.ndarray[bool]:
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
    mask = (array[..., 0] == OBJECT_TO_IDX['key'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['ball'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['box'])
    return mask

def can_contain(array: np.ndarray[int]) -> np.ndarray[bool]:
    return (array[..., 0] == OBJECT_TO_IDX['box'])

def see_behind(array: np.ndarray[int]) -> np.ndarray[bool]:
    neg_mask = (array[..., 0] == OBJECT_TO_IDX['wall'])
    neg_mask |= (
        (array[..., 0] == OBJECT_TO_IDX['door'])
        & (array[..., 2] != STATE_TO_IDX['open'])
    )
    return ~neg_mask

def toggle(array: np.ndarray[int], carrying: np.ndarray):
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

def get_vis_mask(array: np.ndarray[int], agent_pos: tuple[int, int]):
    width, height = array.shape[:2]
    see_behind_mask = see_behind(array)
    vis_mask = np.zeros((width, height), dtype=bool)
    vis_mask[agent_pos] = True

    for j in reversed(range(0, height)):
        for i in range(0, width - 1):
            if not vis_mask[i, j] or see_behind_mask[i, j]:
                continue

            vis_mask[i + 1, j] = True
            if j > 0:
                vis_mask[i + 1, j - 1] = True
                vis_mask[i, j - 1] = True

        for i in reversed(range(1, width)):
            if not vis_mask[i, j] or see_behind_mask[i, j]:
                continue

            vis_mask[i - 1, j] = True
            if j > 0:
                vis_mask[i - 1, j - 1] = True
                vis_mask[i, j - 1] = True

        return vis_mask
