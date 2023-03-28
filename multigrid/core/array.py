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
from .constants import OBJECT_TO_IDX, STATE_TO_IDX



### Constants

EMPTY = np.array([
    OBJECT_TO_IDX['empty'], 0, 0, 0,
    OBJECT_TO_IDX['empty'], 0, 0, 0,
])
ARRAY_DIM = len(EMPTY)



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
