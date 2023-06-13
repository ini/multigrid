from __future__ import annotations

import aenum as enum
import functools
import numpy as np

from numpy.typing import ArrayLike, NDArray as ndarray
from typing import Any



### Helper Functions

@functools.cache
def _enum_array(enum_cls: enum.EnumMeta):
    """
    Return an array of all values of the given enum.

    Parameters
    ----------
    enum_cls : enum.EnumMeta
        Enum class
    """
    return np.array([item.value for item in enum_cls])

@functools.cache
def _enum_index(enum_item: enum.Enum):
    """
    Return the index of the given enum item.

    Parameters
    ----------
    enum_item : enum.Enum
        Enum item
    """
    return list(enum_item.__class__).index(enum_item)



### Enumeration

class IndexedEnum(enum.Enum):
    """
    Enum where each member has a corresponding integer index.
    """

    def __int__(self):
        return self.to_index()

    @classmethod
    def add_item(cls, name: str, value: Any):
        """
        Add a new item to the enumeration.

        Parameters
        ----------
        name : str
            Name of the new enum item
        value : Any
            Value of the new enum item
        """
        enum.extend_enum(cls, name, value)
        _enum_array.cache_clear()
        _enum_index.cache_clear()

    @classmethod
    def from_index(cls, index: int | ArrayLike[int]) -> enum.Enum | ndarray:
        """
        Return the enum item corresponding to the given index.
        Also supports vector inputs.

        Parameters
        ----------
        index : int or ArrayLike[int]
            Enum index (or array of indices)

        Returns
        -------
        enum.Enum or ndarray
            Enum item (or array of enum item values)
        """
        out = _enum_array(cls)[index]
        return cls(out) if out.ndim == 0 else out

    def to_index(self) -> int:
        """
        Return the integer index of this enum item.
        """
        return _enum_index(self)
