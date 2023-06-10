from __future__ import annotations

import numpy as np
from gymnasium import spaces
from typing import Any, Callable, Iterable, Sequence



class Mission(np.ndarray):
    """
    Class representing an agent mission.
    """

    def __new__(cls, string: str, index: Iterable[int] | None = None):
        """
        Parameters
        ----------
        string : str
            Mission string
        index : Iterable[int]
            Index of mission string in :class:`MissionSpace`
        """
        mission = np.array(0 if index is None else index)
        mission = mission.view(cls)
        mission.string = string
        return mission.view(cls)

    def __array_finalize__(self, mission):
        if mission is None: return
        self.string = getattr(mission, 'string', None)

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.string}")'

    def __eq__(self, value: object) -> bool:
        return self.string == str(value)

    def __hash__(self) -> int:
        return hash(self.string)


class MissionSpace(spaces.MultiDiscrete):
    """
    Class representing a space over agent missions.

    Examples
    --------
    >>> observation_space = MissionSpace(
    ...     mission_func=lambda color: f"Get the {color} ball.",
    ...     ordered_placeholders=[["green", "blue"]])
    >>> observation_space.seed(123)
    >>> observation_space.sample()
    Mission("Get the blue ball.")

    >>> observation_space = MissionSpace.from_string("Get the ball.")
    >>> observation_space.sample()
    Mission("Get the ball.")
    """

    def __init__(
        self,
        mission_func: Callable[..., str],
        ordered_placeholders: Sequence[Sequence[str]] = [],
        seed : int | np.random.Generator | None = None):
        """
        Parameters
        ----------
        mission_func : Callable(*args) -> str
            Deterministic function that generates a mission string
        ordered_placeholders : Sequence[Sequence[str]]
            Sequence of argument groups, ordered by placing order in ``mission_func()``
        seed : int or np.random.Generator or None
            Seed for random sampling from the space
        """
        self.mission_func = mission_func
        self.arg_groups = ordered_placeholders
        nvec = tuple(len(group) for group in self.arg_groups)
        super().__init__(nvec=nvec if nvec else (1,))

    def __repr__(self) -> str:
        """
        Get a string representation of this space.
        """
        if self.arg_groups:
            return f'MissionSpace({self.mission_func.__name__}, {self.arg_groups})'
        return f"MissionSpace('{self.mission_func()}')"

    def get(self, idx: Iterable[int]) -> Mission:
        """
        Get the mission string corresponding to the given index.

        Parameters
        ----------
        idx : Iterable[int]
            Index of desired argument in each argument group
        """
        if self.arg_groups:
            args = (self.arg_groups[axis][index] for axis, index in enumerate(idx))
            return Mission(string=self.mission_func(*args), index=idx)
        return Mission(string=self.mission_func())

    def sample(self) -> Mission:
        """
        Sample a random mission string.
        """
        idx = super().sample()
        return self.get(idx)

    def contains(self, x: Any) -> bool:
        """
        Check if an item is a valid member of this mission space.

        Parameters
        ----------
        x : Any
            Item to check
        """
        for idx in np.ndindex(tuple(self.nvec)):
            if self.get(idx) == x:
                return True
        return False

    @staticmethod
    def from_string(string: str) -> MissionSpace:
        """
        Create a mission space containing a single mission string.

        Parameters
        ----------
        string : str
            Mission string
        """
        return MissionSpace(mission_func=lambda: string)
