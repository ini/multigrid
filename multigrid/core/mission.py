from __future__ import annotations

import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Any, Callable, Sequence



def check_if_no_duplicate(duplicate_list: list) -> bool:
    """
    Check if given list contains any duplicates.
    """
    return len(set(duplicate_list)) == len(duplicate_list)



class MissionSpace(spaces.Space[str]):
    """
    A space representing a mission for MultiGridEnv environments.

    The space allows generating random mission strings constructed with
    an input placeholder list.

    Attributes
    ----------
    mission_to_index : dict[str, np.ndarray]
        Dictionary mapping mission strings to their corresponding index
        in the multi-discrete placeholder space

    Examples
    --------
    >>> observation_space = MissionSpace(
    ...     mission_func=lambda color: f"Get the {color} ball.",
    ...     ordered_placeholders=[["green", "blue"]])
    >>> observation_space.seed(123)
    >>> observation_space.sample()
    'Get the green ball.'

    >>> observation_space = MissionSpace(
    ...     mission_func=lambda : "Get the ball.",
    ...     ordered_placeholders=None)
    >>> observation_space.sample()
    'Get the ball.'
    """

    def __init__(
        self,
        mission_func: Callable[..., str],
        ordered_placeholders: list[list[str]] | None = None,
        seed: int | seeding.RandomNumberGenerator | None = None):
        """
        Parameters
        ----------
        mission_func : Callable(*args) -> str
            Function that generates a mission string from random placeholders
        ordered_placeholders : list[list[str]], optional
            List of lists of placeholders ordered in placing order in `mission_func()`
        seed : int | seeding.RandomNumberGenerator, optional
            Seed for sampling from the space
        """
        # Check that the ordered placeholders and mission function are well defined.
        if ordered_placeholders:
            assert (
                len(ordered_placeholders) == mission_func.__code__.co_argcount
            ), (
                f"The number of placeholders {len(ordered_placeholders)} "
                "is different from the number of parameters in the mission function"
                f"{mission_func.__code__.co_argcount}."
            )
            for placeholder_list in ordered_placeholders:
                assert check_if_no_duplicate(
                    placeholder_list
                ), "Make sure that the placeholders don't have any duplicate values."
        else:
            assert (
                mission_func.__code__.co_argcount == 0
            ), (
                f"If the ordered placeholders are {ordered_placeholders}, "
                "the mission function shouldn't have any parameters."
            )

        self.ordered_placeholders = ordered_placeholders
        self.mission_func = mission_func

        if not self.ordered_placeholders:
            self.mission_to_index = {self.mission_func(): 0}
        else:
            self.mission_to_index = {
                self.get(idx): np.array(idx, dtype=int)
                for idx in np.ndindex(
                    tuple(len(var_list) for var_list in self.ordered_placeholders))
            }

        super().__init__(dtype=str, seed=seed)

        # Check that mission_func returns a string
        sampled_mission = self.sample()
        assert isinstance(
            sampled_mission, str
        ), f"mission_func must return type str not {type(sampled_mission)}"

    def placeholder_space(self) -> spaces.MultiDiscrete | spaces.Discrete:
        """
        Return the multi-discrete placeholder space.

        Example
        -------
        >>> mission_space = MissionSpace(
        ...     mission_func=lambda color, obj: f"Get the {color} {obj}.",
        ...     ordered_placeholders=[['red', 'green', 'blue'], ['ball', 'box']])
        >>> mission_space.placeholder_space()
        MultiDiscrete([3, 2])
        """
        if not self.ordered_placeholders:
            return spaces.Discrete(1)

        return spaces.MultiDiscrete(
            tuple(len(var_list) for var_list in self.ordered_placeholders))

    def get(self, idx: Sequence[int]) -> str:
        """
        Get the mission string corresponding to the given index.
        """
        placeholders = [self.ordered_placeholders[i][idx[i]] for i in range(len(idx))]
        return self.mission_func(*placeholders)

    def sample(self) -> str:
        """
        Sample a random mission string.
        """
        if not self.ordered_placeholders:
            return self.mission_func()

        idx = self.np_random.integers(0, self.placeholder_space().shape)
        return self.get(idx)

    def contains(self, x: Any) -> bool:
        """
        Return boolean specifying if x is a valid member of this space.
        """
        return x in self.mission_to_index

    def __repr__(self) -> str:
        """
        Gives a string representation of this space.
        """
        return f"MissionSpace({self.mission_func}, {self.ordered_placeholders})"

    def __eq__(self, other) -> bool:
        """
        Check whether `other` is equivalent to this instance.
        """
        if isinstance(other, MissionSpace):

            # Check that place holder lists are the same
            if self.ordered_placeholders is not None:
                # Check length
                if (
                    len(self.ordered_placeholders) == len(other.ordered_placeholders)
                ) and (
                    all(
                        set(i) == set(j)
                        for i, j in zip(
                            self.ordered_placeholders, other.ordered_placeholders
                        )
                    )
                ):
                    # Check mission string is the same with dummy space placeholders
                    test_placeholders = [""] * len(self.ordered_placeholders)
                    mission = self.mission_func(*test_placeholders)
                    other_mission = other.mission_func(*test_placeholders)
                    return mission == other_mission
            else:

                # Check that other is also None
                if other.ordered_placeholders is None:

                    # Check mission string is the same
                    mission = self.mission_func()
                    other_mission = other.mission_func()
                    return mission == other_mission

        # If none of the statements above return then False
        return False
