import numpy as np
from typing import Iterable, TypeVar
from ..core.constants import Color

T = TypeVar('T')



class RandomMixin:
    """
    Mixin class for random number generation.
    """

    def __init__(self, random_generator: np.random.Generator):
        """
        Parameters
        ----------
        random_generator : np.random.Generator
            Random number generator
        """
        self.__np_random = random_generator

    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in range [low, high).

        :meta public:
        """
        return self.__np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in range [low, high).

        :meta public:
        """
        return self.__np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value.

        :meta public:
        """
        return self.__np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list.

        :meta public:
        """
        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list.

        :meta public:
        """
        lst = list(iterable)
        assert num_elems <= len(lst)

        out: list[T] = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_perm(self, iterable: Iterable[T]) -> list[T]:
        """
        Randomly permute a list.

        :meta public:
        """
        lst = list(iterable)
        self.__np_random.shuffle(lst)
        return lst

    def _rand_color(self) -> Color:
        """
        Generate a random color.

        :meta public:
        """
        return self._rand_elem(Color)

    def _rand_pos(
        self, x_low: int, x_high: int, y_low: int, y_high: int) -> tuple[int, int]:
        """
        Generate a random (x, y) position tuple.

        :meta public:
        """
        return (
            self.__np_random.integers(x_low, x_high),
            self.__np_random.integers(y_low, y_high),
        )
