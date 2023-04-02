import numpy as np

from functools import cached_property
from typing import Any, Callable, Optional, Union

from .agent import Agent
from .constants import OBJECT_TO_IDX, TILE_PIXELS
from .world_object import Wall, WorldObj

from ..utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
)



class Grid:
    """
    Represent a grid and operations on it.
    """

    # Static cache of pre-renderer tiles
    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(self, width: int, height: int):
        """
        Parameters
        ----------
        width : int
            Width of the grid
        height : int
            Height of the grid
        """
        assert width >= 3
        assert height >= 3
        self.world_objects: dict[tuple[int, int], WorldObj] = {}
        self.state: np.ndarray[int] = np.zeros(
            (width, height, WorldObj.dim), dtype=int)
        self.state[...] = WorldObj.empty()

        # Arrays for updating world objects in the grid
        # These can be passed by reference and updated within numba functions
        # (i.e. `handle_actions()`)
        self.needs_update = np.zeros(1, dtype=bool)
        self.locations_to_update = -np.ones((1, 2), dtype=int) # (num_locations, 2)
        self.needs_remove = np.zeros(1, dtype=bool)
        self.locations_to_remove = -np.ones((1, 2), dtype=int) # (num_locations, 2)

    @classmethod
    def from_grid_state(cls, grid_state: np.ndarray) -> 'Grid':
        """
        Create a grid from a grid state array.
        """
        assert grid_state.ndim == 3
        grid = cls.__new__(cls)
        grid.world_objects = {}
        grid.state = grid_state
        return grid

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, WorldObj):
            return key in self.world_objects.values()
        elif isinstance(key, np.ndarray):
            np.may_share_memory(key, self.state)
        elif isinstance(key, tuple):
            for i in range(self.width):
                for j in range(self.height):
                    e = self.get(i, j)
                    if e is None:
                        continue
                    if (e.color, e.type) == key:
                        return True
                    if key[0] is None and key[1] == e.type:
                        return True
        return False

    def __eq__(self, other: 'Grid') -> bool:
        return np.array_equal(self.state, other.state)

    @cached_property
    def width(self) -> int:
        """
        Width of the grid.
        """
        return self.state.shape[0]

    @cached_property
    def height(self) -> int:
        """
        Height of the grid.
        """
        return self.state.shape[1]

    @property
    def grid(self) -> list[WorldObj]:
        """
        Return a list of all world objects in the grid.
        """
        return [self.get(i, j) for i in range(self.width) for j in range(self.height)]

    def copy(self) -> 'Grid':
        """
        Return a copy of this grid object.
        """
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i: int, j: int, v: Union[WorldObj, Agent, None]):
        """
        Set a world object at the given coordinates.
        """
        assert 0 <= i < self.width
        assert 0 <= j < self.height

        # Update world objects
        self.world_objects[i, j] = v

        # Update grid
        if isinstance(v, WorldObj):
            self.state[i, j] = v
        elif isinstance(v, Agent):
            self.state[i, j] = v.world_obj()
        elif v is None:
            self.state[i, j] = WorldObj.empty()
        else:
            raise TypeError(f"cannot set grid value to {type(v)}")

    def get(self, i: int, j: int) -> Optional[WorldObj]:
        """
        Get the world object at the given coordinates.
        """
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        if (i, j) not in self.world_objects:
            self.world_objects[i, j] = WorldObj.from_array(self.state[i, j])

        return self.world_objects[i, j]

    def horz_wall(
        self,
        x: int, y: int,
        length: Optional[int] = None,
        obj_type: Callable[[], WorldObj] = Wall):
        """
        Create a horizontal wall.
        """
        length = self.width - x if length is None else length
        self.state[x:x+length, y] = obj_type()

    def vert_wall(
        self,
        x: int, y: int,
        length: Optional[int] = None,
        obj_type: Callable[[], WorldObj] = Wall):
        """
        Create a vertical wall.
        """
        length = self.height - y if length is None else length
        self.state[x, y:y+length] = obj_type()

    def wall_rect(self, x: int, y: int, w: int, h: int):
        """
        Create a walled rectangle.
        """
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    @classmethod
    def render_tile(
        cls,
        obj: Optional[WorldObj] = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3) -> np.ndarray:
        """
        Render a tile and cache the result.
        """
        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size: int,
        highlight_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render this grid at a given scale.

        Parameters
        ----------
        tile_size: int
            Tile size (in pixels)
        highlight_mask: np.ndarray
            Boolean mask indicating which grid locations to highlight
        """
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                assert highlight_mask is not None
                tile_img = Grid.render_tile(
                    cell,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Produce a compact numpy encoding of the grid.
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        encoding = self.state[..., :WorldObj.encode_dim].copy()
        encoding[~vis_mask] = 0
        return encoding

    @staticmethod
    def decode(array: np.ndarray) -> tuple['Grid', np.ndarray]:
        """
        Decode an array grid encoding back into a grid.
        """
        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=bool)
        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = type_idx != OBJECT_TO_IDX['unseen']

        return grid, vis_mask

    def update_world_objects(self):
        """
        Update WorldObj instances from the grid state.
        """
        self.needs_update[...] = False
        for i, j in self.locations_to_update:
            if (i, j) in self.world_objects:
                self.world_objects[i, j][...] = self.state[i, j]

    def remove_world_objects(self):
        """
        Remove WorldObj instances from the grid.
        """
        self.needs_remove[...] = False
        for i, j in self.locations_to_remove:
            self.world_objects.pop((i, j), None)
