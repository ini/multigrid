import numpy as np
from typing import Any, Callable, Optional

from .array import ARRAY_DIM, EMPTY
from .constants import OBJECT_TO_IDX, TILE_PIXELS
from .world_object import Wall, WorldObj, world_obj_from_array

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
        assert width >= 3
        assert height >= 3
        self.width: int = width
        self.height: int = height
        self.world_objects: dict[tuple[int, int], WorldObj] = {}
        self.array: np.array[int] = np.empty((width, height, ARRAY_DIM), dtype=int)
        self.array[...] = EMPTY

    @staticmethod
    def from_grid_array(grid_array: np.ndarray[int]) -> 'Grid':
        """
        Create a grid from array representation.
        """
        assert grid_array.ndim == 3
        grid = Grid.__new__(Grid)
        grid.width, grid.height, _ = grid_array.shape
        grid.world_objects = {}
        grid.array = grid_array
        return grid

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, WorldObj):
            return key in self.world_objects.values()
        elif isinstance(key, np.ndarray):
            np.may_share_memory(key, self.array)
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
        return np.array_equal(self.array, other.array)

    def copy(self) -> 'Grid':
        """
        Return a copy of this grid object.
        """
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i: int, j: int, v: Optional[WorldObj]):
        """
        Set a world object at the given coordinates.
        """
        assert (
            0 <= i < self.width
        ), f"column index {j} outside of grid of width {self.width}"
        assert (
            0 <= j < self.height
        ), f"row index {j} outside of grid of height {self.height}"

        # Update world objects
        prev_obj = self.world_objects.pop((i, j), None)
        self.world_objects[i, j] = v
        if prev_obj is not None:
            prev_obj.array = prev_obj.array.copy()

        # Update grid
        if v is None:
            self.array[i, j] = EMPTY
        else:
            self.array[i, j] = v.array
            v.array = self.array[i, j]

    def get(self, i: int, j: int) -> Optional[WorldObj]:
        """
        Get the world object at the given coordinates.
        """
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        assert self.array is not None
        if (i, j) not in self.world_objects:
            self.world_objects[i, j] = world_obj_from_array(self.array[i, j])
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
        self.array[x:x+length, y] = obj_type().array

    def vert_wall(
        self,
        x: int, y: int,
        length: Optional[int] = None,
        obj_type: Callable[[], WorldObj] = Wall):
        """
        Create a vertical wall.
        """
        length = self.height - y if length is None else length
        self.array[x, y:y+length] = obj_type().array

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

    def encode(self, vis_mask: np.ndarray | None = None) -> np.ndarray:
        """
        Produce a compact numpy encoding of the grid.
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        encoding = self.array[..., :3].copy()
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
