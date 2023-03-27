from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS
from minigrid.core.world_object import Wall, WorldObj
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)
from .array import (
    ARRAY_DIM,
    empty,
    get_vis_mask,
)
from .world_object import world_obj_from_array



def clip(values, low, high):
    return (max(low, min(high, v)) for v in values)



class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(self, width: int, height: int):
        assert width >= 3
        assert height >= 3
        self.width: int = width
        self.height: int = height
        self.grid: np.array[int] = np.zeros((width, height, ARRAY_DIM), dtype=int)
        self.grid[...] = empty()

    def set(self, i: int, j: int, v: WorldObj | None):
        assert (
            0 <= i < self.width
        ), f"column index {j} outside of grid of width {self.width}"
        assert (
            0 <= j < self.height
        ), f"row index {j} outside of grid of height {self.height}"
        if v is None:
            self.grid[i, j] = empty()
        else:
            self.grid[i, j] = v.array

    def get(self, i: int, j: int) -> WorldObj | None:
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        assert self.grid is not None
        return world_obj_from_array(self.grid[i, j])

    def horz_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        length = self.width - x if length is None else length
        self.grid[x:x+length, y] = obj_type()

    def vert_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        length = self.height - y if length is None else length
        self.grid[x, y:y+length] = obj_type()

    def wall_rect(self, x: int, y: int, w: int, h: int):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self, k: int = 1) -> Grid:
        """
        Rotate the grid to the left (counter-clockwise)
        """
        new_grid_array = np.rot90(self.grid, k=k%4)
        new_width, new_height = new_grid_array.shape[:2]
        new_grid = Grid(new_width, new_height)
        new_grid.grid = new_grid_array
        return new_grid

    def slice(self, topX: int, topY: int, width: int, height: int) -> Grid:
        """
        Get a subset of the grid
        """
        x_min, x_max = clip((topX, topX + width), low=0, high=self.width)
        y_min, y_max = clip((topY, topY + height), low=0, high=self.height)
        i_min, i_max = x_min - topX, x_max - topX
        j_min, j_max = y_min - topY, y_max - topY

        grid = Grid(width, height)
        grid.grid[...] = Wall().array
        grid.grid[i_min:i_max, j_min:j_max] = self.grid[x_min:x_max, y_min:y_max]

        return grid

    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None,
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, highlight, tile_size)
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

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

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
        agent_pos: tuple[int, int],
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
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

                agent_here = np.array_equal(agent_pos, (i, j))
                assert highlight_mask is not None
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
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
        Produce a compact numpy encoding of the grid
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype=int)
        array[vis_mask] = self.grid[vis_mask][..., :3]
        return array

    @staticmethod
    def decode(array: np.ndarray) -> tuple[Grid, np.ndarray]:
        """
        Decode an array grid encoding back into a grid
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
                vis_mask[i, j] = type_idx != OBJECT_TO_IDX["unseen"]

        return grid, vis_mask

    def process_vis(self, agent_pos: tuple[int, int]) -> np.ndarray:
        vis_mask = get_vis_mask(self.grid, agent_pos)
        self.grid[~vis_mask] = empty()
        return vis_mask
