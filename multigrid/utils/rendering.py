from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray as ndarray
from typing import Callable



# Constants

FilterFunction = Callable[[float, float], bool]
White = np.array([255, 255, 255])



# Functions

def downsample(img: ndarray[np.uint8], factor: int) -> ndarray[np.uint8]:
    """
    Downsample an image along both dimensions by some factor.

    Parameters
    ----------
    img : ndarray[uint8] of shape (height, width, 3)
        The image to downsample
    factor : int
        The factor by which to downsample the image

    Returns
    -------
    img : ndarray[uint8] of shape (height/factor, width/factor, 3)
        The downsampled image
    """
    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape(
        [img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3]
    )
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img

def fill_coords(
    img: ndarray[np.uint8],
    fn: FilterFunction,
    color: ndarray[np.uint8]) -> ndarray[np.uint8]:
    """
    Fill pixels of an image with coordinates matching a filter function.

    Parameters
    ----------
    img : ndarray[uint8] of shape (height, width, 3)
        The image to fill
    fn : Callable(float, float) -> bool
        The filter function to use for coordinates
    color : ndarray[uint8] of shape (3,)
        RGB color to fill matching coordinates

    Returns
    -------
    img : ndarray[np.uint8] of shape (height, width, 3)
        The updated image
    """
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img

def rotate_fn(fin: FilterFunction, cx: float, cy: float, theta: float) -> FilterFunction:
    """
    Rotate a coordinate filter function around a center point by some angle.

    Parameters
    ----------
    fin : Callable(float, float) -> bool
        The filter function to rotate
    cx : float
        The x-coordinate of the center of rotation
    cy : float
        The y-coordinate of the center of rotation
    theta : float
        The angle by which to rotate the filter function (in radians)

    Returns
    -------
    fout : Callable(float, float) -> bool
        The rotated filter function
    """
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout

def point_in_line(
    x0: float, y0: float, x1: float, y1: float, r: float) -> FilterFunction:
    """
    Return a filter function that returns True for points within distance r
    from the line between (x0, y0) and (x1, y1).

    Parameters
    ----------
    x0 : float
        The x-coordinate of the line start point
    y0 : float
        The y-coordinate of the line start point
    x1 : float
        The x-coordinate of the line end point
    y1 : float
        The y-coordinate of the line end point
    r : float
        Maximum distance from the line

    Returns
    -------
    fn : Callable(float, float) -> bool
        Filter function
    """
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn

def point_in_circle(cx: float, cy: float, r: float) -> FilterFunction:
    """
    Return a filter function that returns True for points within radius r
    from a given point.

    Parameters
    ----------
    cx : float
        The x-coordinate of the circle center
    cy : float
        The y-coordinate of the circle center
    r : float
        The radius of the circle

    Returns
    -------
    fn : Callable(float, float) -> bool
        Filter function
    """
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn

def point_in_rect(xmin: float, xmax: float, ymin: float, ymax: float) -> FilterFunction:
    """
    Return a filter function that returns True for points within a rectangle.

    Parameters
    ----------
    xmin : float
        The minimum x-coordinate of the rectangle
    xmax : float
        The maximum x-coordinate of the rectangle
    ymin : float
        The minimum y-coordinate of the rectangle
    ymax : float
        The maximum y-coordinate of the rectangle

    Returns
    -------
    fn : Callable(float, float) -> bool
        Filter function
    """
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn

def point_in_triangle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float]) -> FilterFunction:
    """
    Return a filter function that returns True for points within a triangle.

    Parameters
    ----------
    a : tuple[float, float]
        The first vertex of the triangle
    b : tuple[float, float]
        The second vertex of the triangle
    c : tuple[float, float]
        The third vertex of the triangle

    Returns
    -------
    fn : Callable(float, float) -> bool
        Filter function
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn

def highlight_img(
    img: ndarray[np.uint8],
    color: ndarray[np.uint8] = White,
    alpha=0.30) -> ndarray[np.uint8]:
    """
    Add highlighting to an image.

    Parameters
    ----------
    img : ndarray[uint8] of shape (height, width, 3)
        The image to highlight
    color : ndarray[uint8] of shape (3,)
        RGB color to use for highlighting
    alpha : float
        The alpha value to use for blending

    Returns
    -------
    img : ndarray[uint8] of shape (height, width, 3)
        The highlighted image
    """
    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img
