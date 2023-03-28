import numpy as np
from .array import ARRAY_DIM
from .constants import COLORS, IDX_TO_COLOR, DIR_TO_VEC
from .world_object import WorldObj
from ..utils.rendering import fill_coords, point_in_triangle, rotate_fn



class Agent(WorldObj):

    def __init__(self, index: int = 0, view_size: int = 7):
        super().__init__('agent', IDX_TO_COLOR[index])
        self.index = index
        self.view_size = view_size
        self.reset()

    def reset(self):
        self.pos = None
        self.dir = None
        self.carrying = None
        self.terminated = False

    def encode(self, current_agent: bool = False):
        return (
            self.type, # type
            self.color, # color
            self.dir, # state
            #int(current_agent), # indicator (i.e. this is me)
            #*self.carrying[:ARRAY_DIM // 2], # carried object
        )

    def render(self, img: np.ndarray[int]):
        if self.dir is None:
            return

        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        c = COLORS[self.color]
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * self.dir)
        fill_coords(img, tri_fn, c)

    @property
    def carrying(self) -> WorldObj:
        """
        Get the object that the agent is currently carrying.
        Alias for `contains`.
        """
        return self.contains
    
    @carrying.setter
    def carrying(self, obj: WorldObj):
        """
        Set the object that the agent is currently carrying.
        """
        self.contains = obj

    @property
    def dir_vec(self) -> int:
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert (
            0 <= self.dir < 4,
        ), f"Invalid agent_dir: {self.agent_dir} is not within range(0, 4)"
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self) -> np.ndarray[int]:
        """
        Get the vector pointing to the right of the agent.
        """
        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent.
        """
        return self.pos + self.dir_vec

    def get_view_coords(self, i, j) -> tuple[int, int]:
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid).

        Note that the resulting coordinates may be negative or outside of
        the agent's view size.
        """
        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self) -> tuple[int, int, int, int]:
        """
        Get the extents of the square set of tiles visible to the agent.
        Note: the bottom extent indices are not included in the set.
        """
        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2

        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]

        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2

        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1

        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return topX, topY, botX, botY
    
    def relative_coords(self, x, y) -> tuple[int, int]:
        """
        Check if a grid position belongs to the agent's field of view,
        and return the corresponding coordinates.
        """
        vx, vy = self.get_view_coords(x, y)
        if not (0 <= vx < self.view_size) and (0 <= vy < self.view_size):
            return None

        return vx, vy
