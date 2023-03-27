def Empty() -> np.ndarray[int]:
    return np.array([
        OBJECT_TO_IDX['empty'], 0, 0, 0,
        OBJECT_TO_IDX['empty'], 0, 0, 0,
    ])


"""
Base class for grid world objects.

Uses the following encoding:
----------------------------
0: object type
1: object color
2: object state
3: object ID
4: contained object type
5: contained object color
6: contained object state
7: contained object ID
"""

ARRAY_DIM = 8

def contents(array: np.ndarray[int]) -> np.ndarray[int]:
    x = np.zeros_like(array)
    x[..., :4] = array[..., 4:]
    x[..., 4:] = Empty()
    return x

def can_overlap(array: np.ndarray[int]) -> np.array[bool]:
    mask = (array[..., 0] == OBJECT_TO_IDX['empty'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['goal'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['floor'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['lava'])
    mask |= (
        (array[..., 0] == OBJECT_TO_IDX['door'])
        & (array[..., 2] == STATE_TO_IDX['open'])
    )
    return mask

def can_pickup(array: np.ndarray[int]) -> np.array[bool]:
    mask = (array[..., 0] == OBJECT_TO_IDX['key'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['ball'])
    mask |= (array[..., 0] == OBJECT_TO_IDX['box'])
    return mask

def can_contain(array: np.ndarray[int]) -> np.array[bool]:
    return (array[..., 0] == OBJECT_TO_IDX['box'])

def see_behind(array: np.ndarray[int]) -> np.array[bool]:
    neg_mask = (array[..., 0] == OBJECT_TO_IDX['wall'])
    neg_mask |= (
        (array[..., 0] == OBJECT_TO_IDX['door'])
        & (array[..., 2] != STATE_TO_IDX['open'])
    )
    return ~neg_mask

def toggle(self, array: np.ndarray[int], carrying: np.ndarray):
    # Handle doors
    is_door = (array[..., 0] == OBJECT_TO_IDX['door'])
    is_open = is_door & (array[..., 2] == STATE_TO_IDX['open'])
    is_closed = is_door & (array[..., 2] == STATE_TO_IDX['closed'])
    is_locked = is_door & (array[..., 2] == STATE_TO_IDX['locked'])
    can_unlock = (
        is_locked
        & (carrying[..., 0] == OBJECT_TO_IDX['key'])
        & (carrying[..., 1] == array[..., 1])
    )
    array[is_open][..., 2] = STATE_TO_IDX['closed'] # open -> closed
    array[is_closed][..., 2] = STATE_TO_IDX['open'] # closed -> open
    array[can_unlock][..., 2] = STATE_TO_IDX['open'] # locked -> open

    # Handle boxes
    is_box = (array[..., 0] == OBJECT_TO_IDX['box'])
    array[is_box] = MyWorldObj.contents(array[is_box]) # replace the box by its contents

def render(self, array: np.ndarray[int], img: np.ndarray):
    # TODO: vectorize all of this

    if array[..., 0] == OBJECT_TO_IDX['goal']:
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[array[..., 1]])

    elif array[..., 0] == OBJECT_TO_IDX['floor']:
        # Give the floor a pale color
        color = COLORS[array[..., 1]] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)

    elif array[..., 0] == OBJECT_TO_IDX['lava']:
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

    elif array[..., 0] == OBJECT_TO_IDX['wall']:
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[array[..., 1]])
    
    elif array[..., 0] == OBJECT_TO_IDX['door']:
        c = COLORS[array[..., 1]]

        if array[..., 2] == STATE_TO_IDX['open']:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if array[..., 2] == STATE_TO_IDX['locked']:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)
    
    elif array[..., 0] == OBJECT_TO_IDX['key']:
        c = COLORS[array[..., 1]]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))

    elif array[..., 0] == OBJECT_TO_IDX['ball']:
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[array[..., 1]])

    elif array[..., 0] == OBJECT_TO_IDX['box']:
        c = COLORS[array[..., 1]]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)
