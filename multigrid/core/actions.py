import enum



class Actions(enum.IntEnum):
    """
    Enumeration of possible actions.
    """
    left = 0 #: turn left
    right = enum.auto() #: turn right
    forward = enum.auto() #: move forward
    pickup = enum.auto() #: pick up an object
    drop = enum.auto() #: drop an object
    toggle = enum.auto() #: toggle/activate an object
    done = enum.auto() #: done completing task
