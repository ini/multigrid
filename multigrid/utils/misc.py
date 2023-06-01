import functools
from ..core.constants import DIR_TO_VEC



@functools.cache
def front_pos(agent_x: int, agent_y: int, agent_dir: int):
    """
    Get the position in front of an agent.
    """
    dx, dy = DIR_TO_VEC[agent_dir]
    return (agent_x + dx, agent_y + dy)



class PropertyAlias(property):
    """
    A class property that is an alias for an attribute property.

    Instead of:
    ```
    @property
    def x(self):
        self.attr.x

    @x.setter
    def x(self, value):
        self.attr.x = value
    ```

    we can simply declare:
    ```
    x = PropertyAlias('attr', AttributeClass.x)
    ```
    """

    def __init__(self, attr_name: str, attr_property: property) -> None:
        """
        Parameters
        ----------
        attr_name : str
            Name of the base attribute
        attr_property : property
            Property from the base attribute class
        """
        fget = lambda obj: attr_property.fget(getattr(obj, attr_name))
        fset = lambda obj, value: attr_property.fset(getattr(obj, attr_name), value)
        fdel = lambda obj: attr_property.fdel(getattr(obj, attr_name))
        super().__init__(fget, fset, fdel, attr_property.__doc__)
