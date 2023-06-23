import functools
from typing import Any
from ..core.constants import Direction



@functools.cache
def front_pos(agent_x: int, agent_y: int, agent_dir: int):
    """
    Get the position in front of an agent.
    """
    dx, dy = Direction(agent_dir).to_vec()
    return (agent_x + dx, agent_y + dy)



class PropertyAlias(property):
    """
    A class property that is an alias for an attribute property.

    Instead of::

        @property
        def x(self):
            self.attr.x

        @x.setter
        def x(self, value):
            self.attr.x = value

    we can simply just declare::

        x = PropertyAlias('attr', 'x')
    """

    def __init__(self, attr_name: str, attr_property_name: str, doc: str = None) -> None:
        """
        Parameters
        ----------
        attr_name : str
            Name of the base attribute
        attr_property : property
            Property from the base attribute class
        doc : str
            Docstring to append to the property's original docstring
        """
        prop = lambda obj: getattr(type(getattr(obj, attr_name)), attr_property_name)
        fget = lambda obj: prop(obj).fget(getattr(obj, attr_name))
        fset = lambda obj, value: prop(obj).fset(getattr(obj, attr_name), value)
        fdel = lambda obj: prop(obj).fdel(getattr(obj, attr_name))
        super().__init__(fget, fset, fdel, doc=doc)
        self.__doc__ = doc
