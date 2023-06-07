"""
************
Environments
************

This module contains implementations of several MultiGrid environments.

**************
Configurations
**************

* `Blocked Unlock Pickup <./blockedunlockpickup>`_
    * ``MultiGrid-BlockedUnlockPickup-v0``
* `Empty <./empty>`_
    * ``MultiGrid-Empty-5x5-v0``
    * ``MultiGrid-Empty-Random-5x5-v0``
    * ``MultiGrid-Empty-6x6-v0``
    * ``MultiGrid-Empty-Random-6x6-v0``
    * ``MultiGrid-Empty-8x8-v0``
    * ``MultiGrid-Empty-16x16-v0``
* `Locked Hallway <./locked_hallway>`_
    * ``MultiGrid-LockedHallway-4Rooms-v0``
    * ``MultiGrid-LockedHallway-6Rooms-v0``
"""

from .blockedunlockpickup import BlockedUnlockPickupEnv
from .empty import EmptyEnv
from .locked_hallway import LockedHallwayEnv

CONFIGURATIONS = {
    'MultiGrid-BlockedUnlockPickup-v0': (BlockedUnlockPickupEnv, {}),
    'MultiGrid-Empty-5x5-v0': (EmptyEnv, {'size': 5}),
    'MultiGrid-Empty-Random-5x5-v0': (EmptyEnv, {'size': 5, 'agent_start_pos': None}),
    'MultiGrid-Empty-6x6-v0': (EmptyEnv, {'size': 6}),
    'MultiGrid-Empty-Random-6x6-v0': (EmptyEnv, {'size': 6, 'agent_start_pos': None}),
    'MultiGrid-Empty-8x8-v0': (EmptyEnv, {}),
    'MultiGrid-Empty-16x16-v0': (EmptyEnv, {'size': 16}),
    'MultiGrid-LockedHallway-4Rooms-v0': (LockedHallwayEnv, {'num_rooms': 4}),
    'MultiGrid-LockedHallway-6Rooms-v0': (LockedHallwayEnv, {'num_rooms': 6}),
}
