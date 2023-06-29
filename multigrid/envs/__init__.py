"""
************
Environments
************

This package contains implementations of several MultiGrid environments.

**************
Configurations
**************

* `Blocked Unlock Pickup <./multigrid.envs.blockedunlockpickup>`_
    * ``MultiGrid-BlockedUnlockPickup-v0``
* `Empty <./multigrid.envs.empty>`_
    * ``MultiGrid-Empty-5x5-v0``
    * ``MultiGrid-Empty-Random-5x5-v0``
    * ``MultiGrid-Empty-6x6-v0``
    * ``MultiGrid-Empty-Random-6x6-v0``
    * ``MultiGrid-Empty-8x8-v0``
    * ``MultiGrid-Empty-16x16-v0``
* `Locked Hallway <./multigrid.envs.locked_hallway>`_
    * ``MultiGrid-LockedHallway-2Rooms-v0``
    * ``MultiGrid-LockedHallway-4Rooms-v0``
    * ``MultiGrid-LockedHallway-6Rooms-v0``
* `Playground <./multigrid.envs.playground>`_
    * ``MultiGrid-Playground-v0``
* `Red Blue Doors <./multigrid.envs.redbluedoors>`_
    * ``MultiGrid-RedBlueDoors-6x6-v0``
    * ``MultiGrid-RedBlueDoors-8x8-v0``
"""

from .blockedunlockpickup import BlockedUnlockPickupEnv
from .empty import EmptyEnv
from .locked_hallway import LockedHallwayEnv
from .playground import PlaygroundEnv
from .redbluedoors import RedBlueDoorsEnv

CONFIGURATIONS = {
    'MultiGrid-BlockedUnlockPickup-v0': (BlockedUnlockPickupEnv, {}),
    'MultiGrid-Empty-5x5-v0': (EmptyEnv, {'size': 5}),
    'MultiGrid-Empty-Random-5x5-v0': (EmptyEnv, {'size': 5, 'agent_start_pos': None}),
    'MultiGrid-Empty-6x6-v0': (EmptyEnv, {'size': 6}),
    'MultiGrid-Empty-Random-6x6-v0': (EmptyEnv, {'size': 6, 'agent_start_pos': None}),
    'MultiGrid-Empty-8x8-v0': (EmptyEnv, {}),
    'MultiGrid-Empty-16x16-v0': (EmptyEnv, {'size': 16}),
    'MultiGrid-LockedHallway-2Rooms-v0': (LockedHallwayEnv, {'num_rooms': 2}),
    'MultiGrid-LockedHallway-4Rooms-v0': (LockedHallwayEnv, {'num_rooms': 4}),
    'MultiGrid-LockedHallway-6Rooms-v0': (LockedHallwayEnv, {'num_rooms': 6}),
    'MultiGrid-Playground-v0': (PlaygroundEnv, {}),
    'MultiGrid-RedBlueDoors-6x6-v0': (RedBlueDoorsEnv, {'size': 6}),
    'MultiGrid-RedBlueDoors-8x8-v0': (RedBlueDoorsEnv, {'size': 8}),
}

# Register environments with gymnasium
from gymnasium.envs.registration import register
for name, (env_cls, config) in CONFIGURATIONS.items():
    register(id=name, entry_point=env_cls, kwargs=config)
