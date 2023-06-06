
from ray.tune.registry import register_env

from .env_utils import to_rllib_env
from ..envs import *



### Blocked Unlock Pickup

register_env(
    'MultiGrid-BlockedUnlockPickup-v0',
    to_rllib_env(BlockedUnlockPickupEnv),
)


### Empty

register_env(
    'MultiGrid-Empty-5x5-v0',
    to_rllib_env(EmptyEnv, default_config={'size': 5}),
)

register_env(
    'MultiGrid-Empty-Random-5x5-v0',
    to_rllib_env(EmptyEnv, default_config={'size': 5, 'agent_start_pos': None}),
)

register_env(
    'MultiGrid-Empty-6x6-v0',
    to_rllib_env(EmptyEnv, default_config={'size': 6}),
)

register_env(
    'MultiGrid-Empty-Random-6x6-v0',
    to_rllib_env(EmptyEnv, default_config={'size': 6, 'agent_start_pos': None}),
)

register_env(
    'MultiGrid-Empty-8x8-v0',
    to_rllib_env(EmptyEnv)
)

register_env(
    'MultiGrid-Empty-16x16-v0',
    to_rllib_env(EmptyEnv, default_config={'size': 16}),
)
