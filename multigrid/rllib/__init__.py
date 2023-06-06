
from ray.tune.registry import register_env

from .env import rllib_multi_agent_env
from ..envs.minigrid import *


### Blocked Unlock Pickup

register_env(
    'MultiGrid-BlockedUnlockPickup-v0',
    rllib_multi_agent_env(BlockedUnlockPickupEnv),
)


### Empty

register_env(
    'MiniGrid-Empty-5x5-v0',
    rllib_multi_agent_env(EmptyEnv, default_config={'size': 5}),
)

register_env(
    'MiniGrid-Empty-Random-5x5-v0',
    rllib_multi_agent_env(EmptyEnv, default_config={'size': 5, 'agent_start_pos': None}),
)

register_env(
    'MiniGrid-Empty-6x6-v0',
    rllib_multi_agent_env(EmptyEnv, default_config={'size': 6}),
)

register_env(
    'MiniGrid-Empty-Random-6x6-v0',
    rllib_multi_agent_env(EmptyEnv, default_config={'size': 6, 'agent_start_pos': None}),
)

register_env(
    'MiniGrid-Empty-8x8-v0',
    rllib_multi_agent_env(EmptyEnv)
)

register_env(
    'MiniGrid-Empty-16x16-v0',
    rllib_multi_agent_env(EmptyEnv, default_config={'size': 16}),
)
