
from .env import rllib_multi_agent_env
from ..envs.minigrid import BlockedUnlockPickupEnv


from ray.tune.registry import register_env

register_env(
    'MultiGrid-BlockedUnlockPickup-v0',
    rllib_multi_agent_env(BlockedUnlockPickupEnv),
)
