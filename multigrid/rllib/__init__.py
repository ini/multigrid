from .env_utils import to_rllib_env
from ..envs import CONFIGURATIONS
from ..wrappers import OneHotObsWrapper

# Register environments with RLlib
from ray.tune.registry import register_env
for name, (env_cls, config) in CONFIGURATIONS.items():
    register_env(name, to_rllib_env(env_cls, OneHotObsWrapper, default_config=config))
