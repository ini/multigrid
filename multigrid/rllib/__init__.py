"""
This package provides tools for using MultiGrid environments with RLlib.

*****
Usage
*****

Register the environment configurations in :mod:`multigrid.envs` with RLlib:

    >>> import multigrid.rllib

Use a specific MultiGrid environment configuration by name:

    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> algorithm_config = PPOConfig().environment(env='MultiGrid-Empty-8x8-v0')

Convert a custom :class:`.MultiGridEnv` to an RLlib ``MultiAgentEnv``:

    >>> from multigrid.rllib import to_rllib_env
    >>> MyRLLibEnvClass = to_rllib_env(MyEnvClass)
    >>> algorithm_config = PPOConfig().environment(env=MyRLLibEnvClass)
"""

from .env_utils import to_rllib_env
from ..envs import CONFIGURATIONS
from ..wrappers import OneHotObsWrapper

# Register environments with RLlib
from ray.tune.registry import register_env
for name, (env_cls, config) in CONFIGURATIONS.items():
    register_env(name, to_rllib_env(env_cls, OneHotObsWrapper, default_config=config))
