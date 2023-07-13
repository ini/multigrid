# Training MultiGrid agents with RLlib

MultiGrid is compatible with RLlib's multi-agent API.

This folder provides scripts to train and visualize agents over MultiGrid environments.

## Requirements

Using MultiGrid environments with RLlib requires installation of [rllib](https://docs.ray.io/en/latest/rllib/index.html), and one of [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).

## Getting Started

Train 2 agents on the `MultiGrid-Empty-8x8-v0` environment using the PPO algorithm:

    python train.py --algo PPO --env MultiGrid-Empty-8x8-v0 --num-agents 2 --save-dir ~/saved/empty8x8/

Visualize behavior from trained agents policies:

    python visualize.py --algo PPO --env MultiGrid-Empty-8x8-v0 --num-agents 2 --load-dir ~/saved/empty8x8/

For more options, run ``python train.py --help`` and ``python visualize.py --help``.

## Environments

All of the environment configurations registered in [`multigrid.envs`](../multigrid/envs/__init__.py) can also be used with RLlib, and are registered via `import multigrid.rllib`.

To use a specific MultiGrid environment configuration by name:

    >>> import multigrid.rllib
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> algorithm_config = PPOConfig().environment(env='MultiGrid-Empty-8x8-v0')

To convert a custom `MultiGridEnv` to an RLlib `MultiAgentEnv`:

    >>> from multigrid.rllib import to_rllib_env
    >>> MyRLLibEnvClass = to_rllib_env(MyEnvClass)
    >>> algorithm_config = PPOConfig().environment(env=MyRLLibEnvClass)
