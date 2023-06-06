# MultiGrid with RLlib

MultiGrid is compatible with RLlib's multi-agent API.

This module provides scripts and tools to train agents across all MultiGrid environments.

## Requirements

Using MultiGrid environments with RLlib requires installation of [rllib](https://docs.ray.io/en/latest/rllib/index.html), and one of [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).

## Getting Started

Train a 2-agent environment using PPO:

    python scripts/train.py --algo PPO --framework torch --env MultiGrid-BlockedUnlockPickup-v0 --num-agents 2 --num-timesteps 10000000 --save-dir ~/saved/

Visualize agent behavior:

    python scripts/visualize.py --algo PPO --framework torch --env MultiGrid-BlockedUnlockPickup-v0 --num-agents 2 --num-episodes 100 --save-dir ~/saved/

## Environments

To use with RLlib: `import multigrid.rllib`.

The full list of RLlib-registered configurations of MultiGrid environments can be found in [multigrid/rllib/__init__.py](./__init__.py).
