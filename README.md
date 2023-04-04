# multigrid

The **`multigrid`** library provides contains a collection of fast multi-agent discrete gridworld environments for reinforcement learning in Gymnasium. The environments are designed to be fast and easily customizable.

This is a multi-agent extension of the [`minigrid`](https://github.com/Farama-Foundation/Minigrid) library, and the interface is designed to be as similar as possible. 

Compared to `minigrid`, the underlying gridworld logic is **significantly optimized**, with environment simulation 10x to 20x faster by our benchmarks.

## Installation

    git clone https://github.com/ini/multigrid
    cd multigrid
    pip install -e .

This package requires Python 3.9 or later.

## Single-Agent MiniGrid Environments

This library provides the original single-agent `minigrid` environments, using the optimized **`multigrid`** under the hood.

```
from multigrid.envs.minigrid import EmptyEnv
my_env = EmptyEnv(size=5)
```

## Citation

To cite this project please use:

```
@software{multigrid,
  author = {Oguntola, Ini},
  title = {Fast Multi-Agent Gridworld Environment for Gymnasium},
  url = {https://github.com/ini/multigrid},
  year = {2023},
}
```
