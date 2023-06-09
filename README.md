# MultiGrid

<br/>
<p align="center">
  <img src="https://i.imgur.com/usbavAh.gif" width=400 alt="Blocked Unlock Pickup: 2 Agents">
</p>
<br/>

The **MultiGrid** library provides contains a collection of fast multi-agent discrete gridworld environments for reinforcement learning. This is a multi-agent extension of the [minigrid](https://github.com/Farama-Foundation/Minigrid) library, and the interface is designed to be as similar as possible.


The environments are designed to be fast and easily customizable. Compared to minigrid, the underlying gridworld logic is **significantly optimized**, with environment simulation 10x to 20x faster by our benchmarks.

## Installation

    git clone https://github.com/ini/multigrid
    cd multigrid
    pip install -e .

This package requires Python 3.9 or later.

## API

The MultiGrid API is compatible with RLLib's [MultiAgentEnv API](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical) environment specification (and similar to PettingZoo's [ParallelEnv API](https://pettingzoo.farama.org/api/parallel/)).

```python
import gymnasium as gym
import multigrid.envs

env = gym.make('MultiGrid-Empty-8x8-v0', agents=2, render_mode='human')
obs, info = env.reset()

for _ in range(1000):
   actions = {i: policies[i](obs[i]) for i in len(env.agents)}  # user-defined policy functions
   obs, rewards, terminateds, truncateds, infos = env.step(actions) # dicts indexed by agent ID
   if all(terminated.values()) or all(truncated.values()):
      obs, info = env.reset()

env.close()
 ```

## Environments

The `multigrid.envs` package provides implementations of several multi-agent environments. [You can find the full list here](https://ini.io/docs/multigrid/multigrid.envs).

## Documentation

Documentation for this package can be found at [ini.io/docs/multigrid](https://ini.io/docs/multigrid).

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
