# MultiGrid

<br/>
<p align="center">
  <img src="https://i.imgur.com/usbavAh.gif" width=400 alt="Blocked Unlock Pickup: 2 Agents">
</p>
<br/>

The **MultiGrid** library provides contains a collection of fast multi-agent discrete gridworld environments for reinforcement learning in [Gymnasium](https://github.com/Farama-Foundation/Gymnasium). This is a multi-agent extension of the [minigrid](https://github.com/Farama-Foundation/Minigrid) library, and the interface is designed to be as similar as possible.

The environments are designed to be fast and easily customizable. Compared to minigrid, the underlying gridworld logic is **significantly optimized**, with environment simulation 10x to 20x faster by our benchmarks.

Documentation for this library can be found at [ini.io/docs/multigrid](https://ini.io/docs/multigrid).

## Installation

    git clone https://github.com/ini/multigrid
    cd multigrid
    pip install -e .

This package requires Python 3.9 or later.

## Environments

The `multigrid.envs` package provides implementations of several multi-agent environments. [You can find the full list here](https://ini.io/docs/multigrid/multigrid/multigrid.envs).

## API

MultiGrid follows the same pattern as RLlib's [MultiAgentEnv API](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical) and PettingZoo's [ParallelEnv API](https://pettingzoo.farama.org/api/parallel/).

```python
import gymnasium as gym
import multigrid.envs

env = gym.make('MultiGrid-Empty-8x8-v0', agents=2, render_mode='human')

observations, infos = env.reset()
while not env.is_done():
   # this is where you would insert your policy / policies
   actions = {agent.index: agent.action_space.sample() for agent in env.agents}
   observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

More information about using MultiGrid directly with other APIs:
* [PettingZoo](https://ini.io/docs/multigrid/multigrid/multigrid.pettingzoo)
* [RLlib](https://ini.io/docs/multigrid/multigrid/multigrid.rllib)

## Training Agents

See the [scripts folder](./scripts) for an example training with RLlib. 

## Documentation

Documentation for this package can be found at [ini.io/docs/multigrid](https://ini.io/docs/multigrid).

## Citation

To cite this project please use:

```
@software{multigrid,
  author = {Oguntola, Ini},
  title = {Fast Multi-Agent Gridworld Environments for Gymnasium},
  url = {https://github.com/ini/multigrid},
  year = {2023},
}
```
