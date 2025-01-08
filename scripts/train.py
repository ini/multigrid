from __future__ import annotations

import argparse
import json
import multigrid.rllib
import os
import random
import ray
import ray.train
import ray.tune
import torch
import torch.nn as nn

from multigrid.core.constants import Direction
from pathlib import Path
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.from_config import NotProvided
from typing import Callable



### Helper Methods

def get_policy_mapping_fn(
    checkpoint_dir: Path | str | None,
    num_agents: int,
) -> Callable:
    try:
        policies = sorted([
            path for path in (checkpoint_dir / 'policies').iterdir() if path.is_dir()])

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return policies[agent_id % len(policies)].name

        print('Loading policies from:', checkpoint_dir)
        for agent_id in range(num_agents):
            print('Agent ID:', agent_id, 'Policy ID:', policy_mapping_fn(agent_id))

        return policy_mapping_fn

    except:
        return lambda agent_id, *args, **kwargs: f'policy_{agent_id}'

def find_checkpoint_dir(search_dir: Path | str | None) -> Path | None:
    try:
        checkpoints = Path(search_dir).expanduser().glob('**/rllib_checkpoint.json')
        if checkpoints:
            return sorted(checkpoints, key=os.path.getmtime)[-1].parent
    except:
        return None

def preprocess_batch(batch: dict) -> torch.Tensor:
    image = batch['obs']['image']
    direction = batch['obs']['direction']
    direction = 2 * torch.pi * (direction / len(Direction))
    direction = torch.stack([torch.cos(direction), torch.sin(direction)], dim=-1)
    direction = direction[..., None, None, :].expand(*image.shape[:-1], 2)
    x = torch.cat([image, direction], dim=-1).float()
    return x



### Models

class MultiGridEncoder(nn.Module):

    def __init__(self, in_channels: int = 23):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, (3, 3)), nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)), nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        x = x[None] if x.ndim == 3 else x # add batch dimension
        x = x.permute(0, 3, 1, 2) # channels-first (NHWC -> NCHW)
        return self.model(x)


class AgentModule(TorchRLModule, ValueFunctionAPI):

    def setup(self):
        self.base = nn.Identity()
        self.actor = nn.Sequential(
            MultiGridEncoder(in_channels=23),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 7),
        )
        self.critic = nn.Sequential(
            MultiGridEncoder(in_channels=23),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _forward(self, batch, **kwargs):
        x = self.base(preprocess_batch(batch))
        return {Columns.ACTION_DIST_INPUTS: self.actor(x)}

    def _forward_train(self, batch, **kwargs):
        x = self.base(preprocess_batch(batch))
        return {
            Columns.ACTION_DIST_INPUTS: self.actor(x),
            Columns.EMBEDDINGS: self.critic(batch.get('value_inputs', x)),
        }
    
    def compute_values(self, batch, embeddings = None):
        if embeddings is None:
            x = self.base(preprocess_batch(batch))
            embeddings = self.critic(batch.get('value_inputs', x))

        return embeddings.squeeze(-1)

    def get_initial_state(self):
        return {}


    
### Training

def get_algorithm_config(
    env: str = 'MultiGrid-Empty-8x8-v0',
    env_config: dict = {},
    num_agents: int = 2,
    # lstm: bool = False, TODO: implement LSTM model
    num_workers: int = 0,
    num_gpus: int = 0,
    lr: float = NotProvided,
    batch_size: int = NotProvided,
    # centralized_critic: bool = False, TODO: implement centralized critic
    **kwargs,
) -> PPOConfig:
    """
    Return the RL algorithm configuration dictionary.
    """
    config = PPOConfig()
    config = config.api_stack(
        enable_env_runner_and_connector_v2=True,
        enable_rl_module_and_learner=True,
    )
    config = config.debugging(seed=random.randint(0, 1000000))
    config = config.env_runners(
        num_env_runners=num_workers,
        num_envs_per_env_runner=1,
        num_gpus_per_env_runner=num_gpus if torch.cuda.is_available() else 0,
    )
    config = config.environment(env=env, env_config={**env_config, 'agents': num_agents})
    config = config.framework('torch')
    config = config.multi_agent(
        policies={f'policy_{i}' for i in range(num_agents)},
        policy_mapping_fn=get_policy_mapping_fn(None, num_agents),
        policies_to_train=[f'policy_{i}' for i in range(num_agents)],
    )
    config = config.training(lr=lr, train_batch_size=batch_size)
    config = config.rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                f'policy_{i}': RLModuleSpec(module_class=AgentModule)
                for i in range(num_agents)
            }
        )
    )

    return config

def train(
    config: PPOConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str = None,
):
    """
    Train an RLlib algorithm.
    """
    checkpoint = find_checkpoint_dir(load_dir)
    if checkpoint:
        tuner = ray.tune.Tuner.restore(checkpoint)
    else:
        tuner = ray.tune.Tuner(
            config.algo_class,
            param_space=config,
            run_config=ray.train.RunConfig(
                storage_path=save_dir,
                stop=stop_conditions,
                verbose=1,
                checkpoint_config=ray.train.CheckpointConfig(
                    checkpoint_frequency=20,
                    checkpoint_at_end=True,
                ),
            ),
        )

    results = tuner.fit()
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--algo', type=str, default='PPO',
        help="The name of the RLlib-registered algorithm to use.")
    parser.add_argument(
        '--env', type=str, default='MultiGrid-Empty-8x8-v0',
        help="MultiGrid environment to use.")
    parser.add_argument(
        '--env-config', type=json.loads, default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')")
    parser.add_argument(
        '--num-agents', type=int, default=2,
        help="Number of agents in environment.")
    # parser.add_argument(
    #     '--lstm', action='store_true',
    #     help="Use LSTM model.")
    # parser.add_argument(
    #     '--centralized-critic', action='store_true',
    #     help="Use centralized critic for training.")
    parser.add_argument(
        '--num-workers', type=int, default=8,
        help="Number of rollout workers.")
    parser.add_argument(
        '--num-gpus', type=int, default=1,
        help="Number of GPUs to train on.")
    parser.add_argument(
        '--num-timesteps', type=int, default=1e7,
        help="Total number of timesteps to train.")
    parser.add_argument(
        '--lr', type=float,
        help="Learning rate for training.")
    parser.add_argument(
        '--load-dir', type=str,
        help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument(
        '--save-dir', type=str, default='~/ray_results/',
        help="Directory for saving checkpoints, results, and trained policies.")

    args = parser.parse_args()
    config = get_algorithm_config(**vars(args))

    print()
    print(f"Running with following CLI options: {args}")
    print('\n', '-' * 64, '\n', "Training with following configuration:", '\n', '-' * 64)
    print()

    stop_conditions = {
        'learners/__all_modules__/num_env_steps_trained_lifetime': args.num_timesteps}
    train(config, stop_conditions, args.save_dir, args.load_dir)
