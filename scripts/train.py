from __future__ import annotations

import argparse
import json
import os
import random
import ray
import torch
import multigrid.rllib

from ray import tune
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls
from typing import Any

from centralized_critic import CentralizedCriticCallbacks, get_critic_input_space
from models import CustomModel, CustomLSTMModel
from utils import find_checkpoint_dir, get_policy_mapping_fn



def model_config(lstm: bool = False, **model_kwargs) -> dict[str, Any]:
    """
    Return a model configuration dictionary for RLlib.

    Parameters
    ----------
    lstm : bool
        Whether to use an LSTM model
    """
    return {
        'custom_model': CustomLSTMModel if lstm else CustomModel,
        'custom_model_config': model_kwargs,
        'conv_filters': [
            [16, [3, 3], 1],
            [16, [1, 1], 1],
            [32, [3, 3], 1],
            [32, [1, 1], 1],
            [64, [3, 3], 1],
            [64, [1, 1], 1],
        ],
        'fcnet_hiddens': [64, 64],
        'fcnet_activation': 'tanh',
        'post_fcnet_hiddens': [],
        'lstm_cell_size': 64,
        'max_seq_len': 8,
    }

def algorithm_config(
    algo: str = 'PPO',
    env: str = 'MultiGrid-Empty-8x8-v0',
    env_config: dict = {},
    num_agents: int = 2,
    lstm: bool = False,
    num_workers: int = 0,
    num_gpus: int = 0,
    lr: float | None = None,
    centralized_critic: bool = False,
    **kwargs) -> AlgorithmConfig:
    """
    Return the RL algorithm configuration dictionary.
    """
    _, env_creator = Algorithm._get_env_id_and_creator(env, {})
    env_config = {**env_config, 'agents': num_agents}
    dummy_env = env_creator(env_config)
    value_input_space = get_critic_input_space(dummy_env) if centralized_critic else None

    config = (
        get_trainable_cls(algo)
        .get_default_config()
        .debugging(seed=random.randint(0, int(1e6)))
        .environment(env=env, env_config=env_config)
        .framework('torch')
        .multi_agent(
            policies={f'policy_{i}' for i in range(num_agents)},
            policy_mapping_fn=get_policy_mapping_fn(None, num_agents),
        )
        .resources(num_gpus=num_gpus if torch.cuda.is_available() else 0)
        #.rl_module(_enable_rl_module_api=False) # for older RLlib versions
        .experimental(_enable_new_api_stack=False)
        .rollouts(num_rollout_workers=num_workers)
        .training(
            #_enable_learner_api=False, # for older RLlib versions
            model=model_config(lstm=lstm, value_input_space=value_input_space),
            lr=(lr or NotProvided),
        )
    )

    if centralized_critic:
        config = config.callbacks(CentralizedCriticCallbacks)

    return config

def train(
    algo: str,
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str | None = None):
    """
    Train an RLlib algorithm.
    """
    ray.init(num_cpus=min(os.cpu_count(), config.num_rollout_workers + 1))
    tune.run(
        algo,
        stop=stop_conditions,
        config=config,
        local_dir=save_dir,
        verbose=1,
        restore=find_checkpoint_dir(load_dir),
        checkpoint_freq=20,
        checkpoint_at_end=True,
    )
    ray.shutdown()



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
    parser.add_argument(
        '--lstm', action='store_true',
        help="Use LSTM model.")
    parser.add_argument(
        '--centralized-critic', action='store_true',
        help="Use centralized critic for training.")
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
    config = algorithm_config(**vars(args))

    print()
    print(f"Running with following CLI options: {args}")
    print('\n', '-' * 64, '\n', "Training with following configuration:", '\n', '-' * 64)
    print()

    stop_conditions = {'timesteps_total': args.num_timesteps}
    train(args.algo, config, stop_conditions, args.save_dir, args.load_dir)
