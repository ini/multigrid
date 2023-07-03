from __future__ import annotations

import argparse
import json
import os
import ray

from multigrid.rllib.models import TFModel, TorchModel, TorchLSTMModel
from pathlib import Path
from pprint import pprint
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls



def get_checkpoint_dir(search_dir: Path | str | None) -> Path | None:
    """
    Recursively search for checkpoints within the given directory.

    If more than one is found, returns the most recently modified checkpoint directory.

    Parameters
    ----------
    search_dir : Path or str
        The directory to search for checkpoints within
    """
    if search_dir:
        checkpoints = Path(search_dir).expanduser().glob('**/*.is_checkpoint')
        if checkpoints:
            return sorted(checkpoints, key=os.path.getmtime)[-1].parent

    return None

def can_use_gpu() -> bool:
    """
    Return whether or not GPU training is available.
    """
    try:
        _, tf, _ = try_import_tf()
        return tf.test.is_gpu_available()
    except:
        pass

    try:
        torch, _ = try_import_torch()
        return torch.cuda.is_available()
    except:
        pass

    return False

def policy_mapping_fn(agent_id: int, *args, **kwargs) -> str:
    """
    Map an environment agent ID to an RLlib policy ID.
    """
    return f'policy_{agent_id}'

def model_config(
    framework: str = 'torch',
    lstm: bool = False,
    custom_model_config: dict = {}):
    """
    Return a model configuration dictionary for RLlib.
    """
    if framework == 'torch':
        if lstm:
            model = TorchLSTMModel
        else:
            model = TorchModel
    else:
        if lstm:
            raise NotImplementedError
        else:
            model = TFModel

    return {
        'custom_model': model,
        'custom_model_config': custom_model_config,
        'conv_filters': [
            [16, [3, 3], 1],
            [32, [3, 3], 1],
            [64, [3, 3], 1],
        ],
        'fcnet_hiddens': [64, 64],
        'post_fcnet_hiddens': [],
        'lstm_cell_size': 256,
        'max_seq_len': 20,
    }

def algorithm_config(
    algo: str = 'PPO',
    env: str = 'MultiGrid-Empty-8x8-v0',
    env_config: dict = {},
    num_agents: int = 2,
    framework: str = 'torch',
    lstm: bool = False,
    num_workers: int = 0,
    num_gpus: int = 0,
    lr: float | None = None,
    **kwargs) -> AlgorithmConfig:
    """
    Return the RL algorithm configuration dictionary.
    """
    env_config = {**env_config, 'agents': num_agents}
    return (
        get_trainable_cls(algo)
        .get_default_config()
        .environment(env=env, env_config=env_config)
        .framework(framework)
        .rollouts(num_rollout_workers=num_workers)
        .resources(num_gpus=num_gpus if can_use_gpu() else 0)
        .multi_agent(
            policies={f'policy_{i}' for i in range(num_agents)},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            model=model_config(framework=framework, lstm=lstm),
            lr=(lr or NotProvided),
        )
    )

def train(
    algo: str,
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str | None = None):
    """
    Train an RLlib algorithm.
    """
    ray.init(num_cpus=(config.num_rollout_workers + 1))
    tune.run(
        algo,
        stop=stop_conditions,
        config=config,
        local_dir=save_dir,
        verbose=1,
        restore=get_checkpoint_dir(load_dir),
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
        '--framework', type=str, choices=['torch', 'tf', 'tf2'], default='torch',
        help="Deep learning framework to use.")
    parser.add_argument(
        '--lstm', action='store_true', help="Use LSTM model.")
    parser.add_argument(
        '--env', type=str, default='MultiGrid-Empty-8x8-v0',
        help="MultiGrid environment to use.")
    parser.add_argument(
        '--env-config', type=json.loads, default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')")
    parser.add_argument(
        '--num-agents', type=int, default=2, help="Number of agents in environment.")
    parser.add_argument(
        '--num-workers', type=int, default=8, help="Number of rollout workers.")
    parser.add_argument(
        '--num-gpus', type=int, default=1, help="Number of GPUs to train on.")
    parser.add_argument(
        '--num-timesteps', type=int, default=1e7,
        help="Total number of timesteps to train.")
    parser.add_argument(
        '--lr', type=float, help="Learning rate for training.")
    parser.add_argument(
        '--load-dir', type=str,
        help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument(
        '--save-dir', type=str, default='~/ray_results/',
        help="Directory for saving checkpoints, results, and trained policies.")

    args = parser.parse_args()
    config = algorithm_config(**vars(args))
    stop_conditions = {'timesteps_total': args.num_timesteps}

    print()
    print(f"Running with following CLI options: {args}")
    print('\n', '-' * 64, '\n', "Training with following configuration:", '\n', '-' * 64)
    print()
    pprint(config.to_dict())
    train(args.algo, config, stop_conditions, args.save_dir, args.load_dir)
