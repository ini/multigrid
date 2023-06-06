import argparse
import ray

from multigrid.rllib.model import TFModel, TorchModel
from pprint import pprint
from ray import air, tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.tune.registry import get_trainable_cls



def policy_mapping_fn(agent_id: int, *args, **kwargs) -> str:
    """
    Map an environment agent ID to an RLlib policy ID.
    """
    return f'policy_{agent_id}'

def model_config(framework: str = 'torch', custom_model_config: dict = {}):
    """
    Return a model configuration dictionary for RLlib.
    """
    return {
        'custom_model': TorchModel if framework == 'torch' else TFModel,
        'custom_model_config': custom_model_config,
        'conv_filters': [
            [16, [3, 3], 1],
            [32, [3, 3], 1],
            [64, [3, 3], 1],
        ],
        'fcnet_hiddens': [64, 64],
        'post_fcnet_hiddens': [64],
    }

def algorithm_config(
    algo: str = 'PPO',
    env: str = 'MultiGrid-Empty-8x8-v0',
    env_config: dict = {},
    num_agents: int = 2,
    framework: str = 'torch',
    num_workers: int = 0,
    num_gpus: int = 0,
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
        .resources(num_gpus=num_gpus)
        .multi_agent(
            policies={f'policy_{i}' for i in range(num_agents)},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            model=model_config(framework),
            lr=tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6]),
        )
    )

def train(algo: str, config: AlgorithmConfig, stop_conditions: dict, save_dir: str):
    """
    Train an RLlib algorithm.
    """
    ray.init()
    tuner = tune.Tuner(
        algo,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop_conditions,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=20,
                checkpoint_at_end=True,
            ),
            local_dir=save_dir,
        ),
    )
    results = tuner.fit()
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
        '--env', type=str, default='MultiGrid-Empty-8x8-v0',
        help="MultiGrid environment to use.")
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
        '--save-dir', type=str, default='~/ray_results/',
        help="Directory for saving results and trained models.")

    args = parser.parse_args()
    config = algorithm_config(**vars(args))
    stop_conditions = {'timesteps_total': args.num_timesteps}

    print(f"Running with following CLI options: {args}")
    print('\n', '-' * 64, '\n', "Training with following configuration:", '\n', '-' * 64)
    print()
    pprint(config.to_dict())
    train(args.algo, config, stop_conditions, args.save_dir)
