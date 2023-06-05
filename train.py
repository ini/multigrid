import argparse
import ray

from multigrid.rllib.model import TFModel, TorchModel
from multigrid.rllib.env import get_env_creator
from ray import air, tune
from ray.tune.registry import get_trainable_cls



def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f'policy_{agent_id}'

def model_config(framework='torch', custom_model_config={}):
    return {
        'custom_model': TorchModel if framework == 'torch' else TFModel,
        'custom_model_config': custom_model_config,
        'conv_filters': [
                [16, [2, 2], 2],
                [32, [2, 2], 2],
                [64, [2, 2], 2],
        ],
        'post_fcnet_hiddens': [256, 256],
    }

def algorithm_config(
    algo='PPO',
    env='MultiGrid-BlockedUnlockPickup-v0',
    env_config={},
    num_agents=2,
    framework='torch',
    num_workers=8,
    num_gpus=0,
    **kwargs):
    env_config = {**env_config, 'agents': num_agents}
    dummy_env = get_env_creator(env)(env_config)
    custom_model_config = {'joint_obs_space': dummy_env.observation_space}
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
            model=model_config(framework, custom_model_config=custom_model_config),
            lr=1e-5,
        )
    )

def train(algo, config, stop):
    #from central import CentralizedCritic
    ray.init()
    tuner = tune.Tuner(
        algo, #CentralizedCritic, #args.algo,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=20,
                checkpoint_at_end=True,
            ),
        ),   
    )
    results = tuner.fit()
    ray.shutdown()

    
    



parser = argparse.ArgumentParser()



parser.add_argument(
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1e7, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
)






if __name__ == "__main__":
    parser.add_argument(
        '--algo', type=str, default='PPO', help="The RLlib-registered algorithm to use.")
    parser.add_argument(
        '--framework', type=str, choices=['torch', 'tf', 'tf2'], default='torch',
        help="Deep learning framework to use.")
    parser.add_argument(
        '--env', type=str, default='MultiGrid-BlockedUnlockPickup-v0',
        help="MultiGrid environment to use.")
    parser.add_argument(
        '--num-agents', type=int, default=2, help="Number of agents in environment.")
    parser.add_argument(
        '--num-workers', type=int, default=8, help="Number of rollout workers.")
    parser.add_argument(
        '--num-gpus', type=int, default=1, help="Number of GPUs to train on.")
    args = parser.parse_args()

    print(f"Running with following CLI options: {args}")
    print("Training automatically with Ray Tune")

    from multigrid.envs.minigrid import BlockedUnlockPickupEnv
    from multigrid.rllib.env import rllib_multi_agent_env
    from multigrid.wrappers import ActionMaskWrapper, RewardShaping, RewardShaping2
    args.env = rllib_multi_agent_env(
        BlockedUnlockPickupEnv, RewardShaping2.config(scale=1e-4))

    config = algorithm_config(**vars(args))
    stop = {
        #"training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        #"episode_reward_mean": args.stop_reward,
    }

    train(args.algo, config, stop)
