from __future__ import annotations

import numpy as np

from gymnasium import spaces
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import MultiAgentEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import AgentID, PolicyID



def get_critic_input_space(env: MultiAgentEnv) -> spaces.Dict:
    """
    Get gymnasium space for centralized critic input.

    Parameters
    ----------
    env : MultiAgentEnv
        Instance of multi-agent environment
    """
    return spaces.Dict({
        str(agent_id): spaces.Dict({
            SampleBatch.OBS: env.observation_space[agent_id],
            SampleBatch.ACTIONS: env.action_space[agent_id],
        }) for agent_id in sorted(env.get_agent_ids())
    })


class CentralizedCriticCallbacks(DefaultCallbacks):
    """
    Callbacks for training with a centralized critic.

    To use, set the 'callbacks' key in the algorithm config to this class.

    Requires a custom model with a modified `value_function()` method
    that takes in an optional `value_input` argument.

    Example
    -------
    >>> config.callbacks(CentralizedCriticCallbacks)
    >>> config.training(
    ...     model={
    ...         'custom_model': CustomModel,
    ...         'custom_model_config': {'value_input_space': get_critic_input_space(env))
    ...     }
    ... )
    """

    def on_postprocess_trajectory(
        self,
        worker: RolloutWorker,
        episode: Episode | EpisodeV2,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: dict[AgentID, tuple[Policy, SampleBatch]],
        **kwargs):
        """
        Overwrite value function outputs with centralized critic.
        """
        # Get observations and actions from all agents as input to critic
        value_input = {}
        for _agent_id, (_policy_id, _policy, _batch) in original_batches.items():
            # Get observations
            obs = restore_original_dimensions(
                _batch[SampleBatch.OBS], _policy.model.obs_space, tensorlib='numpy')

            # Get actions
            action_space = _policy.model.action_space
            action_encoder = ModelCatalog.get_preprocessor_for_space(action_space)
            action = np.stack([
                action_encoder.transform(a)
                for a in _batch[SampleBatch.ACTIONS]], axis=0)
            action *= (_agent_id != agent_id) # zero out agent's own actions

            # Store in value function input
            value_input[str(_agent_id)] = {
                SampleBatch.OBS: obs,
                SampleBatch.ACTIONS: action,
            }

        # Overwrite value function outputs
        policy = policies[policy_id]
        if policy.framework == 'torch':
            value_input = convert_to_torch_tensor(value_input, policy.device)
            vf_preds = policy.model.value_function(value_input=value_input)
            postprocessed_batch[SampleBatch.VF_PREDS] = vf_preds.cpu().detach().numpy()
        else:
            vf_preds = policy.model.value_function(value_input=value_input)
            postprocessed_batch[SampleBatch.VF_PREDS] = vf_preds

        # Postprocess the batch (again)
        policy.postprocess_trajectory(postprocessed_batch)
