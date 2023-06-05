from ray.rllib.models.tf.complex_input_net import (
    ComplexInputNetwork as TFComplexInputNetwork
)
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.complex_input_net import (
    ComplexInputNetwork as TorchComplexInputNetwork
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()



class TFModel(TFModelV2):
    """
    Basic tensorflow model to use with RLlib.

    Essentially a wrapper for ``ComplexInputNetwork`` that correctly deals with
    ``Dict`` observation spaces.

    For configuration options, see ``rllib/models/catalog.py``.
    """

    def __init__(self, *args, **kwargs):
        """
        See ``TFModelV2.__init__()``.
        """
        super().__init__(*args, **kwargs)
        self.model = TFComplexInputNetwork(*args, **kwargs)
        self.forward = self.model.forward
        self.value_function = self.model.value_function


from gymnasium import spaces
class TorchModel(TorchModelV2, nn.Module):
    """
    Basic torch model to use with RLlib.

    Essentially a wrapper for ``ComplexInputNetwork`` that correctly deals with
    ``Dict`` observation spaces.

    For configuration options, see ``rllib/models/catalog.py``.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        joint_obs_space: spaces.Space = None,
        **kwargs):
        """
        See ``TorchModelV2.__init__()``.
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.model = TorchComplexInputNetwork(
            obs_space, action_space, num_outputs, model_config, name)
        self.forward = self.model.forward
        self.value_function = self.model.value_function

        self.central_critic = TorchComplexInputNetwork(
            joint_obs_space, action_space, 1, model_config, name)

    def central_value_function(self, obs, other_agent_obs, other_agent_actions):
        print("CVF", type(obs))
        obs = torch.cat([obs, other_agent_obs], dim=1)
        inputs = {'obs': obs}
        # print("CVF", other_agent_obs.shape, other_agent_actions.shape)
        out, _ = self.central_critic(inputs, [], None)
        return torch.reshape(out, [-1])
