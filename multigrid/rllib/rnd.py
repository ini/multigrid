import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from .model import TorchModel



class RND(TorchModelV2, nn.Module):
    """
    Basic torch model to use with RLlib.

    Essentially a wrapper for ``ComplexInputNetwork`` that correctly deals with
    ``Dict`` observation spaces.

    For configuration options, see ``rllib/models/catalog.py``.
    """

    def __init__(self, *args, **kwargs):
        """
        See ``TorchModelV2.__init__()``.
        """
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self.target = TorchModel(*args, **kwargs)
        self.target.requires_grad = False
        self.predictor = TorchModel(*args, **kwargs)
        self.model = TorchModel(*args, **kwargs)
        #self.forward = self.model.forward
        self.value_function = self.model.value_function

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def custom_loss(self, policy_loss, loss_inputs):
        print("JOJO", loss_inputs.keys())
        #f = self.target(loss_inputs['new_obs'], [], None)
        #g = self.predictor(loss_inputs['new_obs'], [], None)
        #predictor_loss = (f - g).pow(2).mean()

        return policy_loss