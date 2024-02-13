from __future__ import annotations

import torch
import torch.nn as nn

from gymnasium import spaces
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import TensorType
from torch import Tensor
from typing import Any



### Helper Functions

def to_sample_batch(
    input_dict: SampleBatch | dict,
    state: list[TensorType] = [],
    seq_lens: TensorType | None = None,
    **kwargs) -> SampleBatch:
    """
    Create a ``SampleBatch`` with the given data.

    Parameters
    ----------
    input_dict : SampleBatch or dict
        Batch of data
    state : list[TensorType]
        List of state tensors
    seq_lens : TensorType or None
        1-D tensor holding input sequence lengths
    """
    batch = SampleBatch(input_dict, **kwargs)

    # Process states
    for i in range(len(state)):
        batch[f'state_in_{i}'] = state[i]

    # Process sequence lengths
    if seq_lens is not None:
        batch[SampleBatch.SEQ_LENS] = seq_lens

    return batch



### Models

class CustomModel(TorchModelV2, nn.Module):
    """
    Base class for custom models to use with RLlib.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.

    Example
    -------
    Use a 2-layer MLP with 64 hidden units and tanh activations:

    >>> config.training(
    ...     model={
    ...         'custom_model': CustomModel,
    ...         'custom_model_config': dict(model_kwargs),
    ...         'fcnet_hiddens': [64, 64],
    ...         'fcnet_activation': 'tanh',
    ...     }
    ... )
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        base_model_cls: type[TorchModelV2] = ComplexInputNetwork,
        value_input_space: spaces.Space = None,
        **kwargs):
        """
        Parameters
        ----------
        obs_space : spaces.Space
            Observation space
        action_space : spaces.Space
            Action space
        num_outputs : int
            Number of outputs
        model_config : dict
            Model configuration dictionary
        name : str
            Model name
        base_model_cls : type[TorchModelV2], optional
            Base model class to wrap around (e.g. ComplexInputNetwork)
        value_input_space : spaces.Space, optional
            Space for value function inputs (e.g. for centralized critic)
        """
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)

        # Action
        self.action_model: TorchModelV2 = base_model_cls(
            obs_space, action_space, num_outputs, model_config, None)

        # Value
        self.value_model: TorchModelV2 = base_model_cls(
            value_input_space or obs_space, action_space, 1, model_config, None)

        self._value_input: SampleBatch | None = None

    def forward(
        self,
        input_dict: dict[str, Tensor],
        state: list[Tensor],
        seq_lens: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        Forward pass of the model.

        Parameters
        ----------
        input_dict : dict[str, Tensor]
            Dictionary of input tensors
        state : list[Tensor]
            List of state tensors
        seq_lens : Tensor
            1-D tensor holding input sequence lengths
        """
        self._value_input = to_sample_batch(input_dict, state, seq_lens)
        return self.action_model(input_dict, state, seq_lens)

    def value_function(self, value_input: Any = None) -> Tensor:
        """
        Return the value function output for the most recent forward pass.

        Parameters
        ----------
        value_input : Any, optional
            Value function input (if different from obs, e.g. for centralized critic)
        """
        assert self._value_input is not None, "must call `forward()` first."

        # If using a custom value input space, but no custom value input is provided,
        # return dummy outputs (to be overwritten in postprocessing)
        if self.value_model.obs_space is not self.obs_space and value_input is None:
            batch_size = self._value_input['obs_flat'].shape[0]
            device = self._value_input['obs_flat'].device
            return torch.zeros(batch_size, device=device) # dummy output

        # Use custom value input if provided
        if value_input is not None:
            self._value_input[SampleBatch.OBS] = value_input

        value_out, _ = self.value_model(self._value_input)
        return value_out.flatten()

    def custom_loss(
        self,
        policy_loss: Tensor,
        loss_inputs: SampleBatch) -> Tensor | list[Tensor]:
        """
        Override to customize the loss function used to optimize this model.

        Parameters
        ----------
        policy_loss : Tensor
            Policy loss
        loss_inputs : SampleBatch
            Batch of data used to compute the loss
        """
        return self.action_model.custom_loss(policy_loss, loss_inputs)


class CustomLSTMModel(CustomModel):
    """
    Custom LSTM model to use with RLlib.

    Processes observations with a base model and then passes
    the output through an LSTM layer.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.

    Example
    -------
    Use an CNN-LSTM model with 3 convolutional layers and 1 LSTM layer:

    >>> config.training(
    ...     model={
    ...         'custom_model': CustomModel,
    ...         'custom_model_config': dict(model_kwargs),
    ...         'conv_filters': [
                    [16, [3, 3], 1],
                    [32, [3, 3], 1],
                    [64, [3, 3], 1],
                ],
    ...         'lstm_cell_size': 64,
    ...         'max_seq_len': 4,
    ...     }
    ... )
    """

    def __init__(self, *args, **kwargs):
        """
        See ``CustomModel.__init__()``.
        """
        obs_space, action_space, num_outputs, model_config, name = args
        super().__init__(obs_space, action_space, None, model_config, name, **kwargs)

        # LSTM
        self.lstm = nn.LSTM(
            self.action_model.num_outputs,
            model_config.get('lstm_cell_size', MODEL_DEFAULTS['lstm_cell_size']),
            batch_first=True,
            num_layers=1,
        )

        # Head
        self.head = nn.Linear(self.lstm.hidden_size, num_outputs)

    def forward(
        self,
        input_dict: dict[str, Tensor],
        state: list[Tensor],
        seq_lens: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        Forward pass of the model.

        Parameters
        ----------
        input_dict : dict[str, Tensor]
            Dictionary of input tensors
        state : list[Tensor]
            List of state tensors
        seq_lens : Tensor
            1-D tensor holding input sequence lengths
        """
        # Base
        x, _ = super().forward(input_dict, state, seq_lens)

        # LSTM
        x = add_time_dimension(x, seq_lens=seq_lens, framework='torch', time_major=False)
        h = state[0].transpose(0, 1).contiguous()
        c = state[1].transpose(0, 1).contiguous()
        x, [h, c] = self.lstm(x, [h, c])

        # Output
        logits = self.head(x.reshape(-1, self.lstm.hidden_size))
        return logits, [h.transpose(0, 1), c.transpose(0, 1)]

    def get_initial_state(self) -> list[torch.Tensor]:
        """
        Get initial state for the LSTM.
        """
        return [
            torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
            torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
        ]
