from __future__ import annotations

from gymnasium import spaces
from ray.rllib.models.tf.complex_input_net import (
    ComplexInputNetwork as TFComplexInputNetwork
)
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.complex_input_net import (
    ComplexInputNetwork as TorchComplexInputNetwork
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import TensorType, try_import_torch
from typing import Any

torch, nn = try_import_torch()
from torch import Tensor



def to_sample_batch(
    input_dict: SampleBatch | dict[str, TensorType],
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
    batch = SampleBatch(input_dict)
    batch.update(input_dict)
    batch.update(kwargs)

    # Process states
    for i in range(len(state)):
        batch[f'state_in_{i}'] = state[i]

    # Process sequence lengths
    if seq_lens is not None:
        batch[SampleBatch.SEQ_LENS] = seq_lens

    return batch



class TFModel(TFModelV2):
    """
    Basic tensorflow model to use with RLlib.

    Essentially a wrapper for ``ComplexInputNetwork`` that correctly deals with
    ``Dict`` observation spaces.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs):
        """
        See ``TFModelV2.__init__()``.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = TFComplexInputNetwork(
            obs_space, action_space, num_outputs, model_config, name)
        self.forward = self.model.forward
        self.value_function = self.model.value_function


class TorchModel(TorchModelV2, nn.Module):
    """
    Basic torch model to use with RLlib.

    Essentially a wrapper for ``ComplexInputNetwork`` that correctly deals with
    ``Dict`` observation spaces.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
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
        value_input_space : spaces.Space, optional
            Space for value function inputs (if different from `obs_space`)
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Action
        self.action_model = TorchComplexInputNetwork(
            obs_space, action_space, num_outputs, model_config, None)

        # Value
        self.value_model = TorchComplexInputNetwork(
            value_input_space or obs_space, action_space, 1, model_config, None)

        self._model_input: SampleBatch = None

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
        self._model_input = to_sample_batch(input_dict, state, seq_lens)
        return self.action_model(input_dict, state, seq_lens)

    def value_function(self, value_input: Any = None) -> TensorType:
        """
        Returns the value function output for the most recent forward pass.

        Parameters
        ----------
        value_input : Any, optional
            Value function input (if different from obs, e.g. for centralized critic)
        """
        if value_input is None:
            if self.value_model.obs_space is not self.obs_space:
                return self.action_model.value_function() # dummy output

        value_input = value_input or self._model_input[SampleBatch.OBS]
        self._model_input[SampleBatch.OBS] = value_input
        value_out, _ = self.value_model(self._model_input)
        return value_out.flatten()


class TorchLSTMModel(TorchModel):
    """
    Torch LSTM model to use with RLlib.

    Processes observations with a ``ComplexInputNetwork`` and then passes
    the output through an LSTM layer.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs):
        """
        See ``TorchModel.__init__()``.
        """
        nn.Module.__init__(self)
        lstm_cell_size = self.model_config.get('lstm_cell_size', 256)
        super().__init__(
            obs_space,
            action_space,
            lstm_cell_size,
            model_config,
            name,
            **kwargs
        )

        # LSTM
        self.lstm = nn.LSTM(
            lstm_cell_size,
            lstm_cell_size,
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

    def get_initial_state(self) -> list[Tensor]:
        """
        Get initial state for the LSTM.
        """
        return [
            torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
            torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
        ]
