from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()


class MinimalModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, number_outputs, model_configuration, name):
        TorchModelV2.__init__(self, observation_space, action_space, number_outputs, model_configuration, name)
        nn.Module.__init__(self)
        self.input = None
        self.flatten_input = None
        observation_size = get_preprocessor(observation_space)(observation_space).size
        action_size = get_preprocessor(action_space)(action_space).size
        self.flattener = nn.Flatten()
        self.action_layer = nn.Linear(observation_size, action_size)
        self.value_function_layer = nn.Linear(observation_size, 1)

    def forward(self, input_dict, state, seq_lens):
        self.input = input_dict["obs"]
        self.flatten_input = self.flattener(self.input)
        action = self.action_layer(self.flatten_input)
        return action, []

    def value_function(self):
        value_function = self.value_function_layer(self.flatten_input)
        return torch.reshape(value_function, [-1])
