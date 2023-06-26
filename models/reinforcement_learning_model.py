from abc import ABC

import gymnasium
from gymnasium import Space

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class ReinforcementLearningModel(ABC, TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, number_outputs, model_configuration, name, **customized_model_kwargs):
        TorchModelV2.__init__(self, observation_space, action_space, number_outputs, model_configuration, name)
        nn.Module.__init__(self)
        self.model_configuration: dict = customized_model_kwargs

        self.observation_space: gymnasium.spaces.Dict = observation_space
        self.action_space: gymnasium.spaces.Dict = action_space

        self.keys_observation_space = self.observation_space.keys()
        self.keys_action_space = self.action_space.keys()

        self.flattener = nn.Flatten()

    def compute_size_flatten_observation(self) -> int:
        flatten_observation_size = 0
        for observation_key in self.keys_observation_space:
            sub_observation_size = 1
            for dimension in range(len(self.observation_space[observation_key].shape)):
                sub_observation_size *= self.observation_space[observation_key].shape[dimension]
            self.flatten_observation_size += sub_observation_size

        return flatten_observation_size

    def flatten_observation(self, observation: gymnasium.spaces.Dict):
        flatten_observation = None
        for observation_key in self.keys_observation_space:
            if flatten_observation is None:
                flatten_observation = self.flattener(observation[observation_key])
            else:
                pass
        return


