from abc import ABC
from typing import Dict, List
import numpy

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

        self.keys_observation_space: List = self.observation_space.keys()
        self.keys_action_space: List = self.action_space.keys()
        self.flatten_observation_size: int = self.compute_flatten_observation_size()
        self.flatten_action_size: int = self.compute_flatten_action_size()
        self.list_actions_layers: nn.modules.container.ModuleList = None
        self.dictionary_actions_layers: Dict = None
        self.observation: numpy.ndarray = None
        self.flatten_observation: numpy.ndarray = None

        self.flattener = nn.Flatten()

    def compute_flatten_observation_size(self) -> int:
        flatten_observation_size = 0
        for observation_key in self.keys_observation_space:
            sub_observation_size = 1
            for dimension in range(len(self.observation_space[observation_key].shape)):
                sub_observation_size *= self.observation_space[observation_key].shape[dimension]
            flatten_observation_size += sub_observation_size

        return flatten_observation_size

    def compute_flatten_action_size(self) -> int:
        flatten_action_size = 0
        for sub_action_key in self.action_space.keys():
            if type(self.action_space[sub_action_key]) == gymnasium.spaces.Discrete:
                sub_action_size = self.action_space[sub_action_key].n
            elif type(self.action_space[sub_action_key]) == gymnasium.spaces.Box:
                sub_action_size = 2
                for dimension in range(len(self.action_space[sub_action_key].shape)):
                    sub_action_size *= self.action_space[sub_action_key].shape[dimension]
            else:
                raise TypeError('The type of the sub action space is not taken into account.')

            flatten_action_size += sub_action_size

        return flatten_action_size

    def create_flatten_observation(self, observation: gymnasium.spaces.Dict):
        flatten_observation = None
        for observation_key in self.keys_observation_space:
            if flatten_observation is None:
                flatten_observation = self.flattener(observation[observation_key])
            else:
                flatten_observation = torch.cat((flatten_observation, self.flattener(observation[observation_key])), 1)

        return flatten_observation

    def create_actions_layers(self, input_size: int):
        self.list_actions_layers = nn.ModuleList()
        self.dictionary_actions_layers = {}

        for sub_action_key in self.action_space.keys():
            if type(self.action_space[sub_action_key]) == gymnasium.spaces.Discrete:
                sub_action_size = self.action_space[sub_action_key].n
            elif type(self.action_space[sub_action_key]) == gymnasium.spaces.Box:
                sub_action_size = 2
                for dimension in range(len(self.action_space[sub_action_key].shape)):
                    sub_action_size *= self.action_space[sub_action_key].shape[dimension]
            else:
                raise TypeError('The type of the sub action space is not taken into account.')

            self.list_actions_layers.append(nn.Linear(input_size, sub_action_size))
            self.dictionary_actions_layers[sub_action_key] = self.list_actions_layers[-1]

    def create_flatten_dictionary(self, dictionary: gymnasium.spaces.Dict):
        flatten_dictionary = None
        for observation_key in dictionary.keys():
            if flatten_dictionary is None:
                flatten_dictionary = self.flattener(dictionary[observation_key])
            else:
                flatten_dictionary = torch.cat((flatten_dictionary, self.flattener(dictionary[observation_key])), 1)

        return flatten_dictionary
