import gym
import numpy

from gymnasium import Space

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from environment.ant_agent import AntAgent

torch, nn = try_import_torch()


class MinimalModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, number_outputs, model_configuration, name):
        TorchModelV2.__init__(self, observation_space, action_space, number_outputs, model_configuration, name)
        nn.Module.__init__(self)

        # get_preprocessor(action_space)(action_space).size
        self.observation_space: Space = AntAgent.observation_space
        self.action_space: Space = AntAgent.action_space

        self.observation: numpy.ndarray = None
        self.flatten_observation: numpy.ndarray = None
        self.flattener = nn.Flatten()

        # We calculate the size of the flatten observation
        self.flatten_observation_size = 0
        for sub_observation_key in self.observation_space.keys():
            sub_observation_size = 1
            for dimension in range(len(self.observation_space[sub_observation_key].shape)):
                sub_observation_size *= self.observation_space[sub_observation_key].shape[dimension]
            self.flatten_observation_size += sub_observation_size

        # for each sub action we create a layer of the corresponding size
        self.dictionary_actions_layers: dict = {}
        self.list_actions_layers: nn.modules.container.ModuleList = nn.ModuleList()
        for sub_action_key in action_space.keys():
            if type(self.action_space[sub_action_key]) == gym.spaces.Discrete:
                sub_action_size = self.action_space[sub_action_key].n
            elif type(self.action_space[sub_action_key]) == gym.spaces.Box:
                sub_action_size = 2
                for dimension in range(len(self.action_space[sub_action_key].shape)):
                    sub_action_size *= self.action_space[sub_action_key].shape[dimension]
            else:
                raise TypeError('The type of the sub action space is not taken into account.')

            self.list_actions_layers.append(nn.Linear(self.flatten_observation_size, sub_action_size))
            self.dictionary_actions_layers[sub_action_key] = self.list_actions_layers[-1]

        self.value_function_layer = nn.Linear(self.flatten_observation_size, 1)

    def forward(self, input_dict, state, seq_lens):
        self.observation = input_dict["obs"]
        entity_flatten_observation = self.flattener(self.observation['entity'])
        pheromone_flatten_observation = self.flattener(self.observation['pheromone'])
        self.flatten_observation = torch.cat((entity_flatten_observation, pheromone_flatten_observation), 1)

        basic_action = self.dictionary_actions_layers['basic'](self.flatten_observation)
        pheromone_action = self.dictionary_actions_layers['pheromone'](self.flatten_observation)

        action = torch.cat((basic_action, pheromone_action), 1)
        return action, []

    def value_function(self):
        value_function = self.value_function_layer(self.flatten_observation)
        return torch.reshape(value_function, [-1])
