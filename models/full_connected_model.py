import gym
import numpy

from gymnasium import Space

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from environment.ant_agent import AntAgent

torch, nn = try_import_torch()


class FullConnectedModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, number_outputs, model_configuration, name):
        TorchModelV2.__init__(self, observation_space, action_space, number_outputs, model_configuration, name)
        nn.Module.__init__(self)

        self.observation_space: Space = AntAgent.observation_space
        self.action_space: Space = AntAgent.action_space
        self.number_full_connected_layers: int = 5
        self.size_full_connected_layers: int = 64

        self.observation: numpy.ndarray = None
        self.flatten_observation: numpy.ndarray = None
        self.flattener = nn.Flatten()
        self.full_connected_activation = None

        # We calculate the size of the flatten observation
        self.flatten_observation_size = 0
        for sub_observation_key in self.observation_space.keys():
            sub_observation_size = 1
            for dimension in range(len(self.observation_space[sub_observation_key].shape)):
                sub_observation_size *= self.observation_space[sub_observation_key].shape[dimension]
            self.flatten_observation_size += sub_observation_size

        # We create the projection from the flatten observation to the first fully connected layer.
        self.projection_flatten_fully = nn.Linear(self.flatten_observation_size, self.size_full_connected_layers)

        # We create full connected layers
        self.list_full_connected_layers: nn.modules.container.ModuleList = nn.ModuleList()
        for _ in range(self.number_full_connected_layers):
            self.list_full_connected_layers.append(nn.Linear(self.size_full_connected_layers, self.size_full_connected_layers))

        # For each sub action we create a layer of the corresponding size
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

            self.list_actions_layers.append(nn.Linear(self.size_full_connected_layers, sub_action_size))
            self.dictionary_actions_layers[sub_action_key] = self.list_actions_layers[-1]

        self.value_function_layer = nn.Linear(self.size_full_connected_layers, 1)

    def forward(self, input_dict, state, seq_lens):
        self.observation = input_dict["obs"]
        entity_flatten_observation = self.flattener(self.observation['entity'])
        pheromone_flatten_observation = self.flattener(self.observation['pheromone'])
        self.flatten_observation = torch.cat((entity_flatten_observation, pheromone_flatten_observation), 1)

        flatten_observation_projection = self.projection_flatten_fully(self.flatten_observation)
        self.full_connected_activation = self.list_full_connected_layers[0](flatten_observation_projection)
        for i in range(1, len(self.list_full_connected_layers)):
            self.full_connected_activation = self.list_full_connected_layers[1](self.full_connected_activation)

        basic_action = self.dictionary_actions_layers['basic'](self.full_connected_activation)
        pheromone_action = self.dictionary_actions_layers['pheromone'](self.full_connected_activation)

        action = torch.cat((basic_action, pheromone_action), 1)
        return action, []

    def value_function(self):
        value_function = self.value_function_layer(self.full_connected_activation)
        return torch.reshape(value_function, [-1])