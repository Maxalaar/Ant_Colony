from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from models.reinforcement_learning_model import ReinforcementLearningModel

from ray.rllib.models.torch.misc import SlimFC

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TorchCentralizedCriticFullConnectedModelV2(ReinforcementLearningModel):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        ReinforcementLearningModel.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        self.central_value_function_vector = None
        self.forward_vector = None
        self.actions_layers = None
        self.other_observation_projection = None
        self.self_observation_projection = None
        self.central_value_function_layers = None
        self.other_agent_observation = None
        self.number_other_agents = None
        self.index_vector = None
        self.observation_projection_vector = None
        self.batch_size = None
        self.flatten_self_observation = None
        self.actions_projection = None

        self.activation_function = nn.LeakyReLU
        # Action layers
        self.number_actions_layers = 2
        self.size_actions_layers = 128
        # Observation Projection layers
        self.number_observation_projection_layers = 1
        self.size_observation_projection_layers = 128
        self.observation_projection_size = 128 * 2
        # Central value function layers
        self.number_central_value_function_layers = 4
        self.size_central_value_function_layers = 128 * 2

        self.value_function_layer = nn.Linear(self.flatten_observation_size, 1)
        self.create_actions_layers()
        self.create_observation_projection_layers()
        self.create_central_value_function_layers()


    @override(ModelV2)
    def forward(self, input_dictionary, state, seq_lens):
        self.observation = input_dictionary["obs"]
        self.flatten_observation = self.create_flatten_observation(self.observation)
        return self.actions_projection(self.actions_layers(self.flatten_observation)), []

    def central_value_function(self, self_observation, other_agents_observation, other_agents_action):
        self.flatten_self_observation: torch.Tensor = self.create_flatten_dictionary(self_observation)
        self.forward_vector = self.actions_layers(self.flatten_self_observation)
        self.batch_size: int = self.flatten_self_observation.shape[0]

        self.observation_projection_vector = torch.zeros([self.batch_size, self.observation_projection_size], device='cpu')     # 'cpu' or 'cuda:0'
        self.index_vector = torch.zeros([self.batch_size, 1], device='cpu')
        self.number_other_agents: int = len(other_agents_observation.keys())

        self.observation_projection_vector += self.self_observation_projection(self.flatten_self_observation)

        i: int = 1
        for key in other_agents_observation.keys():
            self.index_vector += i / self.number_other_agents
            self.other_agent_observation: torch.Tensor = torch.cat([self.create_flatten_dictionary(other_agents_observation[key]), other_agents_action[key], self.index_vector], 1)
            self.observation_projection_vector += self.other_observation_projection(self.other_agent_observation)
            i += 1

        self.central_value_function_vector = torch.cat([self.observation_projection_vector, self.forward_vector], 1)

        return torch.reshape(self.central_value_function_layers(self.central_value_function_vector), [-1])

    @override(ModelV2)
    def value_function(self):
        # print('Attention, we are using the non-centralized value function.')
        value_function = self.value_function_layer(self.flatten_observation)
        return torch.reshape(value_function, [-1])

    def create_actions_layers(self):
        self.actions_layers: nn.modules.container.Sequential = nn.Sequential()
        self.actions_layers.add_module('start_actions_layers', SlimFC(self.flatten_observation_size, self.size_actions_layers, activation_fn=self.activation_function))
        for i in range(self.number_actions_layers):
            self.actions_layers.add_module('inter_' + str(i) + '_actions_layers', SlimFC(self.size_actions_layers, self.size_actions_layers, activation_fn=self.activation_function))
        self.actions_projection = SlimFC(self.size_actions_layers, self.flatten_action_size)

    def create_observation_projection_layers(self):
        self.self_observation_projection: nn.modules.container.Sequential = nn.Sequential()
        self.self_observation_projection.add_module('start_self_observation_projection', SlimFC(self.flatten_observation_size, self.size_observation_projection_layers, activation_fn=self.activation_function))
        for i in range(self.number_observation_projection_layers):
            self.self_observation_projection.add_module('inter_' + str(i) + '_self_observation_projection', SlimFC(self.size_observation_projection_layers, self.size_observation_projection_layers,  activation_fn=self.activation_function))
        self.self_observation_projection.add_module('end_self_observation_projection', SlimFC(self.size_observation_projection_layers, self.observation_projection_size))

        self.other_observation_projection: nn.modules.container.Sequential = nn.Sequential()
        self.other_observation_projection.add_module('start_other_observation_projection', SlimFC(self.flatten_observation_size + self.flatten_action_size_in_batch + 1, self.size_observation_projection_layers, activation_fn=self.activation_function))
        for i in range(self.number_observation_projection_layers):
            self.other_observation_projection.add_module('inter_' + str(i) + '_other_observation_projection', SlimFC(self.size_observation_projection_layers, self.size_observation_projection_layers, activation_fn=self.activation_function))
        self.other_observation_projection.add_module('end_other_observation_projection', SlimFC(self.size_observation_projection_layers, self.observation_projection_size))

    def create_central_value_function_layers(self):
        self.central_value_function_layers = nn.Sequential()
        self.central_value_function_layers.add_module('start_central_value_function_layers', SlimFC(self.observation_projection_size + self.size_actions_layers, self.size_central_value_function_layers, activation_fn=self.activation_function))
        for i in range(self.number_central_value_function_layers):
            self.central_value_function_layers.add_module('inter_' + str(i) + '_central_value_function_layers', SlimFC(self.size_central_value_function_layers, self.size_central_value_function_layers, activation_fn=self.activation_function))
        self.central_value_function_layers.add_module('end_central_value_function_layers', SlimFC(self.size_central_value_function_layers, 1))