from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from models.reinforcement_learning_model import ReinforcementLearningModel

from ray.rllib.models.torch.misc import SlimFC

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TorchCentralizedCriticFullConnectedModel(ReinforcementLearningModel):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        ReinforcementLearningModel.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        self.observation_projection_size = 128

        self.value_function_layer = nn.Linear(self.flatten_observation_size, 1)

        self.actions_layers: nn.modules.container.Sequential = nn.Sequential(
            SlimFC(self.flatten_observation_size, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, self.flatten_action_size),
        )

        self.self_observation_projection: nn.modules.container.Sequential = nn.Sequential(
            SlimFC(self.flatten_observation_size, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, self.observation_projection_size),
        )

        self.other_observation_projection: nn.modules.container.Sequential = nn.Sequential(
            SlimFC(self.flatten_observation_size + self.flatten_action_size_in_batch + 1, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, self.observation_projection_size),
        )

        self.central_value_function_layers = nn.Sequential(
            SlimFC(self.observation_projection_size, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 1),
        )

    @override(ModelV2)
    def forward(self, input_dictionary, state, seq_lens):
        self.observation = input_dictionary["obs"]
        self.flatten_observation = self.create_flatten_observation(self.observation)
        return self.actions_layers(self.flatten_observation), []

    def central_value_function(self, self_observation, other_agents_observation, other_agents_action):
        self.flatten_self_observation: torch.Tensor = self.create_flatten_dictionary(self_observation)
        self.batch_size: int = self.flatten_self_observation.shape[0]

        self.observation_projection_vector = torch.zeros([self.batch_size, self.observation_projection_size], device='cpu')
        self.index_vector = torch.zeros([self.batch_size, 1], device='cpu')
        self.number_other_agents: int = len(other_agents_observation.keys())

        self.observation_projection_vector += self.self_observation_projection(self.flatten_self_observation)

        i: int = 1
        for key in other_agents_observation.keys():
            self.index_vector += i/self.number_other_agents
            self.other_agent_observation: torch.Tensor = torch.cat([self.create_flatten_dictionary(other_agents_observation[key]), other_agents_action[key], self.index_vector], 1)
            self.observation_projection_vector += self.other_observation_projection(self.other_agent_observation)
            i += 1

        return torch.reshape(self.central_value_function_layers(self.observation_projection_vector), [-1])

    @override(ModelV2)
    def value_function(self):
        # print('Attention, we are using the non-centralized value function.')
        value_function = self.value_function_layer(self.flatten_observation)
        return torch.reshape(value_function, [-1])
