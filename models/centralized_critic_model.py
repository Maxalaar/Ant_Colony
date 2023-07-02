from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from models.reinforcement_learning_model import ReinforcementLearningModel

from ray.rllib.models.torch.misc import SlimFC

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TorchCentralizedCriticModel(ReinforcementLearningModel):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        ReinforcementLearningModel.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        self.actions_layer = nn.Sequential(
            SlimFC(self.flatten_observation_size, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, self.flatten_action_size),
        )
        self.value_function_layer = nn.Linear(self.flatten_observation_size, 1)
        self.central_value_function_layer = nn.Sequential(
            SlimFC(self.flatten_observation_size, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 1),
        )
        # self.central_value_function_multihead_attention = nn.MultiheadAttention(64, 1)

    @override(ModelV2)
    def forward(self, input_dictionary, state, seq_lens):
        self.observation = input_dictionary["obs"]
        self.flatten_observation = self.create_flatten_observation(self.observation)
        return self.actions_layer(self.flatten_observation), []

    def central_value_function(self, self_observation, other_agents_observation, other_agents_action):
        flatten_self_observation = self.create_flatten_dictionary(self_observation)
        # flatten_ather_ants_observation = self.create_flatten_dictionary(ather_ants_observation)
        # centralized_critic_observation = torch.cat(
        #     [
        #         flatten_self_observation,
        #         flatten_ather_ants_observation,
        #         ather_ants_action,
        #     ],
        #     1,
        # )
        return torch.reshape(self.central_value_function_layer(flatten_self_observation), [-1])

    @override(ModelV2)
    def value_function(self):
        # print('Attention, we are using the non-centralized value function.')
        value_function = self.value_function_layer(self.flatten_observation)
        return torch.reshape(value_function, [-1])
