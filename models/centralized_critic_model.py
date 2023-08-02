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
        self.embedding_dimension_value: int = 128
        self.embedding_dimension_query_key: int = 64
        self.actions_layer = nn.Sequential(
            SlimFC(self.flatten_observation_size, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, self.flatten_action_size),
        )
        self.value_function_layer = nn.Linear(self.flatten_observation_size, 1)
        self.central_value_function_layer = nn.Sequential(
            SlimFC(self.embedding_dimension_value, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, 1),
        )
        self.other_agent_value = nn.Sequential(
            SlimFC(self.flatten_observation_size + self.flatten_action_size_in_batch + 1, 128, activation_fn=nn.LeakyReLU),
            # SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, self.embedding_dimension_value, activation_fn=nn.LeakyReLU),
        )
        self.other_agent_key = nn.Sequential(
            SlimFC(self.flatten_observation_size + self.flatten_action_size_in_batch + 1, 128, activation_fn=nn.LeakyReLU),
            # SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, self.embedding_dimension_query_key, activation_fn=nn.LeakyReLU),
        )
        self.observation_query = nn.Sequential(
            SlimFC(self.flatten_observation_size, 128, activation_fn=nn.LeakyReLU),
            # SlimFC(128, 128, activation_fn=nn.LeakyReLU),
            SlimFC(128, self.embedding_dimension_query_key, activation_fn=nn.LeakyReLU),
        )

    @override(ModelV2)
    def forward(self, input_dictionary, state, seq_lens):
        self.observation = input_dictionary["obs"]
        self.flatten_observation = self.create_flatten_observation(self.observation)
        return self.actions_layer(self.flatten_observation), []

    def central_value_function(self, self_observation, other_agents_observation, other_agents_action):
        self.flatten_self_observation: torch.Tensor = self.create_flatten_dictionary(self_observation)
        self.batch_size: int = self.flatten_self_observation.shape[0]

        self.initialisation_vector = torch.ones([self.batch_size, self.flatten_observation_size], device='cpu')
        self.attention_vector = torch.zeros([self.batch_size, self.embedding_dimension_value], device='cpu')
        self.index_vector = torch.zeros([self.batch_size, 1], device='cpu')
        self.query = self.observation_query(self.initialisation_vector)
        self.number_other_agents: int = len(other_agents_observation.keys())

        self.other_agent_observation: torch.Tensor = torch.cat([self.flatten_self_observation, torch.zeros([self.batch_size, self.flatten_action_size_in_batch], device='cpu'), self.index_vector], 1)
        self.value = self.other_agent_value(self.other_agent_observation)
        self.key = self.other_agent_key(self.other_agent_observation)
        self.attention_vector += nn.functional.scaled_dot_product_attention(self.query, self.key, self.value)

        i: int = 1
        for key in other_agents_observation.keys():
            self.index_vector += i/self.number_other_agents
            self.other_agent_observation: torch.Tensor = torch.cat([self.create_flatten_dictionary(other_agents_observation[key]), other_agents_action[key], self.index_vector], 1)
            self.value = self.other_agent_value(self.other_agent_observation)
            self.key = self.other_agent_key(self.other_agent_observation)
            self.attention_vector += nn.functional.scaled_dot_product_attention(self.query, self.key, self.value)
            i += 1

        # self.attention_vector_and_flatten_self_observation = torch.cat([self.flatten_self_observation, self.attention_vector], 1)

        return torch.reshape(self.central_value_function_layer(self.attention_vector), [-1])

    @override(ModelV2)
    def value_function(self):
        # print('Attention, we are using the non-centralized value function.')
        value_function = self.value_function_layer(self.flatten_observation)
        return torch.reshape(value_function, [-1])
