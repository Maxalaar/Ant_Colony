import random
from typing import Dict

from gymnasium import Space
from ray.rllib.policy.policy import PolicySpec

from environment.ant_colony_environment import AntColonyEnvironment
from models.centralized_critic_full_connected_model import TorchCentralizedCriticFullConnectedModel
from models.centralized_critic_full_connected_model_V2 import TorchCentralizedCriticFullConnectedModelV2
from models.full_connected_model import FullConnectedModel
from models.centralized_critic_model import TorchCentralizedCriticModel
from models.configuration import *


def policies_dictionary(environment_configuration: Dict):
    observation_space: Space = AntColonyEnvironment.compute_ant_agent_observation_space(environment_configuration)
    action_space: Space = AntColonyEnvironment.compute_ant_agent_action_space(environment_configuration)

    if not environment_configuration['ant_agent_configuration']['use_global_reward']:
        model = FullConnectedModel
        custom_model_config = full_connected_model_basic_configuration
    else:
        # model = TorchCentralizedCriticModel
        # model = TorchCentralizedCriticFullConnectedModel
        model = TorchCentralizedCriticFullConnectedModelV2
        custom_model_config = {}

    configuration = {
        'model': {
            'custom_model': model,
            'custom_model_config': custom_model_config,
        },
    }
    dictionary = {
        'policy0': PolicySpec(observation_space=observation_space, action_space=action_space, config=configuration),
    }

    return dictionary


def select_fix_policy(agent_id, episode, worker, **kwargs):
    return 'policy0'


def select_random_policy(agent_id, episode, worker, **kwargs):
    return random.choice([*policies_dictionary()])
