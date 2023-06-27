import random
from typing import Dict

from gymnasium import Space
from ray.rllib.policy.policy import PolicySpec

from environment.ant_colony_environment import AntColonyEnvironment
from models.full_connected_model import FullConnectedModel
from models.configuration import *


def policies_dictionary(environment_configuration: Dict):
    observation_space: Space = AntColonyEnvironment.compute_ant_agent_observation_space(environment_configuration)
    action_space: Space = AntColonyEnvironment.compute_ant_agent_action_space(environment_configuration)
    configuration = {
        'model': {
            # 'custom_model': 'minimal_model',
            # 'custom_model': 'full_connected_model',
            # 'custom_model': 'minimal_lstm_model',
            'custom_model': FullConnectedModel,
            'custom_model_config': full_connected_model_basic_configuration,
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
