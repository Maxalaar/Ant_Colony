import random
import gym

from ray.rllib.policy.policy import PolicySpec

from environment.ant_agent import AntAgent


def policies_dictionary():
    observation_space = AntAgent.observation_space
    action_space = AntAgent.action_space
    configuration = {
        'model': {
            'custom_model': 'minimal_model',
            # 'custom_model': 'minimal_lstm_model',
        },
    }
    dictionary = {
        'policy1': PolicySpec(observation_space=observation_space, action_space=action_space, config=configuration),
    }

    return dictionary


def select_random_policy(agent_id, episode, worker, **kwargs):
    return random.choice([*policies_dictionary()])
