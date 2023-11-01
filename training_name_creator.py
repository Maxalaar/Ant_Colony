from datetime import datetime

from ray.rllib.algorithms.algorithm import AlgorithmConfig


def training_name_creator(algorithm_configuration: AlgorithmConfig, is_trash: bool = False) -> str:
    if is_trash:
        return 'trash'

    separator: str = '_'
    name: str = ''
    name += datetime.today().strftime('%Y-%m-%d_%Hh-%Mm-%Ss') + separator
    name += 'Model:' + algorithm_configuration.multiagent['policies']['policy0'].config['model']['custom_model'].__name__ + separator
    name += 'GlobalReward:' + str(algorithm_configuration.env_config['ant_agent_configuration']['use_global_reward']) + separator
    name += 'NumberAgents:' + str(algorithm_configuration.env_config['number_agents']) + separator
    name += 'MapSize:' + str(algorithm_configuration.env_config['map_size'])

    return name