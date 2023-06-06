import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm

import models.register_models    # Required to register models
from environment.ant_colony_environment import AntColonyEnvironment
from environment.configuration import ant_colony_environment_basic_configuration, ant_colony_environment_complex_configuration

if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init(local_mode=True)

    # path_checkpoint: str = ''
    path_checkpoint: str = '/mnt/5fdcbd7a-f9bb-4644-9e14-f139b450c359/Informatique_Workplace/Workplace_PyCharm/Ant_Colony/ray_result/2023-06-06_11h-56m-06s_complex_5agents_25x25_5foods_PPOConfig/PPO_AntColonyEnvironment_5b1d0_00000_0_2023-06-06_11-56-06/checkpoint_000490'
    environment_configuration: dict = ant_colony_environment_complex_configuration
    environment_configuration['graphic_interface_configuration']['render_environment'] = True

    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    algorithm_config: AlgorithmConfig = Algorithm.get_config(algorithm).copy(copy_frozen=False)
    algorithm_config.evaluation(
        evaluation_duration=1,
        evaluation_config={'render_env': True, },
    )
    algorithm_config.environment(
        env=AntColonyEnvironment,
        env_config=environment_configuration,
    )
    algorithm: Algorithm = algorithm_config.build()
    algorithm.restore(path_checkpoint)
    for i in range(5):
        evaluation_result = algorithm.evaluate()
        print('for the episode ' + str(i) + ' the reward is : ' + str(evaluation_result['evaluation']['episode_reward_mean']))
