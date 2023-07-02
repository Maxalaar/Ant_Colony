import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm

from environment.configuration import *
from environment.ant_colony_environment import AntColonyEnvironment

if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init(local_mode=True)

    # path_checkpoint: str = ''
    # path_checkpoint: str = '/mnt/5fdcbd7a-f9bb-4644-9e14-f139b450c359/Informatique_Workplace/Workplace_PyCharm/Ant_Colony/ray_result/2023-06-27_00h-05m-29s_complex_5agents_25x25_5foods_120steps_PPOConfig/PPO_AntColonyEnvironment_8fd53_00000_0_2023-06-27_00-05-29/checkpoint_000840'
    path_checkpoint: str = '/mnt/5fdcbd7a-f9bb-4644-9e14-f139b450c359/Informatique_Workplace/Workplace_PyCharm/Ant_Colony/ray_result/trash/CentralizedCritic_AntColonyEnvironment_1e4b2_00000_0_2023-07-01_09-51-42/checkpoint_000200'
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    algorithm_config: AlgorithmConfig = Algorithm.get_config(algorithm).copy(copy_frozen=False)

    algorithm_config.evaluation(
        evaluation_duration=1,
        evaluation_config={'render_env': True, },
    )

    # algorithm_config.environment(
    #     env=AntColonyEnvironment,
    #     env_config=ant_colony_environment_epic_configuration,
    # )
    algorithm_config.env_config['graphic_interface_configuration']['render_environment'] = True

    algorithm: Algorithm = algorithm_config.build()
    algorithm.restore(path_checkpoint)
    for i in range(5):
        evaluation_result = algorithm.evaluate()
        print('For episode ' + str(i) + ' the reward is : ' + str(evaluation_result['evaluation']['episode_reward_mean']))
