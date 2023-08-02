import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm

from environment.configuration import *
from environment.ant_colony_environment import AntColonyEnvironment

if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init(local_mode=True)

    # path_checkpoint: str = '/mnt/5fdcbd7a-f9bb-4644-9e14-f139b450c359/Informatique_Workplace/Workplace_PyCharm/Ant_Colony/ray_result/2023-06-17_00h-16m-45s_complex_5agents_25x25_5foods_120steps_PPOConfig/PPO_AntColonyEnvironment_7b02b_00000_0_2023-06-17_00-16-45/checkpoint_000870'
    path_checkpoint: str = '/mnt/5fdcbd7a-f9bb-4644-9e14-f139b450c359/Informatique_Workplace/Workplace_PyCharm/Ant_Colony/ray_result/trash/CentralizedCritic_AntColonyEnvironment_b20b2_00000_0_2023-07-04_01-02-33/checkpoint_000300'
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    algorithm_config: AlgorithmConfig = Algorithm.get_config(algorithm).copy(copy_frozen=False)

    algorithm_config.evaluation(
        evaluation_duration=1,
        evaluation_config={'render_env': True, },
    )
    algorithm_config.rollouts(
        num_rollout_workers=1,
        num_envs_per_worker=1,
    )

    # algorithm_config.environment(
    #     env=AntColonyEnvironment,
    #     env_config=ant_colony_environment_epic_configuration,
    # )
    # algorithm_config.env_config['graphic_interface_configuration']['render_environment'] = True

    algorithm: Algorithm = algorithm_config.build()
    algorithm.restore(path_checkpoint)
    number_iteration: int = 20
    total_foods_collected: float = 0
    for i in range(number_iteration):
        evaluation_result = algorithm.evaluate()
        # print('For episode ' + str(i) + ' the reward is : ' + str(evaluation_result['evaluation']['episode_reward_mean']))
        foods_collected_in_episode:float = evaluation_result['evaluation']['custom_metrics']['foods_collected_in_episode_mean']
        total_foods_collected += foods_collected_in_episode
        print('For episode ' + str(i) + ' the number of foods collected is : ' + str(foods_collected_in_episode))
    print('The mean of the number of foods collected is : ' + str(total_foods_collected / number_iteration))
