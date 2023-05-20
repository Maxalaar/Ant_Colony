import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm

import models.register_models    # Required to register models
import environment.register_ant_colony_environment  # Required to register the environment

if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init(local_mode=True)

    # path_checkpoint: str = ''
    path_checkpoint: str = '/mnt/5fdcbd7a-f9bb-4644-9e14-f139b450c359/Informatique_Workplace/Workplace_PyCharm/Ant_Colony/ray_result/minimal_model_PPOConfig_2023-05-20_14h-47m-00s/PPO_ant_colony_environment_6a1f8_00000_0_2023-05-20_14-47-01/checkpoint_000050'

    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    algorithm_config: AlgorithmConfig = Algorithm.get_config(algorithm).copy(copy_frozen=False)
    algorithm_config.evaluation(
        evaluation_duration=1,
        evaluation_config={"render_env": True, }, )
    algorithm: Algorithm = algorithm_config.build()
    algorithm.restore(path_checkpoint)
    for i in range(5):
        evaluation_result = algorithm.evaluate()
        print('for the episode ' + str(i) + ' the reward is : ' + str(evaluation_result['evaluation']['episode_reward_mean']))
