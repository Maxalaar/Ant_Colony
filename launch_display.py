from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

import environment.register_ant_colony_environment  # Required to register the environment
from custom_callbacks import CustomCallbacks
from custom_policies_mapping import policies_dictionary, select_random_policy

if __name__ == '__main__':
    path_checkpoint: str = '/mnt/5fdcbd7a-f9bb-4644-9e14-f139b450c359/Informatique_Workplace/Workplace_PyCharm/Ant_Colony/ray_result/PPO/PPO_ant_colony_environment_86881_00000_0_2023-05-18_11-22-35/checkpoint_000100'
    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    algorithm_config = Algorithm.get_config(algorithm).copy(copy_frozen=False)
    algorithm_config.evaluation(
        evaluation_duration=1,
        evaluation_num_workers=1,
        evaluation_config={
            "render_env": True, }, )
    algorithm = algorithm_config.build()
    algorithm.train()

    # ppo_configuration = PPOConfig().build()
    # ppo_configuration.restore(path_checkpoint)

