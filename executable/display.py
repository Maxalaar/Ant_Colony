import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

import environment.register_ant_colony_environment  # Required to register the environment
from custom_callbacks import CustomCallbacks
from custom_policies_mapping import policies_dictionary, select_random_policy

if __name__ == '__main__':
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=True)
    # path_checkpoint: str = ''
    path_checkpoint: str = '/ray_result/PPO/PPO_ant_colony_environment_464cf_00000_0_2023-05-19_13-06-59/checkpoint_000018'

    algorithm: Algorithm = Algorithm.from_checkpoint(path_checkpoint)
    algorithm_config = Algorithm.get_config(algorithm).copy(copy_frozen=False)
    algorithm_config.environment(render_env=True)
    algorithm_config.evaluation(
        evaluation_interval=1,
        evaluation_duration=5,
        evaluation_config={"render_env": True, }, )
    algorithm: Algorithm = algorithm_config.build()
    for i in range(1000):
        print(i)
        algorithm.train()
    algorithm.evaluate()

    # algorithm_config.restore(path_checkpoint)

