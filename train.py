import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

import environment.register_ant_colony_environment
from custom_policies_mapping import policies_dictionary, select_random_policy

ppo_classic_config = (
    PPOConfig()
    .framework('torch')
    .environment(
        'ant_colony_environment',
        disable_env_checking=True,
        render_env=False)
    .multi_agent(policies=policies_dictionary(), policy_mapping_fn=select_random_policy)
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=2,
        evaluation_num_workers=1,
        evaluation_config={
            "render_env": False, }, )
)

if __name__ == '__main__':
    ray.init(local_mode=True)

    algorithm_config: AlgorithmConfig = ppo_classic_config

    algorithm: Algorithm = algorithm_config.build()

    for i in range(1):
        algorithm.train()
