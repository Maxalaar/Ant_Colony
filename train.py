import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune

import environment.register_ant_colony_environment  # Required to register the environment
from custom_callbacks import CustomCallbacks
from custom_policies_mapping import policies_dictionary, select_random_policy

ppo_configuration: AlgorithmConfig = (
    PPOConfig()
    .framework('torch')
    .environment(
        'ant_colony_environment',
        disable_env_checking=True,
        render_env=False)
    .multi_agent(policies=policies_dictionary(), policy_mapping_fn=select_random_policy)
    .callbacks(callbacks_class=CustomCallbacks)
    # .evaluation(
    #     evaluation_interval=1,
    #     evaluation_duration=2,
    #     evaluation_num_workers=1,
    #     evaluation_config={
    #         "render_env": False, }, )
)

if __name__ == '__main__':
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=True)

    algorithm_configuration: AlgorithmConfig = ppo_configuration

    algorithm: Algorithm = algorithm_configuration.build()
    # for i in range(1):
    #     algorithm.train()

    tuner = tune.Tuner(
        trainable='PPO',
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            local_dir='./ray_result/',
            stop={'episode_reward_mean': 3.5, 'timesteps_total': 200000, },
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            )
        ),
    )

    tuner.fit()
