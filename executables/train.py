from datetime import datetime

import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune

import models.register_models    # Required to register models
import environment.register_ant_colony_environment  # Required to register environment
from custom_callbacks import CustomCallbacks
from custom_policies_mapping import policies_dictionary, select_random_policy

ppo_configuration: AlgorithmConfig = (
    PPOConfig()
    .framework('torch')
    .environment(
        env='ant_colony_environment',
        disable_env_checking=True,
        render_env=True,
    )
    .multi_agent(policies=policies_dictionary(), policy_mapping_fn=select_random_policy)
    .callbacks(callbacks_class=CustomCallbacks)
    .evaluation(
        evaluation_interval=10,
        evaluation_duration=4,
        evaluation_num_workers=1,
        evaluation_config={'render_env': True, },
    )
)

if __name__ == '__main__':
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=False)

    algorithm_configuration: AlgorithmConfig = ppo_configuration

    algorithm: Algorithm = algorithm_configuration.build()

    tuner = tune.Tuner(
        trainable='PPO',
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            name='minimal_no_custom_model_extern_mode' + '_' + type(algorithm_configuration).__name__ + '_' + datetime.today().strftime('%Y-%m-%d_%Hh-%Mm-%Ss'),
            # name='trash',
            local_dir='../ray_result/',
            stop={
                'episode_reward_mean': 4,
                'timesteps_total': 400000,
                # 'time_total_s': 60 * 15,
            },
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
