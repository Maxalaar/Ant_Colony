from datetime import datetime

import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune

from environment.configuration import *
from policies.custom_policies_mapping import *
from environment.ant_colony_environment import AntColonyEnvironment

ppo_configuration: AlgorithmConfig = PPOConfig()
ppo_configuration.framework('torch')
ppo_configuration.environment(
    env=AntColonyEnvironment,
    env_config=ant_colony_environment_complex_configuration,
    disable_env_checking=True,
    render_env=False,
)
ppo_configuration.multi_agent(policies=policies_dictionary(ppo_configuration.env_config), policy_mapping_fn=select_fix_policy)
# ppo_configuration.callbacks(callbacks_class=CustomCallbacks)
ppo_configuration.resources(
    num_gpus=0,
    num_cpus_per_worker=1,
    num_learner_workers=0,
)
ppo_configuration.rollouts(
    num_rollout_workers=2,
    num_envs_per_worker=1,
)
ppo_configuration.evaluation(
    evaluation_interval=20,
    evaluation_duration=10,
    evaluation_num_workers=1,
    evaluation_config={'render_env': False, },
)
ppo_configuration.experimental(
    _disable_preprocessor_api=True,
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
            name=datetime.today().strftime('%Y-%m-%d_%Hh-%Mm-%Ss') + '_' + 'complex_5agents_25x25_5foods_120steps' + '_' + type(algorithm_configuration).__name__,
            # name='trash',
            storage_path='../ray_result/',
            stop={
                # 'episode_reward_mean': 5,
                # 'timesteps_total': 1000000,
                'time_total_s': 60 * 60 * 7,
            },
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=100,
                checkpoint_at_end=True,
            )
        ),
    )

    tuner.fit()
