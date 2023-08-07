import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune import Tuner
from ray import air, tune

from policies.centralized_critic import CentralizedCritic
from custom_callbacks import CustomCallbacks
from environment.configuration import *
from policies.custom_policies_mapping import *
from environment.ant_colony_environment import AntColonyEnvironment
from training_name_creator import training_name_creator

ppo_configuration: AlgorithmConfig = PPOConfig()
ppo_configuration.framework('torch')
ppo_configuration.environment(
    env=AntColonyEnvironment,
    env_config=ant_colony_environment_complex_configuration,
    disable_env_checking=True,
    render_env=False,
)
ppo_configuration.multi_agent(policies=policies_dictionary(ppo_configuration.env_config), policy_mapping_fn=select_fix_policy)
ppo_configuration.callbacks(callbacks_class=CustomCallbacks)
ppo_configuration.resources(
    num_gpus=0,
    num_cpus_per_worker=1,
    num_gpus_per_worker=0,
    num_learner_workers=0,
    num_cpus_per_learner_worker=0,
    num_gpus_per_learner_worker=0,
)
ppo_configuration.rollouts(
    num_rollout_workers=2,
    num_envs_per_worker=1,
)
ppo_configuration.evaluation(
    evaluation_num_workers=1,
    evaluation_interval=20,
    evaluation_duration=100,
    evaluation_config={'render_env': False, },
)
ppo_configuration.experimental(
    _disable_preprocessor_api=True,
)
# ppo_configuration.training(
#     train_batch_size=2048,
#     sgd_minibatch_size=2048,
#     num_sgd_iter=4,
# )

if __name__ == '__main__':
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=False)

    algorithm_configuration: AlgorithmConfig = ppo_configuration

    algorithm: Algorithm = algorithm_configuration.build()

    if not algorithm_configuration.env_config['ant_agent_configuration']['use_global_reward']:
        trainable = PPO
    else:
        trainable = CentralizedCritic

    tuner: Tuner = tune.Tuner(
        trainable=trainable,
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            name=training_name_creator(algorithm_configuration),
            storage_path='../ray_result/',
            # stop={
            #     # 'episode_reward_mean': 5,
            #     # 'timesteps_total': 1000000,
            #     'time_total_s': 60 * 60 * 24,
            # },
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=20,
                checkpoint_at_end=True,
            )
        ),
    )

    tuner.fit()
