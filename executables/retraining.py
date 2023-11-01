import ray
from ray.tune import Tuner
from policies.centralized_critic import CentralizedCritic
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPO


if __name__ == '__main__':
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=False)
    # storage_directory: str = '/mnt/5fdcbd7a-f9bb-4644-9e14-f139b450c359/Informatique_Workplace/Workplace_PyCharm/Ant_Colony/ray_result/2023-08-06_17h-31m-16s_Model:TorchCentralizedCriticFullConnectedModel_GlobalReward:True_NumberAgents:5_MapSize:(25, 25)'
    storage_directory: str = '/mnt/5fdcbd7a-f9bb-4644-9e14-f139b450c359/Informatique_Workplace/Workplace_PyCharm/Ant_Colony/ray_result/2023-08-10_00h-28m-21s_Model:TorchCentralizedCriticFullConnectedModelV2_GlobalReward:True_NumberAgents:5_MapSize:(25, 25)'
    tuner: Tuner = Tuner.restore(storage_directory, CentralizedCritic)
    tuner.fit()
