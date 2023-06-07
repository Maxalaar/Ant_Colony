from typing import Dict, Optional, Union

import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID

from environment.ant_agent import AntAgent


class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        pass

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        pass

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        number_agents: int = 0
        foods_collected_in_episode: int = 0
        average_pheromones_deposited_per_agent: float = 0

        for environment in base_env.envs:
            agents_list = environment.agents_list
            number_agents += len(agents_list)
            for agents in agents_list:
                agents: AntAgent = agents
                foods_collected_in_episode += agents.total_number_foods_collected
                average_pheromones_deposited_per_agent += np.average(agents.total_pheromones_deposited, axis=0)

        average_foods_collected_per_agents = float(foods_collected_in_episode)/float(number_agents)
        average_pheromones_deposited_per_agent /= float(number_agents)

        episode.custom_metrics['foods_collected_in_episode'] = foods_collected_in_episode
        episode.custom_metrics['average_foods_collected_per_agents_in_episode'] = average_foods_collected_per_agents
        episode.custom_metrics['average_pheromones_deposited_per_agent_in_episode'] = average_pheromones_deposited_per_agent
