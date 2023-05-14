import numpy

import gym
from gym.spaces import Space
from ray.rllib import MultiAgentEnv

from environment.global_include import ActionType, EntityType, ant_colony_environment_configuration
from environment.entity import Entity


class AntAgent(Entity):
    observation_space: Space = gym.spaces.MultiDiscrete([[len(EntityType) for i in range(
        ant_colony_environment_configuration['ant_agent_configuration']['range_vision'] * 2 + 1)] for j in range(
        ant_colony_environment_configuration['ant_agent_configuration']['range_vision'] * 2 + 1)])
    action_space: Space = gym.spaces.Discrete(len(ActionType))

    def __init__(self, environment: MultiAgentEnv, ant_agent_configuration: dict[str:str], number: int):
        super().__init__(id='ant_agent_' + str(number), environment=environment, type=EntityType.ANT_AGENT)
        self.range_vision: int = ant_agent_configuration['range_vision']
        self._current_reward: float = 0
        self._current_observation: int = None
        self._is_done: bool = False
        self.number_foods_collected = 0

    def compute_observation(self) -> int:
        self._current_observation = self._environment.get_observation(self)
        return self._current_observation

    def compute_reward(self) -> float:
        self._current_reward = self.number_foods_collected
        self.number_foods_collected = 0
        return self._current_reward

    def compute_is_done(self) -> bool:
        self._is_done = False
        return self._is_done

    def compute_agent_information(self) -> dict:
        return {}

    def compute_action(self, action: int) -> None:
        action = ActionType(action)
        if action == ActionType.NOTHING:
            pass
        elif action == ActionType.MOVE_UP or action == ActionType.MOVE_RIGHT or action == ActionType.MOVE_DOWN or action == ActionType.MOVE_LEFT:
            self._environment.move_entity(self, action)
        elif action == ActionType.COLLECT:
            self._environment.collect_action(self)