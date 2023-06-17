from typing import Dict
from enum import Enum
import numpy

import gymnasium
from gymnasium.spaces import Space, Box, Discrete
from ray.rllib import MultiAgentEnv

from environment.entity import Entity, EntityType


class BasicActions(Enum):
    NOTHING = 0
    MOVE_UP = 1
    MOVE_RIGHT = 2
    MOVE_DOWN = 3
    MOVE_LEFT = 4
    COLLECT = 5


class AntAgent(Entity):
    def __init__(self, environment: MultiAgentEnv, ant_agent_configuration: Dict[str, str], number: int):
        super().__init__(id='ant_agent_' + str(number), environment=environment, type=EntityType.ANT_AGENT, can_stacked=False)
        self.entities_range_vision: int = ant_agent_configuration['entities_range_vision']
        self.pheromones_range_vision: int = ant_agent_configuration['pheromones_range_vision']
        self.number_pheromone_layers: int = ant_agent_configuration['number_pheromone_layers']

        self.observation_space: Space = gymnasium.spaces.Dict({
            'entity': Box(low=-len(EntityType), high=len(EntityType), shape=(self.entities_range_vision * 2 + 1, self.entities_range_vision * 2 + 1)),
            'pheromone': Box(low=0, high=numpy.inf, shape=(self.pheromones_range_vision * 2 + 1, self.pheromones_range_vision * 2 + 1, self.number_pheromone_layers)),
        })
        self.action_space: Space = gymnasium.spaces.Dict({
            'basic': Discrete(len(BasicActions)),
            'pheromone': Box(low=0, high=1, shape=(self.number_pheromone_layers,)),
        })

        self.maximum_quantity_pheromone_deposited: float = ant_agent_configuration['maximum_quantity_pheromone_deposited_agent']
        self._current_reward: float = 0
        self._current_observation: Space = None
        self._is_done: bool = False
        self._number_foods_collected_timestep: int = 0

        self.total_number_foods_collected: int = 0
        self.total_pheromones_deposited: numpy.ndarray = numpy.zeros(shape=(self.number_pheromone_layers,))

    def compute_observation(self) -> Space:
        entity_observation = self._environment.get_entity_observation(self)
        pheromone_observation = self._environment.get_pheromone_observation(self)
        self._current_observation = {
            'entity': entity_observation,
            'pheromone': pheromone_observation,
        }
        return self._current_observation

    def compute_reward(self) -> float:
        self._current_reward = self._number_foods_collected_timestep
        self._number_foods_collected_timestep = 0
        return self._current_reward

    def compute_is_done(self) -> bool:
        self._is_done = False
        return self._is_done

    def compute_agent_information(self) -> dict:
        agent_information: dict = {
            'number_foods_collected_total': self.total_number_foods_collected,
        }
        return agent_information

    def compute_action(self, action: Dict) -> None:
        pheromone_action: numpy.ndarray = action['pheromone']
        basic_action: int = action['basic']

        # We apply the pheromone ant actions
        self.total_pheromones_deposited += pheromone_action
        self._environment.apply_pheromones(self, pheromone_action * self.maximum_quantity_pheromone_deposited)

        # We apply the basic ant actions
        basic_action: BasicActions = BasicActions(basic_action)
        if basic_action == BasicActions.NOTHING:
            pass
        elif basic_action == BasicActions.MOVE_UP or basic_action == BasicActions.MOVE_RIGHT or basic_action == BasicActions.MOVE_DOWN or basic_action == BasicActions.MOVE_LEFT:
            self._environment.move_entity(self, basic_action)
        elif basic_action == BasicActions.COLLECT:
            self._environment.collect_action(self)

    def compute_is_truncated(self) -> bool:
        return False

    def collect_food(self, number_food: int):
        self._number_foods_collected_timestep += number_food
        self.total_number_foods_collected += number_food
