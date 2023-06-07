from typing import Dict
import numpy


import gym
from gym.spaces import Space, Box, Discrete
from ray.rllib import MultiAgentEnv

from environment.environment_global_include import BasicActions, EntityType, ant_range_vision, number_pheromone_layers
from environment.entity import Entity


class AntAgent(Entity):
    observation_space: Space = gym.spaces.Dict({
        'entity': Box(low=-len(EntityType), high=len(EntityType), shape=(ant_range_vision * 2 + 1, ant_range_vision * 2 + 1)),
        'pheromone': Box(low=0, high=numpy.inf, shape=(ant_range_vision * 2 + 1, ant_range_vision * 2 + 1, number_pheromone_layers)),
    })
    action_space: Space = gym.spaces.Dict({
        'basic': Discrete(len(BasicActions)),
        'pheromone': Box(low=0, high=1, shape=(number_pheromone_layers,)),
    })

    def __init__(self, environment: MultiAgentEnv, ant_agent_configuration: Dict[str, str], number: int):
        super().__init__(id='ant_agent_' + str(number), environment=environment, type=EntityType.ANT_AGENT)
        self.range_vision: int = ant_range_vision
        self.maximum_quantity_pheromone_deposited: float = ant_agent_configuration['maximum_quantity_pheromone_deposited_agent']
        self._current_reward: float = 0
        self._current_observation: Space = None
        self._is_done: bool = False
        self._number_foods_collected_timestep: int = 0

        self.total_number_foods_collected: int = 0
        self.total_pheromones_deposited: numpy.ndarray = numpy.zeros(shape=(number_pheromone_layers,))

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
