import random
from typing import Dict, Tuple, List, Optional
import numpy

import gym
from ray.rllib import MultiAgentEnv
from gym import Space
from ray.rllib.utils.typing import MultiAgentDict

from environment.environment_global_include import BasicActions, EntityType, number_pheromone_layers, ant_range_vision
from environment.entity import Entity
from environment.ant_agent import AntAgent
from environment.food import Food
from environment.graphic_interface import GraphicInterface


class AntColonyEnvironment(MultiAgentEnv):
    def __init__(self, environment_configuration: Dict):
        super().__init__()
        self.map_size: Tuple[int, int] = environment_configuration['map_size']
        self.number_agents: int = environment_configuration['number_agents']
        self.number_foods: int = environment_configuration['number_foods']
        self.ant_agent_configuration = environment_configuration['ant_agent_configuration']
        self.number_pheromone_layers: int = number_pheromone_layers
        self.max_step: int = environment_configuration['max_step']
        self.graphic_interface_configuration: Dict = environment_configuration['graphic_interface_configuration']
        self.graphic_interface: GraphicInterface = None
        self.map: List[List[List[Entity]]] = None
        self.pheromone_layers: numpy.ndarray = None

        self._spaces_in_preferred_format: bool = True
        self._obs_space_in_preferred_format: bool = True
        self._action_space_in_preferred_format: bool = True

        self.current_step: int = None
        self.next_agent_number_id: int = None
        self.next_food_number_id: int = None
        self.agent_ids: set = None
        self.agents_dictionary: Dict[str, AntAgent] = None
        self.entities_position_dictionary: Dict[Entity:Tuple[int, int]] = None
        self.agents_list: list = None

        self.observations_dictionary: Dict[str, Space] = None
        self.rewards_dictionary: Dict[str, float] = None
        self.is_done_dictionary: Dict[str, bool] = None
        self.is_truncated_dictionary: Dict[str, bool] = None
        self.agents_information_dictionary: Dict[str, Dict[str, str]] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[
        Dict[str, Space], Dict[str, Dict[str, str]]]:
        super().reset(seed=seed)
        self.action_space: gym.spaces.Dict() = {}
        self.observation_space: gym.spaces.Dict() = {}

        self.current_step = 0
        self.next_agent_number_id = 0
        self.next_food_number_id = 0
        self.agent_ids = set()
        self.agents_list: List[AntAgent] = []
        self.agents_dictionary = {}
        self.entities_position_dictionary = {}
        self.clear_information_dictionaries()

        self.create_map()
        self.create_pheromone_layers()

        for _ in range(self.number_agents):
            self.add_agent()

        for _ in range(self.number_foods):
            self.add_food()

        return self.observations_dictionary, self.agents_information_dictionary

    def step(self, action_dictionary: Dict) -> Tuple[
        Dict[str, Space], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, str]]]:
        self.clear_information_dictionaries()
        self.current_step += 1

        for agent_id, action in action_dictionary.items():
            self.agents_dictionary[agent_id].compute_action(action)

        for agent_id, _ in action_dictionary.items():
            self.update_information_dictionaries(agent_id)

        self.compute_simulation_is_done()
        self.compute_simulation_is_truncated()
        # self.render()

        return self.observations_dictionary, self.rewards_dictionary, self.is_done_dictionary, self.is_truncated_dictionary, self.agents_information_dictionary

    def render(self):
        if self.graphic_interface is None:
            self.graphic_interface = GraphicInterface(self, self.graphic_interface_configuration)
        self.graphic_interface.update()
        return True

    def clear_information_dictionaries(self):
        self.observations_dictionary = {}
        self.rewards_dictionary = {}
        self.is_done_dictionary = {}
        self.is_truncated_dictionary = {}
        self.agents_information_dictionary = {}

    def update_information_dictionaries(self, agent_id: str):
        agent: AntAgent = self.agents_dictionary[agent_id]
        self.observations_dictionary[agent.id] = agent.compute_observation()
        self.rewards_dictionary[agent.id] = agent.compute_reward()
        self.is_done_dictionary[agent.id] = agent.compute_is_done()
        self.is_truncated_dictionary[agent_id] = agent.compute_is_truncated()
        self.agents_information_dictionary[agent.id] = agent.compute_agent_information()

    def compute_simulation_is_done(self):
        simulation_is_done = True

        # if an agent still has to act, the environment is not stopped
        for agent in self.agents_list:
            if not agent.compute_is_done():
                simulation_is_done = False
                break

        # if the current step is equal or higher than the max step, the environment is stopped
        if self.current_step >= self.max_step:
            simulation_is_done = True

        self.is_done_dictionary['__all__'] = simulation_is_done

    def compute_simulation_is_truncated(self):
        self.is_truncated_dictionary['__all__'] = False

    def add_agent(self):
        agent: AntAgent = AntAgent(self, self.ant_agent_configuration, self.next_agent_number_id)
        self.next_agent_number_id += 1
        self.agent_ids.add(agent.id)
        self.agents_list.append(agent)
        self.agents_dictionary[agent.id] = agent
        self.observation_space[agent.id] = agent.observation_space
        self.action_space[agent.id] = agent.action_space
        self.add_entity_map(agent, self.get_random_map_position())

        self.update_information_dictionaries(agent.id)

    def add_food(self):
        resource: Food = Food(self, self.next_food_number_id)
        self.next_food_number_id += 1
        self.add_entity_map(resource, self.get_random_map_position())

    def create_map(self):
        self.map = [[[] for i in range(self.map_size[0])] for j in range(self.map_size[1])]

    def create_pheromone_layers(self):
        self.pheromone_layers = numpy.zeros((self.map_size[0], self.map_size[1], self.number_pheromone_layers))

    def add_entity_map(self, entity: Entity, position: Tuple[int, int]):
        self.map[position[0]][position[1]].append(entity)
        self.entities_position_dictionary[entity] = position

    def get_random_map_position(self) -> Tuple[int, int]:
        position_x = random.randint(0, self.map_size[0] - 1)
        position_y = random.randint(0, self.map_size[1] - 1)
        return [position_x, position_y]

    def position_is_on_map(self, position: Tuple[int, int]) -> bool:
        return 0 <= position[0] < self.map_size[0] and 0 <= position[1] < self.map_size[1]

    def reposition_entity(self, entity: Entity, new_entity_position: Tuple[int, int]):
        if self.position_is_on_map(new_entity_position):
            entity_position = self.get_entity_position(entity)
            self.map[entity_position[0]][entity_position[1]].remove(entity)
            self.map[new_entity_position[0]][new_entity_position[1]].append(entity)
            self.entities_position_dictionary[entity] = new_entity_position
        else:
            raise TypeError('Attempt to reposition an entity outside the map')

    def get_entity_position(self, entity: Entity) -> Tuple[int, int]:
        if entity not in self.entities_position_dictionary:
            raise TypeError('Attempt to retrieve the position of an entity that is not in the dictionary')
        return self.entities_position_dictionary[entity]

    def get_entity_observation(self, ant_agent: AntAgent) -> numpy.ndarray:
        observation = numpy.zeros(shape=(ant_range_vision * 2 + 1, ant_range_vision * 2 + 1))
        range_vision = ant_agent.range_vision
        ant_position = self.get_entity_position(ant_agent)

        for i in range(-range_vision, range_vision):
            for j in range(-range_vision, range_vision):
                observation_cell = (ant_position[0] + i, ant_position[1] + j)
                value_observation_cell: EntityType = None

                if self.position_is_on_map(observation_cell):
                    cell_map: List[Entity] = self.map[observation_cell[0]][observation_cell[1]]
                    if len(cell_map) > 0:
                        value_observation_cell = cell_map[0].type
                    else:
                        value_observation_cell = EntityType.VOID
                else:
                    value_observation_cell = EntityType.NIL

                if value_observation_cell is None:
                    raise TypeError('An element in the map is not interpretable for observation')

                observation[i][j] = value_observation_cell.value

        return observation

    def get_pheromone_observation(self, ant_agent: AntAgent) -> numpy.ndarray:
        observation = numpy.zeros(shape=(ant_range_vision * 2 + 1, ant_range_vision * 2 + 1, self.number_pheromone_layers))
        range_vision = ant_agent.range_vision
        ant_position = self.get_entity_position(ant_agent)
        nil_vector = numpy.ones(shape=(self.number_pheromone_layers,)) * 0

        for i in range(-range_vision, range_vision):
            for j in range(-range_vision, range_vision):
                observation_cell = (ant_position[0] + i, ant_position[1] + j)
                if self.position_is_on_map(observation_cell):
                    observation[i][j] = self.pheromone_layers[observation_cell[0]][observation_cell[1]]
                else:
                    observation[i][j] = nil_vector

        return observation

    def collect_action(self, ant_agent: AntAgent):
        agent_position: Tuple[int, int] = self.entities_position_dictionary[ant_agent]
        for entity in self.map[agent_position[0]][agent_position[1]]:
            if entity.type == EntityType.FOOD:
                entity: Food = entity
                entity.is_collected(ant_agent)

    def move_entity(self, entity: Entity, move: BasicActions):
        entity_position = self.get_entity_position(entity)
        new_entity_position: Tuple[int, int] = None

        if move == BasicActions.MOVE_UP:
            new_entity_position = (entity_position[0], entity_position[1] - 1)
        if move == BasicActions.MOVE_RIGHT:
            new_entity_position = (entity_position[0] + 1, entity_position[1])
        if move == BasicActions.MOVE_DOWN:
            new_entity_position = (entity_position[0], entity_position[1] + 1)
        if move == BasicActions.MOVE_LEFT:
            new_entity_position = (entity_position[0] - 1, entity_position[1])

        if self.position_is_on_map(new_entity_position):
            self.reposition_entity(entity, new_entity_position)

    def apply_pheromones(self, entity: Entity, pheromone_vector: numpy.ndarray):
        entity_position: Tuple[int, int] = self.get_entity_position(entity)
        self.pheromone_layers[entity_position[0]][entity_position[1]] += pheromone_vector
