from ray.rllib import MultiAgentEnv

from environment.global_include import EntityType
from environment.entity import Entity
from environment.ant_agent import AntAgent


class Food(Entity):
    def __init__(self, environment: MultiAgentEnv, number: int):
        super().__init__(id='food_' + str(number), environment=environment, type=EntityType.FOOD)

    def is_collected(self, ant_agent: AntAgent):
        ant_agent.number_foods_collected += 1
        self._environment.reposition_entity(self, self._environment.get_random_map_position())

