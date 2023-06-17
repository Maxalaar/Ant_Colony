from ray.rllib import MultiAgentEnv

from environment.entity import Entity, EntityType
from environment.ant_agent import AntAgent


class Food(Entity):
    def __init__(self, environment: MultiAgentEnv, number: int):
        super().__init__(id='food_' + str(number), environment=environment, type=EntityType.FOOD, can_stacked=True)

    def is_collected(self, ant_agent: AntAgent):
        ant_agent.collect_food(1)
        self._environment.reposition_entity(self, self._environment.get_random_map_position())

