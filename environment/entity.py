from enum import Enum

from ray.rllib import MultiAgentEnv


class EntityType(Enum):
    NIL = -1
    VOID = 0
    ANT_AGENT = 1
    FOOD = 2


class Entity:
    def __init__(self, id: str, environment: MultiAgentEnv, type: EntityType, can_stacked):
        self.id: str = id
        self._environment: MultiAgentEnv = environment
        self.type: EntityType = type
        self.can_stacked: bool = can_stacked
