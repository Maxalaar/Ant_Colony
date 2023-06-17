from ray.rllib import MultiAgentEnv

from environment.environment_global_include import EntityType


class Entity:
    def __init__(self, id: str, environment: MultiAgentEnv, type: EntityType, can_stacked):
        self.id: str = id
        self._environment: MultiAgentEnv = environment
        self.type: EntityType = type
        self.can_stacked: bool = can_stacked
