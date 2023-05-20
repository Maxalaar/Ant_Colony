from enum import Enum


ant_colony_environment_configuration: dict = {
    'map_size': (4, 4),
    'number_agents': 1,
    'number_foods': 1,
    'max_step': 15,
    'ant_agent_configuration': {
        'range_vision': 2,
    },
    'graphic_interface_configuration': {
        'window_size': (800, 800),
        'steps_per_second': 3,
    },
}


class ActionType(Enum):
    NOTHING = 0
    MOVE_UP = 1
    MOVE_RIGHT = 2
    MOVE_DOWN = 3
    MOVE_LEFT = 4
    COLLECT = 5


class EntityType(Enum):
    NIL = -1
    VOID = 0
    ANT_AGENT = 1
    FOOD = 2

