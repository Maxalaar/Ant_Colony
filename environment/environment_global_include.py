from enum import Enum

ant_range_vision = 2
number_pheromone_layers = 3


class BasicActions(Enum):
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
