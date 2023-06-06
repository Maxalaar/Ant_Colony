from ray.tune import register_env

from environment.ant_colony_environment import AntColonyEnvironment

# ant_colony_environment_configuration: dict = {
#     'map_size': (4, 4),
#     'number_agents': 1,
#     'number_foods': 1,
#     'max_step': 15,
#     'maximum_pheromone_cell': 1,
#     'pheromone_evaporation': 0.05,
#     'ant_agent_configuration': {
#         'maximum_quantity_pheromone_deposited_agent': 0.5,
#     },
#     'graphic_interface_configuration': {
#         'window_size': (800, 800),
#         'steps_per_second': 3,
#     },
# }

ant_colony_environment_basic_configuration: dict = {
    'map_size': (30, 30),
    'number_agents': 4,
    'number_foods': 5,
    'max_step': 40,
    'maximum_pheromone_cell': 1,
    'pheromone_evaporation': 0.10,
    'ant_agent_configuration': {
        'maximum_quantity_pheromone_deposited_agent': 0.5,
    },
    'graphic_interface_configuration': {
        'render_environment': False,
        'window_size': (800, 800),
        'steps_per_second': 3,
    },
}


def ant_colony_environment_env_creator(args):
    ant_colony_environment_configuration: dict = ant_colony_environment_basic_configuration,
    environment = AntColonyEnvironment(ant_colony_environment_configuration)
    return environment


register_env('ant_colony_environment', ant_colony_environment_env_creator)
