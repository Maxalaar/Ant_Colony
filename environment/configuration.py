ant_colony_environment_basic_configuration: dict = {
    'map_size': (4, 4),
    'number_agents': 1,
    'number_foods': 1,
    'max_step': 15,
    'maximum_pheromone_cell': 1,
    'pheromone_evaporation': 0.05,
    'ant_agent_configuration': {
        'maximum_quantity_pheromone_deposited_agent': 0.5,
    },
    'graphic_interface_configuration': {
        'render_environment': False,
        'window_size': (800, 800),
        'steps_per_second': 5,
    },
}

ant_colony_environment_complex_configuration: dict = dict(ant_colony_environment_basic_configuration.copy(),
  **{
      'map_size': (25, 25),
      'max_step': 120,
      'number_agents': 5,
      'number_foods': 5,
  }
)

ant_colony_environment_epic_configuration: dict = dict(ant_colony_environment_basic_configuration.copy(),
  **{
      'map_size': (50, 50),
      'max_step': 300,
      'number_agents': 15,
      'number_foods': 10,
      'pheromone_evaporation': 0.05,
      'ant_agent_configuration': {
          'maximum_quantity_pheromone_deposited_agent': 3,
      },
  }
)
