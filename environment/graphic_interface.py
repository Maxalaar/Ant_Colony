from typing import Tuple
import time
import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame import Color

from ray.rllib import MultiAgentEnv

from environment.entity import Entity, EntityType
from environment.ant_agent import AntAgent


class GraphicInterface:
    def __init__(self, environment: MultiAgentEnv, graphic_interface_configuration: dict):
        pygame.init()
        self._environment: MultiAgentEnv = environment
        self._screen: pygame.Surface = pygame.display.set_mode(
            [graphic_interface_configuration['window_size'][0], graphic_interface_configuration['window_size'][1]])
        pygame.display.set_caption('Entity Map')
        self._steps_per_second = graphic_interface_configuration['steps_per_second']

        self._background_color = Color(255, 255, 255, 255)
        self._ant_agent_color: Color = Color(255, 0, 0, 255)
        self._food_color: Color = Color(0, 255, 0, 255)
        self._field_view_color: Color = Color(0, 0, 255, 50)
        self._cell_size: tuple(float, float) = None

        self._pheromones_average_figure = None
        self._pheromones_average_axis = None
        self._pheromones_average_image = None

    def update(self):
        self.update_entity_map()
        self.update_pheromone_layers()
        time.sleep(float(1) / float(self._steps_per_second))

    def update_entity_map(self):
        self._screen.fill(self._background_color)

        self.compute_cell_size()
        self.display_entities()
        self.display_field_view()

        pygame.display.flip()

    def compute_cell_size(self):
        cell_size_x: float = float(self._screen.get_size()[0]) / float(self._environment.map_size[0])
        cell_size_y: float = float(self._screen.get_size()[1]) / float(self._environment.map_size[1])
        self._cell_size = (cell_size_x, cell_size_y)

    def display_entities(self):
        for x in range(self._environment.map_size[0]):
            for y in range(self._environment.map_size[1]):
                if len(self._environment.entity_map[x][y]) > 0:
                    entity: Entity = self._environment.entity_map[x][y][0]
                    entity_position = self.index_to_position((x, y))
                    if entity.type == EntityType.ANT_AGENT:
                        self.draw_rect(size=self._cell_size, position=entity_position, color=self._ant_agent_color)
                    elif entity.type == EntityType.FOOD:
                        self.draw_rect(size=self._cell_size, position=entity_position, color=self._food_color)

    def display_field_view(self):
        for ant_agent in self._environment.agents_list:
            ant_agent: AntAgent = ant_agent
            ant_agent_range_vision: int = ant_agent.entities_range_vision
            ant_agent_position: Tuple[int, int] = self._environment.entities_position_dictionary[ant_agent]
            for x in range(-ant_agent_range_vision, ant_agent_range_vision + 1):
                for y in range(-ant_agent_range_vision, ant_agent_range_vision + 1):
                    coloring_cell_index = (ant_agent_position[0] + x, ant_agent_position[1] + y)
                    self.draw_rect(size=self._cell_size, position=self.index_to_position(coloring_cell_index), color=self._field_view_color)

    def draw_rect(self, size: Tuple[int, int], position: Tuple[int, int], color: Color):
        surface = pygame.Surface(size)
        surface.set_alpha(color.a)
        surface.fill(color)
        self._screen.blit(surface, position)

    def index_to_position(self, index: Tuple[int, int]) -> Tuple[int, int]:
        return index[0] * self._cell_size[0], index[1] * self._cell_size[1]

    def update_pheromone_layers(self):
        pheromones_average = np.average(self._environment.pheromone_map, axis=2)
        pheromones_average = np.rot90(pheromones_average)
        pheromones_average = np.flip(pheromones_average, 0)

        if self._pheromones_average_figure is None:
            plt.ion()
            self._pheromones_average_figure = plt.figure()
            self._pheromones_average_figure.canvas.set_window_title('Pheromone Map')
            self._pheromones_average_axis = self._pheromones_average_figure.add_subplot(111)
            self._pheromones_average_image = self._pheromones_average_axis.imshow(pheromones_average)

        self._pheromones_average_image.set_array(pheromones_average)
        self._pheromones_average_figure.canvas.draw()
        self._pheromones_average_figure.canvas.flush_events()

    def close(self):
        pygame.quit()
