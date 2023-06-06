import pygame
import time

import numpy as np
import matplotlib.pyplot as plt
from pygame import Color

from ray.rllib import MultiAgentEnv

from environment.environment_global_include import EntityType
from environment.entity import Entity


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

        self._pheromones_average_figure = None
        self._pheromones_average_axis = None
        self._pheromones_average_image = None

    def update(self):
        self.update_entity_map()
        self.update_pheromone_layers()

    def update_entity_map(self):
        cell_size_x: float = float(self._screen.get_size()[0]) / float(self._environment.map_size[0])
        cell_size_y: float = float(self._screen.get_size()[1]) / float(self._environment.map_size[1])
        cell_size: tuple(float, float) = (cell_size_x, cell_size_y)

        self._screen.fill(self._background_color)

        for x in range(self._environment.map_size[0]):
            for y in range(self._environment.map_size[1]):
                if len(self._environment.map[x][y]) > 0:
                    entity: Entity = self._environment.map[x][y][0]
                    if entity.type == EntityType.ANT_AGENT:
                        pygame.draw.rect(self._screen, self._ant_agent_color, pygame.Rect(x * cell_size[0], y * cell_size[1], cell_size[0], cell_size[1]))
                    elif entity.type == EntityType.FOOD:
                        pygame.draw.rect(self._screen, self._food_color, pygame.Rect(x * cell_size[0], y * cell_size[1], cell_size[0], cell_size[1]))

        pygame.display.flip()
        time.sleep(float(1) / float(self._steps_per_second))

    def update_pheromone_layers(self):
        pheromones_average = np.average(self._environment.pheromone_layers, axis=2)
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
