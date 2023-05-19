import gym
from ray.tune import register_env

from environment.environment_global_include import ant_colony_environment_configuration
from environment.ant_colony_environment import AntColonyEnvironment


def ant_colony_environment_env_creator(args):
    environment = AntColonyEnvironment(ant_colony_environment_configuration)
    return environment


register_env('ant_colony_environment', ant_colony_environment_env_creator)
