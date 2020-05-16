
from gym.envs import registration
from .wrapper import *

class EnvRegistry(registration.EnvRegistry):

    def make(self, *args, **kwargs):
        env = super(EnvRegistry, self).make(*args, **kwargs)

        env = Profiling(env)
        return env




registration.registry = EnvRegistry()