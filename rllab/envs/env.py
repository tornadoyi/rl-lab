
import gym
from easydict import EasyDict as edict

class Env(gym.Env):

    @property
    def shared_data(self): return self.__shared_data

    def reset(self):
        self.__shared_data = edict()
        return None







