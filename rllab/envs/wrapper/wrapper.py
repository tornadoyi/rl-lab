
import gym
from easydict import EasyDict as edict

class Wrapper(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super(Wrapper, self).__init__(*args, **kwargs)
        self.env.wrapper = self

        try:
            self._shared_data = self.unwrapped.shared_data
        except:
            self._shared_data = self.unwrapped.shared_data = edict()


    @property
    def shared_data(self): return self._shared_data


