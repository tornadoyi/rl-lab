import numpy as np
import gym
from gym import spaces


class RunwayEnv(gym.Env):
    def __init__(
            self,
            length=1,
            init_pos=0,
            move_success_rate=1.0,
            **kwargs
    ):

        super(RunwayEnv, self).__init__(**kwargs)

        # config
        self._length = length
        self._init_pos = init_pos
        self._move_success_rate = move_success_rate

        # gym
        self.action_space = spaces.Discrete(3)     # 0: wait  1: move left  2: move right
        self.observation_space = spaces.Box(0.0, 1.0, (self._length, ))

        # state
        self._pos = self._init_pos

        # render
        self._render = None


    def reset(self):
        self._pos = self._init_pos
        ob = [0.0] * self._length
        ob[int(self._pos)] = 1.0
        return ob


    def step(self, action):
        # move
        if action != 0 and np.random.rand() < self._move_success_rate:
            if action == 1:
                self._pos = np.clip(self._pos - 1, 0, self._length - 1)
            elif action == 2:
                self._pos = np.clip(self._pos + 1, 0, self._length - 1)

        # save ob
        ob = [0.0] * self._length
        ob[int(self._pos)] = 1.0
        return ob, 0, False, {}


    def render(self, mode='human'):
        from .render import Render
        if self._render is None: self._render = Render(self)
        self._render()


    def render_infos(self): return []

