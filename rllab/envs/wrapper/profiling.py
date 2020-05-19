import gym
from .utils import get_user_data

class Profiling(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super(Profiling, self).__init__(*args, **kwargs)
        self.ud = get_user_data(self)
        self.ud.num_resets = 0

    def reset(self):
        ob = super(Profiling, self).reset()
        self.ud.num_resets += 1
        self.ud.steps = 0
        self.ud.observation = ob
        self.ud.action = None
        self.ud.reward = None
        self.ud.total_reward = 0
        self.ud.info = None
        self.ud.done = False
        return ob


    def step(self, action):
        ob, r, d, info = super(Profiling, self).step(action)
        self.ud.steps += 1
        self.ud.action = action
        self.ud.reward = r
        self.ud.observation = ob
        self.ud.info = info
        self.ud.total_reward += r
        self.ud.done = d
        return ob, r, d, info



