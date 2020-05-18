import gym

class Profiling(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super(Profiling, self).__init__(*args, **kwargs)
        self.unwrapped.num_resets = 0

    def reset(self):
        ob = super(Profiling, self).reset()
        self.unwrapped.num_resets += 1
        self.unwrapped.steps = 0
        self.unwrapped.observation = ob
        self.unwrapped.action = None
        self.unwrapped.reward = None
        self.unwrapped.total_reward = 0
        self.unwrapped.info = None
        self.unwrapped.done = False
        return ob


    def step(self, action):
        ob, r, d, info = super(Profiling, self).step(action)
        self.unwrapped.steps += 1
        self.unwrapped.action = action
        self.unwrapped.reward = r
        self.unwrapped.observation = ob
        self.unwrapped.info = info
        self.unwrapped.total_reward += r
        self.unwrapped.done = d
        return ob, r, d, info



