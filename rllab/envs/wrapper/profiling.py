

from .wrapper import Wrapper

class Profiling(Wrapper):
    def __init__(self, *args, **kwargs):
        super(Profiling, self).__init__(*args, **kwargs)
        self._num_resets = 0

    @property
    def steps(self): return self._steps

    @property
    def observations(self): return self._observations

    @property
    def actions(self): return self._actions

    @property
    def rewards(self): return self._rewards

    @property
    def done(self): return self._done

    @property
    def infos(self): return self._infos

    @property
    def num_resets(self): return self._num_resets

    def reset(self):
        ob = super(Profiling, self).reset()
        self._num_resets += 1
        self._steps = 0
        self._observations = [ob]
        self._actions = []
        self._rewards = []
        self._total_reward = 0
        self._infos = []
        self._done = False
        return ob


    def step(self, action):
        ob, r, d, info = super(Profiling, self).step(action)
        self._steps += 1
        self._actions.append(action)
        self._rewards.append(r)
        self._observations.append(ob)
        self._infos.append(info)
        self._total_reward += r
        self._done = d

        # save to shared data
        self.shared_data.steps = self._steps
        self.shared_data.total_reward = self._total_reward

        return ob, r, d, info



