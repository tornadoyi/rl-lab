from .runway import RunwayEnv

class ShuttleRunEnv(RunwayEnv):
    def __init__(self, score_mode='normal', **kwargs):
        # config
        self._score_mode = score_mode

        super(ShuttleRunEnv, self).__init__(**kwargs)

    def reset(self):
        ob = super(ShuttleRunEnv, self).reset()
        self._half_finish = False
        return [float(self._half_finish)] + ob



    def step(self, action):
        ob, r, d, info = super(ShuttleRunEnv, self).step(action)

        # reward
        r = 0
        if self._score_mode == 'sparse':
            if self._half_finish and self._pos == 0:
                r = 1.0
        elif self._score_mode == 'guide':
            if not self._half_finish:
                if action == 1: r = -1.0
                elif action == 2: r = 1.0
            else:
                if action == 1: r = 1.0
                elif action == 2: r = -1.0
        else:
            r = -1.0
            if not self._half_finish:
                if self._pos == self._length - 1: r = 1.0
            else:
                if self._pos == 0: r = 1.0

        # half finish and terminal
        if self._pos == self._length - 1: self._half_finish = True
        if self._half_finish and self._pos == 0: d = True

        # observation
        ob = [float(self._half_finish)] + ob
        return ob, r, d, info


    def render_infos(self):
        return ['half goal: {}'.format('ok' if self._half_finish else 'no')]



import numpy as np
from gym.envs.registration import register

register(
    id='ShuttleRun-100m-easy-v0',
    entry_point='rllab.envs.race:ShuttleRunEnv',
    kwargs={'length': 100, 'move_success_rate': 1.0, 'score_mode': 'guide'},
    max_episode_steps=np.inf,
    reward_threshold=1.0,
)

register(
    id='ShuttleRun-100m-medium-v0',
    entry_point='rllab.envs.race:ShuttleRunEnv',
    kwargs={'length': 100, 'move_success_rate': 0.9, 'score_mode': 'normal'},
    max_episode_steps=int(3 * 200 / 0.9),
    reward_threshold=1.0,
)

register(
    id='ShuttleRun-100m-hard-v0',
    entry_point='rllab.envs.race:ShuttleRunEnv',
    kwargs={'length': 100, 'move_success_rate': 0.8, 'score_mode': 'sparse'},
    max_episode_steps=int(2 * 200 / 0.8),
    reward_threshold=1.0,
)