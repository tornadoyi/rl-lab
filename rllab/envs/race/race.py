

from .runway import RunwayEnv


class RaceEnv(RunwayEnv):

    def __init__(self, score_mode='normal',  **kwargs):
        # config
        self._score_mode = score_mode

        super(RaceEnv, self).__init__(**kwargs)


    def on_step(self, action):
        super(RaceEnv, self).on_step(action)

        # check terminated
        if self._terminated: return

        # reward
        if self._score_mode == 'sparse':
            reward = 0.0 if self._pos < self._length - 1 else 1.0
        elif self._score_mode == 'guide':
            if action == 0: reward = 0.0
            elif action == 1: reward = -1.0
            else: reward = 1.0
        else:
            reward = -1.0 if self._pos < self._length - 1 else 1.0

        self.store_reward(reward)

        # terminate
        if self._pos >= self._length - 1: self._terminated = True




import numpy as np
from gym.envs.registration import register


register(
    id='Race-100m-easy-v0',
    entry_point='rllab.envs.race:RaceEnv',
    kwargs={'length': 100, 'move_success_rate': 1.0, 'score_mode': 'guide'},
    max_episode_steps=np.inf,
    reward_threshold=1.0,
)

register(
    id='Race-100m-medium-v0',
    entry_point='rllab.envs.race:RaceEnv',
    kwargs={'length': 100, 'move_success_rate': 0.9, 'score_mode': 'normal', 'max_steps_reward': -1.0},
    max_episode_steps=int(3 * 100 / 0.9),
    reward_threshold=1.0,
)

register(
    id='Race-100m-hard-v0',
    entry_point='rllab.envs.race:RaceEnv',
    kwargs={'length': 100, 'move_success_rate': 0.8, 'score_mode': 'sparse', 'max_steps_reward': -1.0},
    max_episode_steps=int(2 * 100 / 0.8),
    reward_threshold=1.0,
)