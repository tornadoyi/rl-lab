from .runway import RunwayEnv

class ShuttleRunEnv(RunwayEnv):
    def __init__(self, score_mode='normal', **kwargs):
        # config
        self._score_mode = score_mode

        super(ShuttleRunEnv, self).__init__(**kwargs)

    def on_reset(self):
        super(ShuttleRunEnv, self).on_reset()
        self._half_finish = False

        ob = [0] + self._observations[-1]
        self.store_observation(ob)


    def on_step(self, action):
        super(ShuttleRunEnv, self).on_step(action)

        # save ob
        ob = [1] + self._observations[-1] if self._half_finish else [0] + self._observations[-1]
        self.store_observation(ob)

        # check terminated
        if self._terminated: return

        # reward
        reward = 0
        if self._score_mode == 'sparse':
            if self._half_finish and self._pos == 0:
                reward = 1.0
        elif self._score_mode == 'guide':
            if not self._half_finish:
                if action == 1: reward = -1.0
                elif action == 2: reward = 1.0
            else:
                if action == 1: reward = 1.0
                elif action == 2: reward = -1.0
        else:
            reward = -1.0
            if not self._half_finish:
                if self._pos == self._length - 1: reward = 1.0
            else:
                if self._pos == 0: reward = 1.0

        self.store_reward(reward)

        # half finish and terminal
        if self._pos == self._length - 1: self._half_finish = True
        if self._half_finish and self._pos == 0: self._terminated = True







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
    kwargs={'length': 100, 'move_success_rate': 0.9, 'score_mode': 'normal', 'max_steps_reward': -1.0},
    max_episode_steps=int(3 * 200 / 0.9),
    reward_threshold=1.0,
)

register(
    id='ShuttleRun-100m-hard-v0',
    entry_point='rllab.envs.race:ShuttleRunEnv',
    kwargs={'length': 100, 'move_success_rate': 0.8, 'score_mode': 'sparse', 'max_steps_reward': -1.0},
    max_episode_steps=int(2 * 200 / 0.8),
    reward_threshold=1.0,
)