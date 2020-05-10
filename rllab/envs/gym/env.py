import numpy as np
import gym

class Env(gym.Env):
    def __init__(self):
        self._render = None

    @property
    def steps(self): return self._steps

    @property
    def max_steps(self): return np.inf if self.spec is None else self.spec.max_episode_steps

    @property
    def observations(self): return self._observations

    @property
    def actions(self): return self._actions

    @property
    def rewards(self): return self._rewards

    @property
    def terminated(self): return self._terminated

    @property
    def total_reward(self): return self._total_reward


    def reset(self):
        self._steps = 0
        self._observations = []
        self._actions = []
        self._rewards = []
        self._terminated = False
        self._total_reward = 0

        self._observations.append(None)
        self.on_reset()

        return self._observations[self._steps]


    def step(self, action):
        # check
        if self._terminated: raise Exception('call step on terminated env')

        # record action
        self._steps += 1
        self._actions.append(action)

        # check terminal
        if self.spec is None: return False
        self._terminated = self._steps >= self.spec.max_episode_steps

        # on step
        self._observations.append(None)
        self._rewards.append(0)
        self.on_step(action)

        return self._observations[self._steps], self._rewards[self._steps-1], self._terminated, {}


    def render(self, mode='human'):
        if self._render is None: self.on_create_render()
        self.on_render()


    def store_reward(self, reward):
        if self.spec is not None: reward *= self.spec.reward_threshold
        self._rewards[self._steps-1] = reward
        self._total_reward += reward


    def store_observation(self, ob):
        self._observations[self._steps] = ob


    # events
    def on_reset(self): raise NotImplementedError('on_reset should be implemented')

    def on_step(self, action): raise NotImplementedError('on_step should be implemented')

    def on_create_render(self, mode='human'): raise NotImplementedError('on_create_render should be implemented')

    def on_render(self, mode='human'): self._render()