import numpy as np
import gym

class Env(gym.Env):
    def __init__(self, max_steps_reward=0.0):
        self._max_steps_reward = max_steps_reward
        self._render = None

    @property
    def steps(self): return self._steps

    @property
    def max_steps(self): return self.spec.max_episode_steps

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

        self.on_reset()

        return self._observations[self._steps]


    def step(self, action):
        # check
        if self._terminated: raise Exception('call step on terminated env')

        # record action
        self._steps += 1
        self._actions.append(action)

        # check terminal
        self._terminated = self._steps >= self.spec.max_episode_steps

        # on step
        self.store_reward(self._max_steps_reward if self._terminated else 0.0)
        self.on_step(action)

        return self._observations[self._steps], self._rewards[self._steps-1], self._terminated, {}


    def render(self, mode='human'):
        if self._render is None: self.on_create_render()
        self.on_render()


    def store_reward(self, reward):
        reward *= self.spec.reward_threshold
        if len(self._rewards) == self._steps-1:
            self._rewards.append(reward)
        elif len(self._rewards) == self._steps:
            self._rewards[self._steps-1] = reward
        else:
            raise Exception('invalid rewards number')
        self._total_reward += reward


    def store_observation(self, ob):
        if len(self._observations) == self._steps:
            self._observations.append(ob)
        elif len(self._observations) == self._steps + 1:
            self._observations[self._steps] = ob
        else:
            raise Exception('invalid observations number')


    # events
    def on_reset(self): raise NotImplementedError('on_reset should be implemented')

    def on_step(self, action): raise NotImplementedError('on_step should be implemented')

    def on_create_render(self, mode='human'): raise NotImplementedError('on_create_render should be implemented')

    def on_render(self, mode='human'): self._render()