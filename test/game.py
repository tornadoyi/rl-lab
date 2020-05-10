
import gym
from rllab import envs


env = gym.make('ShuffleRun-100m-v0')
while True:
    env.reset()
    while True:
        env.render()
        _, _, t, _ = env.step(env.action_space.sample())
        if t: break