
import gym
from rllab import envs

import pygame

env = gym.make('ShuttleRun-100m-hard-v0')
#env = gym.make('Race-100m-hard-v0')
env.reset()

while True:
    env.render()

    keys = pygame.key.get_pressed()

    action = -1
    if keys[pygame.K_SPACE]: action = 0
    elif keys[pygame.K_LEFT]: action = 1
    elif keys[pygame.K_RIGHT]: action = 2

    if action > 0:
        _, _, t, _ = env.step(action)
        if t: break
