
import gym

def make(**kwargs):
    name = kwargs['id']
    return gym.make(name).unwrapped