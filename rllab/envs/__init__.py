
import gym

# import envs
from . import race as _


_make = gym.make
def make(id, **kwargs):
    from .wrapper import Profiling, RewardRatio, atari
    env = _make(id)

    # check env type
    k = kind(env)
    if k == 'atari': env = atari.wrap(env, **kwargs)

    # add profiling wrapper
    env = Profiling(env)
    return env

gym.make = make


def kind(env):
    packs = env.unwrapped.__class__.__module__.split('.')
    if '.'.join(packs[:3]) == 'gym.envs.atari': return 'atari'
    elif '.'.join(packs[:3]) == 'gym.envs.classic_control': return 'classic_control'
    return 'other'