
import gym

# import envs
from . import race as _


_gym_make = gym.make
def make(*args, **kwargs):
    from .wrapper import Profiling, RewardRatio
    env = _gym_make(*args, **kwargs)
    if env.spec.reward_threshold is not None: env = RewardRatio(env)
    env = Profiling(env)
    return env

gym.make = make