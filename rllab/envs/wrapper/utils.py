
from easydict import EasyDict as edict

def get_user_data(env):
    env = env.unwrapped
    ud = getattr(env, '__userdata__', None)
    if ud is None:
        ud = edict()
        env.__userdata__ = ud
        env.__class__.userdata = property(lambda self: self.__userdata__)
    return ud