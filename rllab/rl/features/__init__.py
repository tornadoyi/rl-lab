
from gym.spaces import Box

__FEATURES = {}

def register(name):
    global __FEATURES
    def _thunk(func):
        __FEATURES[name] = func
        return func
    return _thunk



def build(ob_space, name=None, **feature_kwargs):
    # specific feature
    input_shape = (None, ) + ob_space.shape
    if name is not None: return __FEATURES[name](input_shape, **feature_kwargs)

    # build from ob
    if not isinstance(ob_space, Box): raise Exception('Invalid ob space {}'.format(ob_space))
    if len(ob_space.shape) == 1:
        return __FEATURES['mlp'](input_shape, **feature_kwargs)
    else:
        raise Exception('Unsupported shape of ob space {}'.format(input_shape))
