
import os

_ROOT_PATH = os.path.expanduser('~/.rl-lab')

_MODEL_PATH = 'models'

_PROFILING_PATH = 'profiling'

_EXPERIMENT_PATH = None


def root_path(): return _ROOT_PATH

def set_root_path(p):
    global _ROOT_PATH
    _ROOT_PATH = p

def model_path():
    if os.path.isabs(_MODEL_PATH): return _MODEL_PATH
    return os.path.join(experiment_path(), _MODEL_PATH)

def set_model_path(p):
    global _MODEL_PATH
    _MODEL_PATH = p

def profiling_path():
    if os.path.isabs(_PROFILING_PATH): return _PROFILING_PATH
    return os.path.join(experiment_path(), _PROFILING_PATH)

def set_profiling_path(p):
    global _PROFILING_PATH
    _PROFILING_PATH = p

def module_path(): return os.path.join(root_path(), which_module())


def experiment_path():
    if _EXPERIMENT_PATH is None: raise Exception('empty experiment path')
    if os.path.isabs(_EXPERIMENT_PATH): return _EXPERIMENT_PATH
    return os.path.join(module_path(), _EXPERIMENT_PATH)

def set_experiment_path(p):
    global _EXPERIMENT_PATH
    _EXPERIMENT_PATH = p


_MODULE_NAME = None
def which_module():
    import traceback
    global _MODULE_NAME
    if _MODULE_NAME is not None: return _MODULE_NAME

    stacks = traceback.extract_stack()
    if len(stacks) < 2: raise Exception('can not get module from stack {}'.format(stacks))
    for i in range(len(stacks)-2, -1 ,-1):
        f = stacks[i].filename
        try:
            return module(f)
        except: continue

    raise Exception('can not get module from stack {}'.format(stacks))


def module(f):
    __MODULE_MATCH_HEADER = os.path.join('rllab', 'algorithms', '').replace('\\', '/')
    f = f.replace('\\', '/')
    index = f.rfind(__MODULE_MATCH_HEADER)
    if index < 0: raise Exception('Can not get module from file {}'.format(f))
    st = index + len(__MODULE_MATCH_HEADER)
    ed = f.find('/', st)
    if ed < 0: raise Exception('Can not get module from file {}'.format(f))
    return f[st:ed]



