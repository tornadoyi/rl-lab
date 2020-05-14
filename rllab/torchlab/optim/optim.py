import inspect
from torch import optim

def build(**kwargs):
    name = kwargs.get('name', None)
    if name is None: raise Exception('Build optimizer without name')

    # get optimizer
    opt = getattr(optim, name)
    if opt is None: raise Exception('Can not find optimizer {}'.format(name))

    # filter args
    opt_args = set(inspect.getfullargspec(opt).args)
    args = dict([(k, v) for k, v in kwargs.items() if k in opt_args])

    return opt(**args)