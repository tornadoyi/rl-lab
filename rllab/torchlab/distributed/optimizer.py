from torch import distributed as dist


class Optimizer(object):
    def __init__(self, optimizer, op=dist.reduce_op.AVG, params=None):
        self._optimizer = optimizer
        self._op = op
        self._params = params or [p for grp in self._optimizer.param_groups for p in grp['params']]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)


    def step(self, closure=None):
        """
        Gradient reduce
        """
        #size = float(dist.get_world_size())
        for p in self._params:
            dist.all_reduce(p, self._op)
        self._optimizer.step(closure)