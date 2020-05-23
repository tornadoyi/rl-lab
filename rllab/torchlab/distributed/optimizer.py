from torch import distributed as dist


class Optimizer(object):
    def __init__(self, optimizer, params=None):
        self._optimizer = optimizer
        self._params = params or [p for grp in self._optimizer.param_groups for p in grp['params']]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def step(self, closure=None):
        self._optimizer.step(closure)



class GradientReducer(Optimizer):
    def __init__(self, *args, reduce='mean', **kwargs):
        super(GradientReducer, self).__init__(*args, **kwargs)
        self._reduce = reduce
        assert reduce in ['mean', 'sum']


    def step(self, closure=None):

        if self._reduce == 'mean': self._reudce_mean()
        elif self._reduce == 'sum': self._reudce_sum()

        super(GradientReducer, self).step(closure)

    def _reudce_sum(self):
        for p in self._params:
            dist.all_reduce(p.grad.data, dist.reduce_op.SUM)


    def _reudce_mean(self):
        size = float(dist.get_world_size())
        for p in self._params:
            dist.all_reduce(p.grad.data, dist.reduce_op.SUM)
            p.grad.data = p.grad.data / size