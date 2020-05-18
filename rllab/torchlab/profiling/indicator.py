import numpy as np
import torch
from . import condition



__INDICATORS = {}

def register(name):
    global __INDICATORS
    def _thunk(func):
        __INDICATORS[name] = func
        return func
    return _thunk


def create(type, name=None):
    if type not in __INDICATORS: raise Exception('Unknown indicator type {}'.format(type))
    return __INDICATORS[type]().name(name)



class Indicator(object):
    def __init__(self):
        self._name = None
        self._profiling = None
        self._conditions = []
        self._updates = 0
        self._reset()

    @property
    def updates(self): return self._updates

    def __call__(self): pass

    def name(self, name):
        self._name = name
        return self

    def profiling(self, profiling):
        self._profiling = profiling
        return self

    def cond(self, t, *args, **kwargs):
        c = condition.create(t, *args, **kwargs)
        c.indicator = self
        self._conditions.append(c)
        return self

    def update(self, v, signals=(), **kwargs):
        self._updates += 1
        self._update(v, signals=signals, **kwargs)
        for c in self._conditions:
            if not c(signals): continue
            self.save()
            break

    def save(self):
        self._save()
        self._reset()

    def _write(self, fname, *args, **kwargs): getattr(self._profiling.writer, fname)(self._name, *args, global_step=self._profiling.steps, **kwargs)

    def _update(self, *args, **kwargs): raise NotImplementedError('_update is not implemented')

    def _save(self): raise NotImplementedError('_save is not implemented')

    def _reset(self): raise NotImplementedError('_reset is not implemented')


@register('scalar')
class Scalar(Indicator):
    def __init__(self):
        self._vfunc = _vfunc('mean')
        self._walltime = None
        super(Scalar, self).__init__()

    def __call__(self): return self._vfunc(self._values)

    def vtype(self, type):
        self._vfunc = _vfunc(type)
        return self

    def walltime(self, walltime):
        self._walltime = walltime
        return self

    def _reset(self): self._values = []

    def _update(self, v, **kwargs):
        self._values.append(_scalar(v))

    def _save(self):
        self._write('add_scalar', self(), walltime=self._walltime)


@register('scalars')
class Scalars(Indicator):
    def __init__(self):
        self._scalars = {}
        self._walltime = None
        super(Scalars, self).__init__()

    def __call__(self): return dict([(n, s()) for n, s in self._scalars.items()])

    def profiling(self, profiling):
        super(Scalars, self).profiling(profiling)
        for _, s in self._scalars.items(): s.profiling(profiling)
        return self

    def walltime(self, walltime):
        self._walltime = walltime
        return self

    def scalar(self, name):
        s = create('scalar', name)
        self._scalars[name] = s
        return self

    def _reset(self):
        for _, s in self._scalars.items(): s._reset()

    def _update(self, vdict, *args, **kwargs):
        for k, v in vdict.items():
            s = self._scalars[k]
            s.update(v, *args, **kwargs)

    def _save(self):
        self._write('add_scalars', self(), walltime=self._walltime)


@register('histogram')
class Histogram(Scalar):
    def __init__(self):
        super(Histogram, self).__init__()
        self._bins = 'tensorflow'
        self._max_bins = None
        self._vfunc = _vfunc(None)

    def vtype(self, type): raise Exception('can not set vtype for histogram')

    def bins(self, bins):
        self._bins = bins
        return self

    def max_bins(self, max_bins):
        self._max_bins = max_bins
        return self

    def _save(self):
        self._write('add_histogram', self(), bins=self._bins, max_bins=self._max_bins, walltime=self._walltime)




def _scalar(v):
    if isinstance(v, torch.Tensor):
        v = v.cpu().data.numpy()
    elif isinstance(v, (np.ndarray, float, int)):
        pass
    else:
        raise Exception('Invalid indicator type {}'.format(type(v)))
    if len(np.shape(v)) != 0: raise Exception('Invalid indicator value shape: {}'.format(np.shape(v)))
    return v


def _vfunc(t):
    d = {
        None: lambda x: x,
        'min': lambda x: np.min(x),
        'max': lambda x: np.max(x),
        'mean': lambda x: np.mean(x),
        'sum': lambda x: np.sum(x),
    }
    if t not in d: raise Exception('invalid value type {}'.format(t))
    return d[t]