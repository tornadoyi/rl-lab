import abc
from collections import OrderedDict
import numpy as np
import torch


class Indicator(abc.ABC):
    def __init__(self, name=''):
        self._name = name

    @property
    def name(self): return self._name

    @abc.abstractmethod
    def __call__(self, type=None): pass

    @abc.abstractmethod
    def append(self, *args, **kwargs): pass

    @abc.abstractmethod
    def flush(self, writer, steps, *args, **kwargs): pass



class Scalar(Indicator):
    def __init__(self, *args, **kwargs):
        super(Scalar, self).__init__(*args, **kwargs)
        self._values = []
        self._vfunc = {
            None: lambda: self._values,
            'min': lambda: np.min(self._values),
            'max': lambda: np.max(self._values),
            'mean': lambda: np.mean(self._values),
        }

    def __call__(self, type=None): return self._vfunc[type]()

    def append(self, v):
        self._values.append(_scalar(v))

    def flush(self, writer, steps, type='mean', walltime=None, **kwargs):
        writer.add_scalar(self.name, self(type), steps, walltime=walltime)


class Scalars(Indicator):
    def __init__(self, *args, **kwargs):
        super(Scalars, self).__init__(*args, **kwargs)
        self._scalars = OrderedDict()

    def __call__(self, type=None):
        d = OrderedDict()
        for n, s in self._scalars.items():
            d[n] = s(type)
        return d


    def append(self, vdict):
        for k, v in vdict.items():
            if k not in self._scalars: s = Scalar(k)
            else: s = self._scalars[k]
            s.append(v)

    def flush(self, writer, steps, type='mean', walltime=None, **kwargs):
        d = dict([(n, s(type)) for n, s in self._scalars.items()])
        writer.add_scalars(self.name, d, steps, walltime=walltime)



class Histogram(Scalar):

    def __call__(self, *args, **kwargs): return self._vfunc[None]()

    def flush(self, writer, steps, bins='tensorflow', max_bins=None, walltime=None, **kwargs):
        writer.add_histogram(self.name, self(), steps, bins=bins, max_bins=max_bins, walltime=walltime)






def _scalar(v):
    if isinstance(v, torch.Tensor):
        v = v.data.numpy()
    elif isinstance(v, np.ndarray):
        pass
    else:
        raise Exception('Invalid indicator type {}'.format(type(v)))
    if len(np.shape(v)) != 0: raise Exception('Invalid indicator value shape: {}'.format(np.shape(v)))
    return v


