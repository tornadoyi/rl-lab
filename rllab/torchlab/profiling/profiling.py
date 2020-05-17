import os
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from . import indicator

class Profiling(object):
    def __init__(
            self,
            log_dir,
            step_func,
    ):
        self._log_dir = log_dir
        self._writer = SummaryWriter(self.log_dir)
        self._indicators = OrderedDict()
        self._step_func = step_func

        # create log path
        os.makedirs(self._log_dir, exist_ok=True)

    @property
    def log_dir(self): return self._log_dir

    @property
    def steps(self): return self._step_func()

    @property
    def writer(self): return self._writer

    def __contains__(self, tag): return tag in self._indicators

    def add(self, tag, *args, **kwargs):
        if tag in self._indicators: raise Exception('repeated indicator {}'.format(tag))
        self._indicators[tag] = indicator(tag, *args, **kwargs).profiling(self)

    def remove(self, tag):
        if tag not in self._indicators: return
        del self._indicators[tag]

    def update(self, tag, value, signals=(), creator=None):
        id = self._indicators.get(tag, None)
        if id is None:
            if creator is None: raise Exception('can not find indicator {}'.format(tag))
            id = creator().name(tag).profiling(self)
        id.update(value, signals)
