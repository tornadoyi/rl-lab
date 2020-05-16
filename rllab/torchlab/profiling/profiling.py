
from torch.utils.tensorboard import SummaryWriter
from .indicator import *

class Profiling(object):
    def __init__(
            self,
            log_dir,
    ):

        self._log_dir = log_dir
        self._writer = SummaryWriter(self.log_dir)
        self._indicators = OrderedDict()


    @property
    def log_dir(self): return self._log_dir

    @property
    def tensorboard(self): return self._writer

    def add_scalar(self, tag, v):
        if tag not in self._indicators: id = Scalar(tag)
        else: id = self._indicators[tag]
        id.append(v)


    def add_scalars(self, tag, vdict):
        if tag not in self._indicators: id = Scalars(tag)
        else: id = self._indicators[tag]
        id.append(vdict)


    def flush(self, global_step, **indicator_kwargs):
        for tag, id in self._indicators.items():
            id.flush(self._writer, global_step, **indicator_kwargs.get(tag, {}))
        self._indicators = OrderedDict()