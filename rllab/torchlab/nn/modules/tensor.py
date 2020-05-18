from inspect import ismethoddescriptor
import torch
from torch import nn
import humps

def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
    nn.Module.__init__(self)

def _create_module_class(name, func):
    return type(
        name,
        (nn.Module, ),
        {
            "__init__": __init__,
            "forward": lambda self, x: func(x, *self.args, **self.kwargs)
        }
    )

for name in dir(torch.Tensor):
    if name.startswith('_') or name.endswith('_'): continue
    f = getattr(torch.Tensor, name)
    if not ismethoddescriptor(f): continue
    name = humps.pascalize(name)
    globals()[name] = _create_module_class(name, f)

