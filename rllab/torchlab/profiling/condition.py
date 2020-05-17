
__CONDITIONS = {}

def register(name):
    global __CONDITIONS
    def _thunk(func):
        __CONDITIONS[name] = func
        return func
    return _thunk


def create(type, *args, **kwargs):
    if type not in __CONDITIONS: raise Exception('Unknown condition {}'.format(type))
    return __CONDITIONS[type](*args, **kwargs)



class Condition(object):
    def __init__(self):
        self.indicator = None

    def __call__(self, *args, **kwargs): raise NotImplementedError('__call__ is not implemented')


@register('none')
class Unconditional(Condition):
    def __call__(self, *args, **kwargs): return False


@register('update')
class Updates(Condition):
    def __init__(self, updates=1):
        super(Updates, self).__init__()
        self.updates = updates

    def __call__(self, *args, **kwargs): return self.indicator.updates % self.updates == 0


@register('signal')
class Signal(Condition):
    def __init__(self, *signals):
        super(Signal, self).__init__()
        self.signals = set(signals)

    def __call__(self, signals, **kwargs):
        for s in signals:
            if s in self.signals: return True
        return False
