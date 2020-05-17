
from rllab.torchlab import profiling
from rllab.torchlab.profiling import indicator
from rllab import define

class Profiling(profiling.Profiling):
    def __init__(self, env, flush_freq=1, print_freq=1, **kwargs):
        super(Profiling, self).__init__(define.profiling_path(), **kwargs)
        self.env = env
        self.flush_freq = flush_freq
        self.print_freq = print_freq

    def __call__(self, *args, **kwargs):
        if self.env.reward:
            self.update('env/mean_reward_100s', self.env.reward, creator=lambda: indicator('scalar').cond('update', 100))

        if self.env.done:
            self.update('env/round_steps', self.env.steps, creator=lambda: indicator('scalar').cond('signal', 'done'))
            self.update('env/round_reward', self.env.total_reward, creator=lambda: indicator('scalar').cond('signal', 'done'))


    def update(self, tag, value, signals=(), creator=None):
        signals = set(signals)
        if self.env.done: signals.add('done')
        super(Profiling, self).update(tag, value, signals, creator)



