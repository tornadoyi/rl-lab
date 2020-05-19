
from rllab.torchlab import profiling
from rllab.torchlab.profiling import indicator
from rllab import define

class Profiling(profiling.Profiling):
    def __init__(self, env, **kwargs):
        super(Profiling, self).__init__(define.profiling_path(), **kwargs)
        self.env = env


    def __call__(self, *args, **kwargs):
        ud = self.env.userdata
        if ud.reward:
            self.update('env/mean_reward_100s', ud.reward, creator=lambda: indicator('scalar').cond('update', 100))

        if ud.done:
            self.update('env/round_steps', ud.steps, creator=lambda: indicator('scalar').cond('signal', 'done'))
            self.update('env/round_reward',ud.total_reward, creator=lambda: indicator('scalar').cond('signal', 'done'))


    def update(self, tag, value, signals=(), creator=None):
        signals = set(signals)
        if self.env.userdata.done: signals.add('done')
        super(Profiling, self).update(tag, value, signals, creator)



