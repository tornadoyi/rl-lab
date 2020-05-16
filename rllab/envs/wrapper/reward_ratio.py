


from .wrapper import Wrapper

class RewardRatio(Wrapper):

    def step(self, action):
        ob, r, d, info = super(RewardRatio, self).step(action)
        r = r * self.spec.reward_threshold
        return ob, r, d, info
