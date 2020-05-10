from .runway import RunwayEnv


class ShuttleRunEnv(RunwayEnv):

    def on_reset(self):
        super(ShuttleRunEnv, self).on_reset()
        self._half_finish = False

        ob = [0] + self._observations[-1]
        self.store_observation(ob)


    def on_step(self, action):
        super(ShuttleRunEnv, self).on_step(action)

        reward = 0
        if not self._half_finish:
            if self._pos == self._length - 1:
                self._half_finish = True
                reward = 1.0
        else:
            if self._pos == 0:
                reward = 2.0
                self._terminated = True

        # save ob
        ob = [1] + self._observations[-1] if self._half_finish else [0] + self._observations[-1]
        self.store_observation(ob)

        # save reward
        self.store_reward(reward)



