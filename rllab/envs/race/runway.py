import numpy as np
import gym
from gym import spaces
from rllab.envs import render

class RunwayEnv(gym.Env):
    def __init__(
            self,
            length=1,
            init_pos=0,
            move_success_rate=1.0,
            **kwargs
    ):

        super(RunwayEnv, self).__init__(**kwargs)

        # config
        self._length = length
        self._init_pos = init_pos
        self._move_success_rate = move_success_rate

        # gym
        self.action_space = spaces.Discrete(3)     # 0: wait  1: move left  2: move right
        self.observation_space = spaces.Box(0.0, 1.0, (self._length, ))

        # state
        self._pos = self._init_pos

        # render
        self._render = None


    def reset(self):
        self._pos = self._init_pos
        ob = [0.0] * self._length
        ob[int(self._pos)] = 1.0
        return ob


    def step(self, action):
        # move
        if action != 0 and np.random.rand() < self._move_success_rate:
            if action == 1:
                self._pos = np.clip(self._pos - 1, 0, self._length - 1)
            elif action == 2:
                self._pos = np.clip(self._pos + 1, 0, self._length - 1)

        # save ob
        ob = [0.0] * self._length
        ob[int(self._pos)] = 1.0
        return ob, 0, False, {}


    def render(self, mode='human'):
        if self._render is None: self._render = Render(self)
        self._render()


    def render_infos(self): return []


class Render(render.Application):

    def __init__(self, *args, **kwargs):
        super(Render, self).__init__(*args, **kwargs)


    def on_render(self):
        w, h = self.screen.get_width(), self.screen.get_height()

        # clear screen
        self.screen.fill(render.Color('white'))

        # draw way
        way_rect = render.Rect(0.1 * w, 0.6 * h, 0.8 * w, 0.1 * h)
        render.draw.lines(self.screen, render.Color('black'), False, [
            way_rect.topleft,
            way_rect.bottomleft,
            way_rect.bottomright,
            way_rect.topright
        ], 5)

        # draw player
        ratio = (self._env._pos + 1) / self._env._length
        player_x = way_rect.topleft[0] + way_rect.width * ratio
        render.draw.line(self.screen, render.Color('red'), (player_x, way_rect.topleft[1]), (player_x, way_rect.bottomleft[1]), 5)


        # episodes
        infos = []
        shared = self._env.shared_data
        if 'steps' in shared:
            if self._env.spec.max_episode_steps is not None:
                infos.append('episodes: {}/{}'.format(shared.steps, self._env.spec.max_episode_steps))
            else:
                infos.append('episodes: {}'.format(shared.steps))

        # reward
        if 'total_reward' in shared:
            infos.append('rewards: {}'.format(shared.total_reward))

        # location
        infos.append('location: {}/{}'.format(self._env._pos + 1, self._env._length))

        # extra
        infos += self._env.render_infos()

        # draw informations
        render.font.blit_text(self.screen, '\n'.join(infos), (0, 0), render.font.Font(render.font.get_default_font(), 20), )


