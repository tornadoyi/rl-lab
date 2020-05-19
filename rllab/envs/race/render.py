
from rllab.envs import render



class Render(render.Render):

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
        ud = self._env.userdata
        if 'steps' in ud:
            if self._env.spec.max_episode_steps is not None:
                infos.append('episodes: {}/{}'.format(ud.steps, self._env.spec.max_episode_steps))
            else:
                infos.append('episodes: {}'.format(ud.steps))

        # reward
        if 'total_reward' in ud:
            infos.append('rewards: {}'.format(ud.total_reward))

        # location
        infos.append('location: {}/{}'.format(self._env._pos + 1, self._env._length))

        # extra
        infos += self._env.render_infos()

        # draw informations
        render.font.blit_text(self.screen, '\n'.join(infos), (0, 0), render.font.Font(render.font.get_default_font(), 20), )


