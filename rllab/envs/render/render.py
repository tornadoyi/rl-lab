import pygame


class Render(object):

    def __init__(self, env, caption=None, win_size=(640, 480)):
        self._env = env

        # init
        if not pygame.get_init(): pygame.init()
        if not pygame.font.get_init(): pygame.font.init()
        pygame.display.set_caption(caption or str(env))
        pygame.display.set_mode(win_size)


    @property
    def screen(self): return pygame.display.get_surface()


    def __call__(self, *args, **kwargs): self.update()

    def update(self):
        # process event
        for event in pygame.event.get():
            self.on_event(event)

        # render
        self.on_render()
        pygame.display.flip()


    def on_event(self, event):  pass

    def on_render(self):  pass