import torch
import numpy as np
from .network import QFunc


class DeepQ(object):
    def __init__(
            self,
            ob_space,
            ac_space,

            **kwargs,
    ):
        self.ac_space = ac_space
        # todo noise action

        # q function
        self.net_qf = QFunc(ob_space, ac_space)




    def act(self, ob, eps):
        deterministic_actions = torch.argmax(self.net_qf(ob), 1)
        random_actions = deterministic_actions.uniform_(0, self.ac_space.n).int()
        #chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        #stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)