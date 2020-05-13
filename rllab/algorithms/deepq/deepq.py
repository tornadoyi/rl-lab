import torch
from torch import nn
from rllab.torchlab.nn import functional as F
import numpy as np
from .network import QFunc


class DeepQ(object):
    def __init__(
            self,
            ob_space,
            ac_space,
            double_q=False,
            gamma=1.0,
            **kwargs,
    ):
        # config
        self.ac_space = ac_space
        self.double_q = double_q
        self.gamma = gamma

        # q function
        self.net_q_eval = QFunc(ob_space, ac_space)
        self.net_q_target = QFunc(ob_space, ac_space)


    def act(self, obs, eps):
        # todo noise action
        deterministic_actions = torch.argmax(self.net_q_eval(obs), 1)
        random_actions = deterministic_actions.uniform_(0, self.ac_space.n).int()
        conditions = deterministic_actions.uniform_(0, 1) < eps
        final_actions = torch.where(conditions, random_actions, deterministic_actions)
        return final_actions


    def learn(self, obs, acs, rews, obs_n, dones, weights=None):
        # convert to tensor
        obs = torch.from_numpy(obs)
        acs = torch.from_numpy(acs)
        rews = torch.from_numpy(rews)
        obs_n = torch.from_numpy(obs_n)
        dones = torch.from_numpy(dones)
        weights = torch.from_numpy(weights or [1.0]*obs.shape[0])

        # calculate q evaluation
        q_eval = self.net_q_eval(obs)

        # calculate q target
        with torch.no_grad():
            q_target = self.net_q_target(obs_n)

        # q scores for actions which we know were selected in the given state.
        q_selected = torch.sum(q_eval * F.one_hot(acs, self.ac_space.n), 1)

        # todo double q
        if self.double_q:
            pass
        else:
            q_best = torch.max(q_target, 1)

        # mask terminal
        q_best = (1.0 - dones) * q_best

        # compute RHS of bellman equation
        y = rews + self.gamma * q_best

        # compute the error (potentially clipped)
        td_error = q_selected - q_target

        # loss
        errors = F.huber_loss(td_error)
        errors = torch.mean(weights * errors)

        # compute optimization op (potentially with gradient clipping)
