from rllab import torchlab as tl
from rllab.torchlab import nn
from rllab.torchlab.nn import functional as F
from rllab.rl.profiling import indicator
from .network import QFunc


class DeepQ(nn.Module):
    def __init__(
            self,
            ob_space,
            ac_space,
            feature_creator,
            double_q=False,
            grad_norm_clipping=None,
            gamma=1.0,
            qfunc={},
            **_,
    ):
        super(DeepQ, self).__init__()

        # config
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.double_q = double_q
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping

        # q function
        self.net_q_eval = QFunc(ac_space, feature_creator, **qfunc)
        self.net_q_target = QFunc(ac_space, feature_creator,  **qfunc)


    @property
    def trained_parameters(self): return self.net_q_eval.parameters()


    def act(self, ob, eps):
        obs = ob.reshape(*[(-1, ) + self.ob_space.shape])

        # todo noise action
        deterministic_actions = tl.argmax(self.net_q_eval(obs), 1)
        random_actions = deterministic_actions.float().uniform_(0.0, float(self.ac_space.n)).long()
        conditions = deterministic_actions.float().uniform_(0, 1) < eps
        final_actions = tl.where(conditions, random_actions, deterministic_actions)
        return final_actions.squeeze()


    def learn(self, optimizer, obs, acs, rews, obs_n, dones, weights=None):
        # calculate q evaluation
        q_eval = self.net_q_eval(obs)

        # calculate q target and stop gradients
        q_target = self.net_q_target(obs_n).detach()

        # q scores for actions which we know were selected in the given state.
        q_eval_selected = tl.sum(q_eval * F.one_hot(acs, self.ac_space.n), 1)

        # double q
        if self.double_q:
            q_eval_n = self.net_q_eval(obs_n)
            max_q_ac_n = tl.argmax(q_eval_n, 1)
            q_best = tl.sum(q_target * F.one_hot(max_q_ac_n, self.ac_space.n), 1)
        else:
            q_best = q_target.max(1)[0]

        # mask terminal
        q_best = (1.0 - dones) * q_best

        # compute RHS of bellman equation
        q_target_selected = rews + self.gamma * q_best

        # compute the error (potentially clipped)
        td_error = q_eval_selected - q_target_selected

        # loss
        errors = F.huber_loss(td_error)
        errors = tl.mean(weights * errors)

        # compute gradients (potentially with gradient clipping)
        optimizer.zero_grad()
        errors.backward()
        if self.grad_norm_clipping is not None:
            for p in self.net_q_eval.parameters(): nn.utils.clip_grad_norm(p, self.grad_norm_clipping)
        optimizer.step()

        # profiling
        indicators = {
            'deepq/loss':                 (errors,                        lambda: indicator('scalar').cond('update')),
            'deepq/td_error':             (tl.mean(td_error),          lambda: indicator('scalar').cond('update')),
            'deepq/q_eval_selected':      (tl.mean(q_eval_selected),   lambda: indicator('scalar').cond('update')),
            'deepq/q_target_selected':    (tl.mean(q_target_selected), lambda: indicator('scalar').cond('update')),
        }

        # gradients profiling
        for k, v in self.net_q_eval.state_dict().items():
            indicators['gradients/{}'.format(k)] = (v.abs().mean(), lambda: indicator('scalar').cond('update'))

        return indicators


    def update_target_network(self):
        self.net_q_target.load_state_dict(self.net_q_eval.state_dict())







