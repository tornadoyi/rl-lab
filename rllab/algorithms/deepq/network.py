import torch
from torch import nn
from rllab.rl import features

class QFunc(nn.Module):
    def __init__(
            self,
            ob_space,
            ac_space,
            hiddens=(256, ),
            dueling=True,
            layer_norm=False,
            feature={},
            **_,
    ):
        super(QFunc, self).__init__()

        self.dueling = dueling

        # create feature extractor
        self.net_features = features.build(ob_space, **feature)

        # create action score network
        l = []
        in_features = self.net_features.output_shape[1]
        for hidden in list(hiddens):
            l.append(nn.Linear(in_features, hidden))
            in_features = hidden
            if layer_norm: l.append(nn.LayerNorm([in_features]))
            l.append(nn.ReLU())
        l.append(nn.Linear(in_features, ac_space.n))
        self.net_action_score = nn.Sequential(*l)

        # dueling
        if dueling:
            l = []
            in_features = self.net_features.output_shape[1]
            for hidden in list(hiddens):
                l.append(nn.Linear(in_features, hidden))
                in_features = hidden
                if layer_norm: l.append(nn.LayerNorm([in_features]))
                l.append(nn.ReLU())
            l.append(nn.Linear(in_features, 1))
            self.net_state_score = nn.Sequential(*l)


    def forward(self, ob):
        ob = self.net_features(ob)
        self.action_score = self.net_action_score(ob)

        # calculate advantage for dueling network
        if self.dueling:
            self.state_score = self.net_state_score(ob)
            self.action_scores_mean = torch.mean(self.action_score, 1)
            self.action_scores_centered = self.action_score  - self.action_scores_mean.unsqueeze(1)
            self.q = self.state_score + self.action_scores_centered
        else:
            self.q = self.action_score

        return self.q
