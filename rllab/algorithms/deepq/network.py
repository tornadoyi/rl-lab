from collections import OrderedDict
from rllab import torchlab as tl
from rllab.torchlab import nn


class QFunc(nn.Module):
    def __init__(
            self,
            ac_space,
            feature_creator,
            hiddens=(256, ),
            dueling=True,
            layer_norm=False,
            **_,
    ):
        super(QFunc, self).__init__()

        self.dueling = dueling

        # create feature extractor
        self.net_features = feature_creator()

        # create action score network
        l = []
        in_features = 1
        for d in self.net_features.output_shape[1:]: in_features *= d
        l.append(nn.Flatten())

        for hidden in list(hiddens):
            l.append(nn.Linear(in_features, hidden))
            in_features = hidden
            if layer_norm: l.append(nn.LayerNorm([in_features]))
            l.append(nn.ReLU())
        l.append(nn.Linear(in_features, ac_space.n))
        layers = [('{}_{}'.format(l[i].__class__.__name__.lower(), i), l[i]) for i in range(len(l))]
        self.net_action_score = nn.Sequential(OrderedDict(layers))

        # dueling
        if dueling:
            l = []
            in_features = 1
            for d in self.net_features.output_shape[1:]: in_features *= d
            l.append(nn.Flatten())

            for hidden in list(hiddens):
                l.append(nn.Linear(in_features, hidden))
                in_features = hidden
                if layer_norm: l.append(nn.LayerNorm([in_features]))
                l.append(nn.ReLU())
            l.append(nn.Linear(in_features, 1))
            layers = [('{}_{}'.format(l[i].__class__.__name__.lower(), i), l[i]) for i in range(len(l))]
            self.net_state_score = nn.Sequential(OrderedDict(layers))


    def forward(self, ob):
        ob = self.net_features(ob)
        self.action_score = self.net_action_score(ob)

        # calculate advantage for dueling network
        if self.dueling:
            self.state_score = self.net_state_score(ob)
            self.action_scores_mean = tl.mean(self.action_score, 1)
            self.action_scores_centered = self.action_score  - self.action_scores_mean.unsqueeze(1)
            self.q = self.state_score + self.action_scores_centered
        else:
            self.q = self.action_score

        return self.q
