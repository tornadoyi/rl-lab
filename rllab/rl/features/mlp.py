import numpy as np
from torch import nn
from .features import register


@register("mlp")
class MLP(nn.Sequential):
    def __init__(self, input_shape, num_layers=2, num_hidden=64, activation=None, layer_norm=False):
        """
        Stack of fully-connected layers to be used in a policy / q-function approximator
        Parameters:
        ----------
        num_layers: int                 number of fully-connected layers (default: 2)
        num_hidden: int                 size of fully-connected layers (default: 64)
        activation:                     activation function (default: tf.tanh)
        Returns:
        -------
        Sequential build by fully connected network
        """

        self.output_shape = (input_shape[0], num_hidden)

        # calculate in features
        in_features = 1
        for d in input_shape[1:]: in_features *= d

        l = [nn.Flatten()]
        for i in range(num_layers):
            # fc
            x = nn.Linear(in_features, num_hidden)
            nn.init.orthogonal_(x.weight.data, gain=np.sqrt(2))
            nn.init.constant_(x.bias.data, 0.0)
            l.append(x)
            in_features = num_hidden

            # normalize
            if layer_norm: l.append(nn.LayerNorm([in_features]))

            # activation
            l.append(nn.Tanh() if activation is None else activation)

        super(MLP, self).__init__(*l)
