from collections import OrderedDict
import numpy as np
from rllab.torchlab import nn
from .features import register





@register("conv_only")
class cnn_small(nn.Sequential):
    def __init__(self, input_shape):
    #def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net

    Parameters:
    ----------

    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.

    Returns:

    function that takes tensorflow tensor as input and returns the output of the last convolutional layer

    '''

    def network_fn(X):
        out = tf.cast(X, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = tf.contrib.layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu,
                                           **conv_kwargs)

        return out
    return network_fn


@register("cnn_small")
class cnn_small(nn.Sequential):
    def __init__(self, input_shape):
        """
        Stack of convolution layers to be used in a policy / q-function approximator
        Parameters:
        ----------
        input_shape: tuple              should be a shape with format (batch, height, width, channel)
        num_layers: int                 number of fully-connected layers (default: 2)
        num_hidden: int                 size of fully-connected layers (default: 64)
        activation:                     activation function (default: tf.tanh)
        Returns:
        -------
        Sequential build by cnn network
        """

        super(cnn_small, self).__init__()

    @register("cnn_small")
    def cnn_small(**conv_kwargs):
        def network_fn(X):
            h = tf.cast(X, tf.float32) / 255.

            activ = tf.nn.relu
            h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
            h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
            h = conv_to_fc(h)
            h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
            return h
        return network_fn