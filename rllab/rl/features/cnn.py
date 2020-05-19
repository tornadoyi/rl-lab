import numpy as np
from collections import OrderedDict
from rllab.torchlab import nn
from .features import register


@register("conv_only")
class conv_only(nn.Sequential):
    def __init__(self, input_shape, input_format='NCHW', convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
        '''
        convolutions-only net

        Parameters:
        ----------
        input_shape:    4 dims tuple with format NCHW
        conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.

        Returns:

        function that takes tensorflow tensor as input and returns the output of the last convolutional layer

        '''
        assert input_format == 'NCHW' or 'NHWC'
        l = []
        if input_format == 'NHWC':
            l.append(nn.Permute((0, 3, 1, 2)))
            input_shape = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])

        l.append(nn.Float()),
        l.append(nn.TrueDivide(255.))

        output_size = input_shape[2:4]
        in_channels = input_shape[1]
        for out_channels, kernel_size, stride in convs:
            # conv2d
            conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride)
            l.append(conv)
            in_channels = out_channels

            # activation
            l.append(nn.ReLU())

            # evaluate output shape
            output_size = nn.eval_conv_output_size(conv, output_size)

        layers = [('{}_{}'.format(l[i].__class__.__name__.lower(), i), l[i]) for i in range(len(l))]
        super(conv_only, self).__init__(OrderedDict(layers))

        self.output_shape = (input_shape[0], in_channels) + output_size




