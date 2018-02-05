import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from utils import nn


# //// discriminator ////
def disc_spec(x, init=False, layers_per_block=8, channel_multiplier=4, nonlinearity='crelu', ema=None, **kwargs):

    with arg_scope([nn.conv2d, nn.dense], counters={}, init=init, weight_norm=True, ema=ema):

        def block(x):
            for rep in range(layers_per_block):
                x1,x2 = tf.split(x, num_or_size_splits=2, axis=3)
                xs = x1.get_shape().as_list()
                y1 = nn.conv2d(x2, channel_multiplier*xs[-1], filter_size=[3, 3], pre_activation=nonlinearity)
                y1 = nn.conv2d(y1, xs[-1], filter_size=[3, 3], pre_activation=nonlinearity, init_scale=0.1)
                x1 += y1
                y2 = nn.conv2d(x1, channel_multiplier*xs[-1], filter_size=[3, 3], pre_activation=nonlinearity)
                y2 = nn.conv2d(y2, xs[-1], filter_size=[3, 3], pre_activation=nonlinearity, init_scale=0.1)
                x2 += y2

                # shuffle channels
                x11,x12 = tf.split(x1, 2, 3)
                x21,x22 = tf.split(x2, 2, 3)
                x = tf.concat([x12,x21,x11,x22],3)

            return x

        x = nn.conv2d(x, 4, filter_size=[5, 5], pre_activation=None)

        x = block(x)

        x = tf.space_to_depth(x, 2)
        #x = nn.conv2d(x, 64, filter_size=[5, 5], pre_activation=None, stride=[2, 2])

        x = block(x)

        x = tf.space_to_depth(x, 2)
        #x = nn.conv2d(x, 256, filter_size=[5, 5], pre_activation=None, stride=[2, 2])

        x = block(x)

        x = tf.space_to_depth(x, 2)
        #x = nn.conv2d(x, 1024, filter_size=[5, 5], pre_activation=None, stride=[2, 2])

        x = block(x)

        xs = x.get_shape().as_list()
        x = tf.reshape(x, [-1, np.prod(xs[1:])])

        # return the features
        return x

discriminator = tf.make_template('discriminator', disc_spec)
