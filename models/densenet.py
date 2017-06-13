import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from utils import nn

# //// discriminator ////
def disc_spec(x, init=False, layers_per_block=16, filters_per_layer=16, nonlinearity='crelu', ema=None, **kwargs):

    with arg_scope([nn.conv2d, nn.dense], counters={}, init=init, weight_norm=True, ema=ema):

        def block(x):
            if type(x) is not list:
                x = [x]
            for rep in range(layers_per_block):
                x.append(nn.conv2d(x, filters_per_layer, pre_activation=nonlinearity))
            return x

        def downsample(x):
            if type(x) is not list:
                x = [x]
            return nn.conv2d(x, np.sum([int(xi.get_shape()[-1]) for xi in x])//2, pre_activation=nonlinearity, stride=[2,2])

        x = nn.conv2d(x, 2*filters_per_layer, pre_activation=None)

        x = block(x)

        x = downsample(x)

        x = block(x)

        x = downsample(x)

        x = block(x)

        x = downsample(x)

        x = tf.concat(x,3)
        x = tf.concat([tf.nn.relu(x), tf.nn.relu(-x)],3)
        xs = x.get_shape().as_list()
        x = tf.reshape(x, [-1, np.prod(xs[1:])])

        x /= tf.sqrt(tf.reduce_sum(tf.square(x), axis=1, keep_dims=True))

        # return the features
        return x

discriminator = tf.make_template('discriminator', disc_spec)


# //// generator ////
def gen_spec(batch_size, init=False, layers_per_block=16, filters_per_layer=16, nonlinearity='crelu', ema=None, **kwargs):

    u = [tf.random_uniform(shape=(batch_size, 100), minval=-1., maxval=1.),
         tf.random_uniform(shape=(batch_size, 8, 8, filters_per_layer), minval=-1., maxval=1.),
         tf.random_uniform(shape=(batch_size, 16, 16, filters_per_layer), minval=-1., maxval=1.),
         tf.random_uniform(shape=(batch_size, 32, 32, filters_per_layer), minval=-1., maxval=1.)]

    with arg_scope([nn.conv2d, nn.dense], counters={}, init=init, weight_norm=True, ema=ema):

        def block(x):
            if type(x) is not list:
                x = [x]
            for rep in range(layers_per_block):
                x.append(nn.conv2d(x, filters_per_layer, pre_activation=nonlinearity))
            return x

        def upsample(x):
            if type(x) is list:
                x = tf.concat(x,3)
            xs = x.get_shape().as_list()
            x = tf.image.resize_nearest_neighbor(x, [xs[1]*2, xs[2]*2])
            x = nn.conv2d(x, xs[3]//2, pre_activation=nonlinearity)
            return x

        x = nn.dense(u[0], 8*8*filters_per_layer, pre_activation=None)
        x = [tf.reshape(x, shape=(batch_size,8,8,filters_per_layer)), u[1]]

        x = block(x)
        x = upsample(x)
        x = [x, u[2]]
        x = block(x)
        x = upsample(x)
        x = [x, u[3]]
        x = block(x)

        x = tf.nn.tanh(nn.conv2d(x, 3, pre_activation=nonlinearity, init_scale=0.1))

        return x

generator = tf.make_template('generator', gen_spec)
