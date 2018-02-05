import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

def adamax_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adamax_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adamax_v')
            v_t = mom1*v + (1. - mom1)*g
            updates.append(v.assign(v_t))
        else:
            v_t = g
        mg_t = tf.maximum(mom2*mg + 1e-8, tf.abs(g))
        g_t = v_t / mg_t
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    return tf.group(*updates)

def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)

def nesterov_updates(params, cost_or_grads, lr=0.01, mom1=0.9):
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    for p, g in zip(params, grads):
        v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_nesterov_mom')
        v_new = mom1*v - lr*g
        p_new = p - mom1*v + (1. + mom1)*v_new
        updates.append(p.assign(p_new))
        updates.append(v.assign(v_new + 0.*p_new)) # to ensure it runs after updating p
    return tf.group(*updates)

def get_var_maybe_avg(var_name, ema, **kwargs):
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def get_name(layer_name, counters):
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

# get params, using data based initialization & (optionally) weight normalization, and using moving averages
def get_params(layer_name, x=None, init=False, ema=None, use_W=True, use_g=True, use_b=True,
               f=tf.matmul, weight_norm=True, init_scale=1., filter_size=None, num_units=None, pre_activation=None):
    params = {}
    with tf.variable_scope(layer_name):
        if init:
            if type(x) is list:
                xs = x[0].get_shape().as_list()
                xs[-1] = np.sum([xi.get_shape().as_list()[-1] for xi in x])
            else:
                xs = x.get_shape().as_list()
            if num_units is None:
                num_units = xs[-1]
            norm_axes = [i for i in np.arange(len(xs) - 1)]

            # weights
            if use_W:
                nr_in = xs[-1]
                if pre_activation in ['celu', 'crelu']:
                    nr_in *= 2
                if filter_size is not None:
                    V = tf.get_variable('V', filter_size + [nr_in, num_units], tf.float32,
                                        tf.random_normal_initializer(0, 0.05), trainable=True)
                else:
                    V = tf.get_variable('V', [nr_in, num_units], tf.float32,
                                    tf.random_normal_initializer(0, 0.05), trainable=True)
                if weight_norm:
                    W = tf.nn.l2_normalize(V, [i for i in np.arange(len(V.get_shape())-1)])
                else:
                    W = V

            # moments for normalization
            if use_W:
                x_init = f(x, W)
            else:
                x_init = x
            m_init, v_init = tf.nn.moments(x_init, norm_axes)

            # scale
            init_g = init_scale / tf.sqrt(v_init)
            if use_g:
                g = tf.get_variable('g', dtype=tf.float32, shape=num_units, initializer=tf.ones_initializer(), trainable=True)
                g = g.assign(init_g)
                if use_W:
                    W *= tf.reshape(g, [1]*(len(W.get_shape())-1)+[num_units])
                else: # g is used directly if there are no weights
                    params['g'] = g
                m_init *= init_g
            elif use_W and not weight_norm: # init is the same as when using weight norm
                W = V.assign(tf.reshape(init_g, [1]*(len(W.get_shape())-1) + [num_units]) * W)
                m_init *= init_g

            # (possibly) scaled weights
            if use_W:
                params['W'] = W

            # bias
            if use_b:
                b = tf.get_variable('b', dtype=tf.float32, shape=num_units, initializer=tf.zeros_initializer(), trainable=True)
                b = b.assign(-m_init)
                params['b'] = b

        else:
            # get variables, use the exponential moving average if provided
            if use_b:
                params['b'] = get_var_maybe_avg('b', ema)
            if use_g:
                g = get_var_maybe_avg('g', ema)
                if not use_W: # g is used directly if there are no weights
                    params['g'] = g
            if use_W:
                V = get_var_maybe_avg('V', ema)
                Vs = V.get_shape().as_list()
                if weight_norm:
                    W = tf.nn.l2_normalize(V, [i for i in np.arange(len(Vs)-1)])
                else:
                    W = V
                if use_g:
                    W *= tf.reshape(g, [1]*(len(Vs)-1) + [Vs[-1]])
                params['W'] = W

    return params

def apply_pre_activation(x, pre_activation, axis=3):
    if type(x) is tuple:
        x = list(x)
    elif type(x) is not list:
        x = [x]
    if pre_activation is None:
        return tf.concat(x,axis)
    elif pre_activation == 'celu':
        return tf.nn.elu(tf.concat([xs for xi in x for xs in [xi,-xi]],axis))
    elif pre_activation == 'crelu':
        return tf.nn.relu(tf.concat([xs for xi in x for xs in [xi,-xi]],axis))
    elif pre_activation == 'elu':
        return tf.nn.elu(tf.concat(x,axis))
    elif pre_activation == 'relu':
        return tf.nn.relu(tf.concat(x,axis))
    else:
        raise('unsupported pre-activation')

def _dense(x, W, pre_activation=None):
    x_out = tf.matmul(apply_pre_activation(x, pre_activation, 1), W)
    return x_out

def _conv2d(x, W, stride=[1,1], pad='SAME', dilate=1, pre_activation=None):
    x = apply_pre_activation(x, pre_activation, 3)
    if dilate > 1:
        x_out = tf.nn.atrous_conv2d(x, W, dilate, pad)
    else:
        x_out = tf.nn.conv2d(x, W, [1] + stride + [1], pad)
    return x_out

@add_arg_scope
def dense(x, num_units, pre_activation='celu', init_scale=1., counters={}, init=False,
          ema=None, weight_norm=True, use_b=True, use_g=True, **kwargs):
    layer_name = get_name('dense', counters)
    f = lambda x, W: _dense(x, W, pre_activation)
    params = get_params(layer_name, x, init, ema, use_W=True, use_g=use_g, use_b=use_b, f=f,
                        weight_norm=weight_norm, init_scale=init_scale, num_units=num_units, pre_activation=pre_activation)
    x = f(x, params['W'])
    if use_b:
        x = tf.nn.bias_add(x, params['b'])
    return x

@add_arg_scope
def conv2d(x, num_filters, pre_activation='celu', filter_size=[3,3], stride=[1,1], pad='SAME', dilate=1,
           init_scale=1., counters={}, init=False, ema=None, weight_norm=True, use_b=True, use_g=True, **kwargs):
    layer_name = get_name('conv2d', counters)
    f = lambda x,W: _conv2d(x, W, stride, pad, dilate, pre_activation)
    params = get_params(layer_name, x, init, ema, use_W=True, use_g=use_g, use_b=use_b, f=f,
                        weight_norm=weight_norm, init_scale=init_scale, filter_size=filter_size, num_units=num_filters, pre_activation=pre_activation)
    x = f(x, params['W'])
    if use_b:
        x = tf.nn.bias_add(x, params['b'])
    return x
