""" simplified, more efficient, but slightly wrong, version of the original (two-batch) training code """

import argparse
import time
import os
import numpy as np
import tensorflow as tf
from utils import nn
from utils import plotting
from models.densenet import generator, discriminator
from data import cifar10_data

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--learning_rate_disc', type=float, default=0.0001)
parser.add_argument('--learning_rate_gen', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='/home/tim/data')
parser.add_argument('--save_dir', type=str, default='/local_home/tim/med_gan')
parser.add_argument('--optimizer', type=str, default='adamax')
parser.add_argument('--nonlinearity', type=str, default='crelu')
parser.add_argument('--layers_per_block', type=int, default=8)
parser.add_argument('--filters_per_layer', type=int, default=48)
parser.add_argument('--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
parser.add_argument('--nr_gen_per_disc', type=int, default=10, help='How many times to update the generator for each update of the discriminator?')
parser.add_argument('--sinkhorn_lambda', type=float, default=30.)
parser.add_argument('--nr_sinkhorn_iter', type=int, default=1000)
args = parser.parse_args()
print(args)

# extract model settings
model_opts = {'batch_size': args.batch_size, 'layers_per_block': args.layers_per_block,
              'filters_per_layer': args.filters_per_layer, 'nonlinearity': args.nonlinearity}

# fix random seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# run once for data dependent initialization of parameters
x_init = tf.placeholder(tf.float32, shape=(args.batch_size, 32, 32, 3))
f = discriminator(x_init, init=True, **model_opts)
generator(init=True, **model_opts)
num_features = f.get_shape().as_list()[-1]
print("model has a hidden representation with %d features" % num_features)

# get list of all params
all_params = tf.trainable_variables()
saver = tf.train.Saver(all_params)
disc_params = [p for p in all_params if 'discriminator' in p.name]
gen_params = [p for p in all_params if 'generator' in p.name]

# data placeholders
x_data = [tf.placeholder(tf.float32, shape=(args.batch_size, 32, 32, 3)) for i in range(args.nr_gpu)]

# generate samples
x_gens = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        x_gens.append(generator(**model_opts))

# feature extraction
features_dat = []
features_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        features_gen.append(discriminator(x_gens[i], **model_opts))
        features_gen[i] /= tf.sqrt(tf.reduce_sum(tf.square(features_gen[i]), axis=1, keep_dims=True))

        with tf.control_dependencies([features_gen[i]]):  # prevent TF from trying to do this simultaneously and running out of memory
            features_dat.append(discriminator(x_data[i], **model_opts))
            features_dat[i] /= tf.sqrt(tf.reduce_sum(tf.square(features_dat[i]), axis=1, keep_dims=True))

# gather all features
with tf.control_dependencies(features_dat):  # prevent TF from trying to do this simultaneously and running out of memory
    fg_all = tf.concat(features_gen,axis=0)
    fd_all = tf.concat(features_dat,axis=0)

# calculate all distances
dist_gen_gen = []
dist_dat_dat = []
dist_gen_dat = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        dist_gen_gen.append(1. - tf.matmul(features_gen[i],fg_all,transpose_b=True))
        dist_dat_dat.append(1. - tf.matmul(features_dat[i],fd_all,transpose_b=True))
        dist_gen_dat.append(1. - tf.matmul(features_gen[i],fd_all,transpose_b=True))

# combine results + add a bit to the diagonal to prevent self-matches
distances = [tf.concat(dist_gen_gen,0) + 999.*tf.eye(args.nr_gpu * args.batch_size),
             tf.concat(dist_dat_dat,0) + 999.*tf.eye(args.nr_gpu * args.batch_size),
             tf.concat(dist_gen_dat,0)]

# use Sinkhorn algorithm to do soft assignment
assignments = []
for i in range(len(distances)):
    with tf.device('/gpu:%d' % (i%args.nr_gpu)):
        log_a = -args.sinkhorn_lambda * distances[i]

        for it in range(args.nr_sinkhorn_iter):
            log_a -= tf.reduce_logsumexp(log_a, axis=1, keep_dims=True)
            log_a -= tf.reduce_logsumexp(log_a, axis=0, keep_dims=True)

        assignments.append(tf.nn.softmax(log_a))

assignment_gen_gen, assignment_dat_dat, assignment_gen_dat = assignments

# get matched features
features_gen_gen_matched = tf.split(tf.matmul(assignment_gen_gen, fg_all), args.nr_gpu, 0)
features_dat_dat_matched = tf.split(tf.matmul(assignment_dat_dat, fd_all), args.nr_gpu, 0)
features_gen_dat_matched = tf.split(tf.matmul(assignment_gen_dat, fd_all), args.nr_gpu, 0)
features_dat_gen_matched = tf.split(tf.matmul(assignment_gen_dat, fg_all, transpose_a=True), args.nr_gpu, 0)

# get distances
dist = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        nd_gen_gen = tf.reduce_sum(features_gen[i] * features_gen_gen_matched[i])
        nd_dat_dat = tf.reduce_sum(features_dat[i] * features_dat_dat_matched[i])
        nd_gen_dat = tf.reduce_sum(features_gen[i] * features_gen_dat_matched[i])
        dist.append(nd_dat_dat + nd_gen_gen - 2. * nd_gen_dat)
total_dist = sum(dist)/(2*args.batch_size*args.nr_gpu)

# get gradients
grads_gen = []
grads_disc = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        grad_features_gen_i = features_gen_gen_matched[i] - features_gen_dat_matched[i]
        grad_features_dat_i = features_dat_dat_matched[i] - features_dat_gen_matched[i]

        with tf.control_dependencies([total_dist]): # prevent TF from trying to do this simultaneously and running out of memory
            grad_disc_i = tf.gradients(ys=features_dat[i], xs=disc_params, grad_ys=grad_features_dat_i)

        with tf.control_dependencies(grad_disc_i): # prevent TF from trying to do this simultaneously and running out of memory
            grad_disc_and_sample_i = tf.gradients(ys=features_gen[i], xs=disc_params+[x_gens[i]], grad_ys=grad_features_gen_i)

        for j in range(len(grad_disc_i)):
            grad_disc_i[j] += grad_disc_and_sample_i[j]

        with tf.control_dependencies(grad_disc_i): # prevent TF from trying to do this simultaneously and running out of memory
            grad_gen_i = tf.gradients(ys=x_gens[i], xs=gen_params, grad_ys=grad_disc_and_sample_i[-1])

        grads_disc.append(grad_disc_i)
        grads_gen.append(grad_gen_i)

# add gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        for j in range(len(grads_gen[0])):
            grads_gen[0][j] += grads_gen[i][j]
        for j in range(len(grads_disc[0])):
            grads_disc[0][j] += grads_disc[i][j]

    if args.optimizer == 'adam':
        gen_optimizer = nn.adam_updates(gen_params, grads_gen[0], lr=tf_lr, mom1=0.5, mom2=0.99)
        disc_optimizer = nn.adam_updates(disc_params, grads_disc[0], lr=-tf_lr, mom1=0.5, mom2=0.95)
    elif args.optimizer == 'adamax':
        gen_optimizer = nn.adamax_updates(gen_params, grads_gen[0], lr=tf_lr, mom1=0.5, mom2=0.99)
        disc_optimizer = nn.adamax_updates(disc_params, grads_disc[0], lr=-tf_lr, mom1=0.5, mom2=0.95)
    elif args.optimizer == 'nesterov':
        gen_optimizer = nn.nesterov_updates(gen_params, grads_gen[0], lr=tf_lr, mom1=0.5)
        disc_optimizer = nn.nesterov_updates(disc_params, grads_disc[0], lr=-tf_lr, mom1=0.5)
    else:
        raise('unsupported optimizer')

# init
initializer = tf.global_variables_initializer()

# load CIFAR-10 training data
trainx, trainy = cifar10_data.load(args.data_dir + '/cifar-10-python')
trainx = np.transpose(trainx, (0,2,3,1))/127.5 - 1.
nr_batches_train_per_gpu = int(trainx.shape[0]/(args.nr_gpu*args.batch_size))

def maybe_flip(x):
    x_out = np.zeros_like(x)
    for i in range(x.shape[0]):
        if np.random.rand() < 0.5:
            x_out[i] = x[i,:,::-1,:]
        else:
            x_out[i] = x[i]
    return x_out

# //////////// perform training //////////////
try:
    os.stat(args.save_dir)
except:
    os.mkdir(args.save_dir)
print('starting training')
step_counter = 0
with tf.Session() as sess:
    for epoch in range(1000000):
        begin = time.time()

        # randomly permute
        inds = np.random.permutation(trainx.shape[0])
        trainx = trainx[inds]

        # init
        if epoch==0:
            sess.run(initializer, feed_dict={x_init: trainx[:args.batch_size]})

        # train
        np_distances = []
        for t in range(nr_batches_train_per_gpu):
            feed_dict = {}
            for i in range(args.nr_gpu):
                td = t + i * nr_batches_train_per_gpu
                feed_dict[x_data[i]] = maybe_flip(trainx[td * args.batch_size:(td + 1) * args.batch_size])

            # train discriminator once every args.nr_gen_per_disc batches
            if step_counter % (args.nr_gen_per_disc+1) == 0:
                feed_dict.update({tf_lr: args.learning_rate_disc})
                npd, _ = sess.run([total_dist, disc_optimizer], feed_dict=feed_dict)
                step_counter += 1

            else: # train generator
                feed_dict.update({tf_lr: args.learning_rate_gen})
                npd, _ = sess.run([total_dist, gen_optimizer], feed_dict=feed_dict)
                step_counter += 1

            np_distances.append(npd)

        # log
        print("Iteration %d, time = %ds, train distance = %.6f" % (epoch, time.time()-begin, np.mean(np_distances)))

        # save generated image
        sample_x = sess.run(x_gens)
        sample_x = np.concatenate(sample_x)
        img_tile = plotting.img_tile(sample_x[:100], aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile, title='CIFAR10 samples')
        plotting.plt.savefig(os.path.join(args.save_dir, 'sample%d.png' % epoch))
        plotting.plt.close('all')

        # save parameters
        if epoch+1 % 50 == 0:
            saver.save(sess, os.path.join(args.save_dir, 'med_gan_params'), global_step=epoch)
