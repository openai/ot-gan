import argparse
import time
import os
import numpy as np
import tensorflow as tf
from utils import nn
from utils import plotting
from utils.matching import minibatch_energy_distance
from utils.inception import get_inception_score
from data import cifar10_data
import sys

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--learning_rate_disc', type=float, default=0.001)
parser.add_argument('--learning_rate_gen', type=float, default=0.001)
parser.add_argument('--learning_rate_decay', type=float, default=0.9995)
parser.add_argument('--data_dir', type=str, default='/home/tim/data')
parser.add_argument('--save_dir', type=str, default='/local_home/tim/ot_gan')
parser.add_argument('--optimizer', type=str, default='adamax')
parser.add_argument('--nonlinearity', type=str, default='crelu')
parser.add_argument('--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
parser.add_argument('--nr_gen_per_disc', type=int, default=3, help='How many times to update the generator for each update of the discriminator?')
parser.add_argument('--nr_sinkhorn_iter', type=int, default=500)
parser.add_argument('--matching_entropy', type=float, default=-np.log(0.5))
parser.add_argument('--wasserstein_p', type=int, default=1)
parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--load_params', dest='load_params', action='store_true')
parser.add_argument('--model_name', type=str, default='ot_gan_params-999')
parser.add_argument('--nr_epochs', type=int, default=10000)
args = parser.parse_args()
assert args.nr_gpu % 2 == 0
half_ngpu = args.nr_gpu // 2
print(args)
bs_per_gpu = args.batch_size // args.nr_gpu

if args.model == 'dcgan':
    from models.dcgan import generator, discriminator
elif args.model == 'densenet':
    from models.densenet import generator, discriminator

# extract model settings
model_opts = {'batch_size': bs_per_gpu, 'nonlinearity': args.nonlinearity}

# fix random seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# run once for data dependent initialization of parameters
x_init = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
f = discriminator(x_init, init=True, **model_opts)
generator(init=True, **model_opts)
num_features = f.get_shape().as_list()[-1]
print("model has a hidden representation with %d features" % num_features)

# get list of all params
all_params = tf.trainable_variables()
saver = tf.train.Saver(all_params)
disc_params = [p for p in all_params if 'discriminator' in p.name]
gen_params = [p for p in all_params if 'generator' in p.name]
ema = tf.train.ExponentialMovingAverage(decay=0.999)
maintain_averages_op = ema.apply(gen_params)

# data placeholders
x_data = [tf.placeholder(tf.float32, shape=(bs_per_gpu, 32, 32, 3)) for i in range(args.nr_gpu)]

# generate samples
x_gens = []
x_gens_ema = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        x_gens.append(generator(**model_opts))
        x_gens_ema.append(generator(ema=ema, **model_opts))

# feature extraction
features_dat = []
features_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        features_dat.append(discriminator(x_data[i], **model_opts))
        features_gen.append(discriminator(x_gens[i], **model_opts))

# global scale normalization
avg_square_norms = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        sqn_dat = tf.reduce_mean(tf.square(features_dat[i]))
        sqn_gen = tf.reduce_mean(tf.square(features_gen[i]))
        avg_square_norms.append(sqn_dat+sqn_gen)
normalizer = tf.sqrt(sum(avg_square_norms)/(2.*args.nr_gpu))
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        features_dat[i] /= normalizer
        features_gen[i] /= normalizer

# match samples and get loss
loss, entropies = minibatch_energy_distance(features_gen, features_dat, args.matching_entropy, args.nr_sinkhorn_iter, args.wasserstein_p)

# get gradients for generator and discriminator
grads_gen = tf.gradients(loss, gen_params, colocate_gradients_with_ops=True)
grads_disc = tf.gradients(-loss, disc_params, colocate_gradients_with_ops=True)

# get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    if args.optimizer == 'adam':
        gen_optimizer = nn.adam_updates(gen_params, grads_gen, lr=tf_lr, mom1=0., mom2=0.99)
        disc_optimizer = nn.adam_updates(disc_params, grads_disc, lr=tf_lr, mom1=0., mom2=0.99)
    elif args.optimizer == 'adamax':
        gen_optimizer = nn.adamax_updates(gen_params, grads_gen, lr=tf_lr, mom1=0., mom2=0.99)
        disc_optimizer = nn.adamax_updates(disc_params, grads_disc, lr=tf_lr, mom1=0., mom2=0.99)
    elif args.optimizer == 'nesterov':
        gen_optimizer = nn.nesterov_updates(gen_params, grads_gen, lr=tf_lr, mom1=0.5)
        disc_optimizer = nn.nesterov_updates(disc_params, grads_disc, lr=tf_lr, mom1=0.5)
    else:
        raise ('unsupported optimizer')

# init
initializer = tf.global_variables_initializer()

# load CIFAR-10 data
trainx, trainy = cifar10_data.load(args.data_dir + '/cifar-10-python')
trainx = np.transpose(trainx, (0,2,3,1))/127.5 - 1.
nr_batches_train_per_gpu = trainx.shape[0]//args.batch_size
testx, testy = cifar10_data.load(args.data_dir + '/cifar-10-python', subset='test')
testx = np.transpose(testx, (0,2,3,1))/127.5 - 1.


# //////////// perform training //////////////
try:
    os.stat(args.save_dir)
except:
    os.mkdir(args.save_dir)
print('starting training')
step_counter = 0

test_batches_per_gpu = 50000 // args.batch_size + 1
max_inception_score = 0
max_iter = 0
current_epoch = 0

with tf.Session() as sess:
    sess.run(initializer, feed_dict={x_init: trainx[:bs_per_gpu]})
    if args.load_params:
        saver.restore(sess, os.path.join(args.save_dir, args.model_name))
        ix = args.model_name.rfind('-')
        current_epoch = int(args.model_name[ix + 1:])

    start_time = time.time()
    for epoch in range(current_epoch, args.nr_epochs):
        begin = time.time()
        lr_gen = args.learning_rate_gen * args.learning_rate_decay ** epoch
        lr_disc = args.learning_rate_disc * args.learning_rate_decay ** epoch

        # randomly permute
        inds = np.random.permutation(trainx.shape[0])
        trainx = trainx[inds]

        # train
        np_distances = []
        np_entropies = [[] for i in range(6)]
        for t in range(nr_batches_train_per_gpu):
            feed_dict = {}
            for i in range(args.nr_gpu):
                td = t + i * nr_batches_train_per_gpu
                feed_dict[x_data[i]] = trainx[td * bs_per_gpu:(td + 1) * bs_per_gpu]

            # train discriminator once every args.nr_gen_per_disc batches
            if step_counter % (args.nr_gen_per_disc + 1) == 0:
                feed_dict.update({tf_lr: lr_disc})
                npd, e, _ = sess.run([loss, entropies, disc_optimizer], feed_dict=feed_dict)
                step_counter += 1
                np_distances.append(npd)
                for i in range(6):
                    np_entropies[i].append(e[i])

            else:  # train generator
                feed_dict.update({tf_lr: lr_gen})
                npd, e, _, _ = sess.run([loss, entropies, gen_optimizer, maintain_averages_op], feed_dict=feed_dict)
                step_counter += 1
                np_distances.append(npd)
                for i in range(6):
                    np_entropies[i].append(e[i])

        # log
        log_str = "Iteration %d, time = %ds, train distance = %.6f, entropies" % (epoch, time.time()-begin, np.mean(np_distances))
        for el in np_entropies:
            log_str += ' %.4f' % np.mean(el)
        print(log_str)

        # save generated image
        sample_x = sess.run(x_gens)
        sample_x = np.concatenate(sample_x)
        img_tile = plotting.img_tile(sample_x[:100], aspect_ratio=1.0, border_color=1.0, stretch=False)
        plotting.save_tile_img(img_tile, os.path.join(args.save_dir, 'sample%d.png' % epoch))

        # save EMA generated image
        sample_x_ema = sess.run(x_gens_ema)
        sample_x_ema = np.concatenate(sample_x_ema)
        img_tile = plotting.img_tile(sample_x_ema[:100], aspect_ratio=1.0, border_color=1.0, stretch=False)
        plotting.save_tile_img(img_tile, os.path.join(args.save_dir, 'ema_sample%d.png' % epoch))

        # save parameters + historical distances
        # calculate inception score
        if (epoch + 1) % 100 == 0 and epoch != current_epoch:
            sample_x = []
            sample_x_ema = []
            for t in range(test_batches_per_gpu):
                sample_x_temp = sess.run(x_gens)
                sample_x_temp = np.concatenate(sample_x_temp)
                sample_x_ema_temp = sess.run(x_gens_ema)
                sample_x_ema_temp = np.concatenate(sample_x_ema_temp)
                sample_x.append(sample_x_temp)
                sample_x_ema.append(sample_x_ema_temp)
            sample_x = np.concatenate(sample_x)
            sample_x_ema = np.concatenate(sample_x_ema)

            sample_x = [127.5*(sample_x[i]+1.) for i in range(sample_x.shape[0])]
            sample_x_ema = [127.5 * (sample_x_ema[i] + 1.) for i in range(sample_x_ema.shape[0])]
            inception_score = get_inception_score(sample_x[:50000], splits=10)
            print('inception score was %.6f, std was %.3f' % (inception_score[0], inception_score[1]))
            if inception_score[0] > max_inception_score:
                max_inception_score = inception_score[0]
                max_iter = epoch
            inception_score = get_inception_score(sample_x_ema[:50000], splits=10)
            print('EMA inception score was %.6f, std was %.3f ' % (inception_score[0], inception_score[1]))
            if inception_score[0] > max_inception_score:
                max_inception_score = inception_score[0]
                max_iter = epoch
            print('max inception score was %.6f, iter was %d' % (max_inception_score, max_iter))
            sys.stdout.flush()

        if (epoch + 1) % 200 == 0 and epoch != current_epoch:
            saver.save(sess, os.path.join(args.save_dir, 'ot_gan_params'), global_step=epoch)
            print('current epoch %d, elapsed hours from start epoch %.3f, total updates %d' % (
                epoch, (time.time()-start_time)/3600, step_counter
            ))
            sys.stdout.flush()

