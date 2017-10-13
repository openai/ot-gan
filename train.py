import argparse
import time
import os
import numpy as np
import tensorflow as tf
from utils import nn
from utils import plotting
from utils import matching
from utils.inception import get_inception_score
from data import cifar10_data
import sys

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=625)
parser.add_argument('--learning_rate_disc', type=float, default=0.0003)
parser.add_argument('--learning_rate_gen', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='/home/tim/data')
parser.add_argument('--save_dir', type=str, default='/local_home/tim/med_gan')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--nonlinearity', type=str, default='crelu')
parser.add_argument('--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
parser.add_argument('--nr_gen_per_disc', type=int, default=5, help='How many times to update the generator for each update of the discriminator?')
parser.add_argument('--sinkhorn_lambda', type=float, default=500.)
parser.add_argument('--nr_sinkhorn_iter', type=int, default=500)
parser.add_argument('--single_batch', dest='single_batch', action='store_true', help='Use simplified batching using a single batch instead of 2')
parser.add_argument('--train_disc_against_ema', dest='train_disc_against_ema', action='store_true', help='Should discriminator be trained against samples of EMA generator?')
parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--load_params', dest='load_params', action='store_true')
parser.add_argument('--model_name', type=str, default='med_gan_params-2399')
parser.add_argument('--no_sinkhorn', dest='no_sinkhorn', action='store_true')
args = parser.parse_args()
assert args.nr_gpu % 2 == 0
half_ngpu = args.nr_gpu // 2
print(args)

if args.model == 'dcgan':
    from models.dcgan import generator, discriminator
elif args.model == 'densenet':
    from models.densenet import generator, discriminator


# extract model settings
model_opts = {'batch_size': args.batch_size, 'nonlinearity': args.nonlinearity}

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
x_data = [tf.placeholder(tf.float32, shape=(None, 32, 32, 3)) for i in range(args.nr_gpu)]

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
features_gen_ema = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        features_dat.append(discriminator(x_data[i], **model_opts))
        features_gen.append(discriminator(x_gens[i], **model_opts))
        features_gen_ema.append(discriminator(x_gens_ema[i], **model_opts))

# match samples and get features
if args.single_batch:
    features_matched = matching.get_matched_features_single_batch(features_gen, features_dat, args.sinkhorn_lambda, args.nr_sinkhorn_iter)
    features_matched_ema = matching.get_matched_features_single_batch(features_gen_ema, features_dat, args.sinkhorn_lambda, args.nr_sinkhorn_iter)
elif args.no_sinkhorn:
    print('using random image matching')
    features_matched = matching.get_matched_features_random(features_gen, features_dat)
    features_matched_ema = matching.get_matched_features_random(features_gen_ema, features_dat)
else:
    features_matched = matching.get_matched_features(features_gen, features_dat, args.sinkhorn_lambda, args.nr_sinkhorn_iter)
    features_matched_ema = matching.get_matched_features(features_gen_ema, features_dat, args.sinkhorn_lambda, args.nr_sinkhorn_iter)
avg_entropy = features_matched[-1]

# get distances
total_dist_gen = matching.calc_distance(features_gen, features_dat, features_matched)
if args.train_disc_against_ema:
    total_dist_disc = matching.calc_distance(features_gen_ema, features_dat, features_matched_ema)
else:
    total_dist_disc = total_dist_gen

# get gradients for generator
grads_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        grad_features_gen_i = features_matched[0][i] - features_matched[2][i]
        grad_gen_i = tf.gradients(ys=features_gen[i], xs=gen_params, grad_ys=grad_features_gen_i)
        grads_gen.append(grad_gen_i)

# get gradients for discriminator (potentially uses EMA samples!)
grads_disc = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        if args.train_disc_against_ema:
            grad_features_gen_i = features_matched_ema[0][i] - features_matched_ema[2][i]
            grad_features_dat_i = features_matched_ema[1][i] - features_matched_ema[3][i]
            grad_disc_i = tf.gradients(ys=[features_dat[i], features_gen_ema[i]], xs=disc_params,
                                       grad_ys=[grad_features_dat_i, grad_features_gen_i])
        else:
            grad_features_gen_i = features_matched[0][i] - features_matched[2][i]
            grad_features_dat_i = features_matched[1][i] - features_matched[3][i]
            grad_disc_i = tf.gradients(ys=[features_dat[i], features_gen[i]], xs=disc_params,
                                       grad_ys=[grad_features_dat_i, grad_features_gen_i])

        grads_disc.append(grad_disc_i)

# add gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        for j in range(len(grads_gen[0])):
            grads_gen[0][j] += grads_gen[i][j]
        for j in range(len(grads_disc[0])):
            grads_disc[0][j] += grads_disc[i][j]

    if args.optimizer == 'adam':
        gen_optimizer = nn.adam_updates(gen_params, grads_gen[0], lr=tf_lr, mom1=0.5, mom2=0.999)
        disc_optimizer = nn.adam_updates(disc_params, grads_disc[0], lr=-tf_lr, mom1=0.5, mom2=0.999)
    elif args.optimizer == 'adamax':
        gen_optimizer = nn.adamax_updates(gen_params, grads_gen[0], lr=tf_lr, mom1=0.5, mom2=0.999)
        disc_optimizer = nn.adamax_updates(disc_params, grads_disc[0], lr=-tf_lr, mom1=0.5, mom2=0.999)
    elif args.optimizer == 'nesterov':
        gen_optimizer = nn.nesterov_updates(gen_params, grads_gen[0], lr=tf_lr, mom1=0.5)
        disc_optimizer = nn.nesterov_updates(disc_params, grads_disc[0], lr=-tf_lr, mom1=0.5)
    else:
        raise('unsupported optimizer')

# init
initializer = tf.global_variables_initializer()

# load CIFAR-10 data
trainx, trainy = cifar10_data.load(args.data_dir + '/cifar-10-python')
trainx = np.transpose(trainx, (0,2,3,1))/127.5 - 1.
nr_batches_train_per_gpu = trainx.shape[0]//(args.nr_gpu*args.batch_size)
testx, testy = cifar10_data.load(args.data_dir + '/cifar-10-python', subset='test')
testx = np.transpose(testx, (0,2,3,1))/127.5 - 1.

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
mean_dist_gen = []
mean_dist_disc = []

dis_updates = 0
test_batches_per_gpu = 50000 // (args.nr_gpu*args.batch_size) + 1
max_inception_score = 0
max_iter = 0
current_epoch = 0

with tf.Session() as sess:
    sess.run(initializer, feed_dict={x_init: trainx[:args.batch_size]})
    if args.load_params:
        saver.restore(sess, os.path.join(args.save_dir, args.model_name))
        ix = args.model_name.rfind('-')
        current_epoch = int(args.model_name[ix + 1:])

    start_time = time.time()
    for epoch in range(current_epoch, 1000000):
        begin = time.time()

        # randomly permute
        inds = np.random.permutation(trainx.shape[0])
        trainx = trainx[inds]

        # train
        np_distances_gen = []
        np_distances_disc = []
        np_entropy = []
        for t in range(nr_batches_train_per_gpu):
            feed_dict = {}
            for i in range(args.nr_gpu):
                td = t + i * nr_batches_train_per_gpu
                feed_dict[x_data[i]] = maybe_flip(trainx[td * args.batch_size:(td + 1) * args.batch_size])

            # train discriminator once every args.nr_gen_per_disc batches
            if step_counter % (args.nr_gen_per_disc+1) == 0:
                feed_dict.update({tf_lr: args.learning_rate_disc})
                npd_ema, e, _ = sess.run([total_dist_disc, avg_entropy, disc_optimizer], feed_dict=feed_dict)
                step_counter += 1
                np_distances_disc.append(npd_ema)
                np_entropy.append(e)

            else: # train generator
                feed_dict.update({tf_lr: args.learning_rate_gen})
                npd, e, _, _ = sess.run([total_dist_gen, avg_entropy, gen_optimizer, maintain_averages_op], feed_dict=feed_dict)
                step_counter += 1
                np_distances_gen.append(npd)
                np_entropy.append(e)

        # log
        mean_dist_gen.append(np.mean(np_distances_gen))
        mean_dist_disc.append(np.mean(np_distances_disc))
        print("Iteration %d, time = %ds, train distance before gen = %.6f, train distance before disc = %.6f, avg matching entropy = %.6f" % (epoch, time.time()-begin, mean_dist_gen[-1], mean_dist_disc[-1], np.mean(np_entropy)))

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

        if (epoch + 1) % 100 == 0 and epoch != current_epoch:
            # save parameters + historical distances
            # calculate inception score
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
            saver.save(sess, os.path.join(args.save_dir, 'med_gan_params'), global_step=epoch)
            np.savez(os.path.join(args.save_dir, 'distances.npz'), mean_dist_gen=np.array(mean_dist_gen), mean_dist_disc=np.array(mean_dist_disc))
            print('current epoch %d, elapsed hours from start epoch %.3f, discriminator updates %d, total updates %d' % (
                epoch, (time.time()-start_time)/3600, dis_updates, step_counter
            ))
            sys.stdout.flush()

