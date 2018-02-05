import tensorflow as tf

def safe_sqrt(x):
    return tf.sqrt(tf.maximum(x, 1e-8))

def sinkhorn_distance(x, y, target_entropy, nr_sinkhorn_iter, wasserstein_p=1):

    # globally normalize features to avoid blowing up the cost
    x /= tf.sqrt(tf.reduce_mean(tf.square(x)))
    y /= tf.sqrt(tf.reduce_mean(tf.square(y)))

    # calculate transport cost matrix
    yt = tf.transpose(y,(1,0))
    n,k = x.get_shape().as_list()
    cost = tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + tf.reduce_mean(tf.square(yt), axis=0, keepdims=True) - (2. / k) * tf.matmul(x, yt)
    if wasserstein_p == 1:
        cost = safe_sqrt(cost)

    # use Sinkhorn algorithm to do soft assignment
    sinkhorn_inv_lambda = tf.Variable(500., dtype=tf.float32, trainable=False)
    log_a = -sinkhorn_inv_lambda * cost
    for it in range(nr_sinkhorn_iter):
        log_a -= tf.reduce_logsumexp(log_a, axis=1, keepdims=True)
        log_a -= tf.reduce_logsumexp(log_a, axis=0, keepdims=True)

    # adjust Sinkhorn lambda
    M = tf.stop_gradient((tf.nn.softmax(log_a, axis=1) + tf.nn.softmax(log_a, axis=0)) / 2.)
    H = -tf.reduce_sum(M * log_a) / n
    delta_H = H - target_entropy
    delta_H *= (10. / tf.maximum(abs(delta_H), 10.))
    new_inv_lambda = tf.minimum(sinkhorn_inv_lambda * tf.exp(0.1 * delta_H), 100000.)

    # final Sinkhorn distance
    with tf.control_dependencies([sinkhorn_inv_lambda.assign(new_inv_lambda)]):
        W = tf.reduce_sum(M * cost) / n
        if wasserstein_p == 2:
            W = tf.sqrt(W)

    return W,H

def minibatch_energy_distance(features_a, features_b, target_entropy, nr_sinkhorn_iter, wasserstein_p=1):
    assert isinstance(features_a, list)
    ngpu = len(features_a)
    assert len(features_b) == ngpu
    half_ngpu = ngpu // 2

    # gather features, split into two batches
    fa1 = tf.concat(features_a[:half_ngpu], axis=0)
    fa2 = tf.concat(features_a[half_ngpu:], axis=0)
    fb1 = tf.concat(features_b[:half_ngpu], axis=0)
    fb2 = tf.concat(features_b[half_ngpu:], axis=0)

    # calculate all Sinkhorn distances
    with tf.device('gpu:0'):
        W_a1_a2, H_a1_a2 = sinkhorn_distance(fa1, fa2, target_entropy, nr_sinkhorn_iter, wasserstein_p)
    with tf.device('gpu:%s' % (1 % ngpu)):
        W_b2_b1, H_b2_b1 = sinkhorn_distance(fb2, fb1, target_entropy, nr_sinkhorn_iter, wasserstein_p)
    with tf.device('gpu:%s' % (2 % ngpu)):
        W_a1_b1, H_a1_b1 = sinkhorn_distance(fa1, fb1, target_entropy, nr_sinkhorn_iter, wasserstein_p)
    with tf.device('gpu:%s' % (3 % ngpu)):
        W_a1_b2, H_a1_b2 = sinkhorn_distance(fa1, fb2, target_entropy, nr_sinkhorn_iter, wasserstein_p)
    with tf.device('gpu:%s' % (4 % ngpu)):
        W_a2_b1, H_a2_b1 = sinkhorn_distance(fa2, fb1, target_entropy, nr_sinkhorn_iter, wasserstein_p)
    with tf.device('gpu:%s' % (5 % ngpu)):
        W_a2_b2, H_a2_b2 = sinkhorn_distance(fa2, fb2, target_entropy, nr_sinkhorn_iter, wasserstein_p)

    # get minibatch energy distance
    med_loss = 0.5*(W_a1_b1 + W_a1_b2 + W_a2_b1 + W_a2_b2) - W_a1_a2 - W_b2_b1
    entropies = [H_a1_b1, H_a1_b2, H_a2_b1, H_a2_b2, H_a1_a2, H_b2_b1]
    
    return med_loss, entropies
