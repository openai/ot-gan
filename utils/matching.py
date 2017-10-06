import tensorflow as tf

def get_matched_features_random(features_a, features_b):
    features_a_a = features_a[1:] + features_a[:1]
    features_b_b = features_b[1:] + features_b[:1]
    features_a_b = features_b
    features_b_a = features_a

    return features_a_a, features_b_b, features_a_b, features_b_a, tf.zeros(shape=[], dtype=tf.float32, name=None)

def get_matched_features(features_a, features_b, sinkhorn_lambda, nr_sinkhorn_iter):
    ngpu = len(features_a)
    half_ngpu = ngpu // 2

    # gather all features, split into two batches
    fa_batch1 = tf.concat(features_a[:half_ngpu],axis=0)
    fa_batch2 = tf.concat(features_a[half_ngpu:],axis=0)
    fb_batch1 = tf.concat(features_b[:half_ngpu],axis=0)
    fb_batch2 = tf.concat(features_b[half_ngpu:],axis=0)

    # calculate all distances
    dist_a1_a2 = []
    dist_b2_b1 = []
    dist_a1_b1 = []
    dist_a1_b2 = []
    dist_a2_b1 = []
    dist_a2_b2 = []
    
    for i in range(half_ngpu):
        with tf.device('/gpu:%d' % i):
            dist_a1_a2.append(1. - tf.matmul(features_a[i],fa_batch2,transpose_b=True))
            dist_a1_b1.append(1. - tf.matmul(features_a[i],fb_batch1,transpose_b=True))
            dist_a1_b2.append(1. - tf.matmul(features_a[i],fb_batch2,transpose_b=True))

    for i in range(half_ngpu,2*half_ngpu):
        with tf.device('/gpu:%d' % i):
            dist_a2_b1.append(1. - tf.matmul(features_a[i],fb_batch1,transpose_b=True))
            dist_a2_b2.append(1. - tf.matmul(features_a[i],fb_batch2,transpose_b=True))
            dist_b2_b1.append(1. - tf.matmul(features_b[i],fb_batch1,transpose_b=True))
    
    distances = [tf.concat(dist_a1_a2,0), tf.concat(dist_b2_b1,0),
                 tf.concat(dist_a1_b1,0), tf.concat(dist_a1_b2,0),
                 tf.concat(dist_a2_b1,0), tf.concat(dist_a2_b2,0)]
    
    # use Sinkhorn algorithm to do soft assignment
    assignments = []
    entropy = []
    for i in range(len(distances)):
        with tf.device('/gpu:%d' % (i%ngpu)):
            log_a = -sinkhorn_lambda * distances[i]
    
            for it in range(nr_sinkhorn_iter):
                log_a -= tf.reduce_logsumexp(log_a, axis=1, keep_dims=True)
                log_a -= tf.reduce_logsumexp(log_a, axis=0, keep_dims=True)
    
            assignments.append(tf.nn.softmax(log_a))
            entropy.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=assignments[-1], logits=log_a)))
    
    assignment_a1_a2, assignment_b2_b1, assignment_a1_b1, assignment_a1_b2, \
        assignment_a2_b1, assignment_a2_b2 = assignments
    entropy = sum(entropy)/len(entropy)
    
    # get matched features
    features_a1_a2_matched = tf.split(tf.matmul(assignment_a1_a2, fa_batch2), half_ngpu, 0)
    features_b1_b2_matched = tf.split(tf.matmul(assignment_b2_b1, fb_batch2, transpose_a=True), half_ngpu, 0)
    features_a1_b1_matched = tf.split(tf.matmul(assignment_a1_b1, fb_batch1), half_ngpu, 0)
    features_a1_b2_matched = tf.split(tf.matmul(assignment_a1_b2, fb_batch2), half_ngpu, 0)
    features_a2_b1_matched = tf.split(tf.matmul(assignment_a2_b1, fb_batch1), half_ngpu, 0)
    features_a2_b2_matched = tf.split(tf.matmul(assignment_a2_b2, fb_batch2), half_ngpu, 0)
    features_a2_a1_matched = tf.split(tf.matmul(assignment_a1_a2, fa_batch1, transpose_a=True), half_ngpu, 0)
    features_b2_b1_matched = tf.split(tf.matmul(assignment_b2_b1, fb_batch1), half_ngpu, 0)
    features_b1_a1_matched = tf.split(tf.matmul(assignment_a1_b1, fa_batch1, transpose_a=True), half_ngpu, 0)
    features_b2_a1_matched = tf.split(tf.matmul(assignment_a1_b2, fa_batch1, transpose_a=True), half_ngpu, 0)
    features_b1_a2_matched = tf.split(tf.matmul(assignment_a2_b1, fa_batch2, transpose_a=True), half_ngpu, 0)
    features_b2_a2_matched = tf.split(tf.matmul(assignment_a2_b2, fa_batch2, transpose_a=True), half_ngpu, 0)
    
    # combine
    features_a_a = features_a1_a2_matched + features_a2_a1_matched
    features_b_b = features_b1_b2_matched + features_b2_b1_matched
    features_a_b = [0.5*(f1+f2) for f1,f2 in zip(features_a1_b1_matched+features_a2_b1_matched,
                                                 features_a1_b2_matched+features_a2_b2_matched)]
    features_b_a = [0.5*(f1+f2) for f1, f2 in zip(features_b1_a1_matched+features_b2_a1_matched,
                                                 features_b1_a2_matched + features_b2_a2_matched)]
    
    return features_a_a, features_b_b, features_a_b, features_b_a, entropy


def get_matched_features_single_batch(features_a, features_b, sinkhorn_lambda, nr_sinkhorn_iter):
    """ simplified, more efficient, but slightly wrong, version of the original (two-batch) matching code """

    ngpu = len(features_a)
    batch_size = features_a[0].get_shape().as_list()[0]

    # gather all features
    fa_all = tf.concat(features_a, axis=0)
    fb_all = tf.concat(features_b, axis=0)

    # calculate all distances
    dist_a_a = []
    dist_b_b = []
    dist_a_b = []
    for i in range(ngpu):
        with tf.device('/gpu:%d' % i):
            dist_a_a.append(1. - tf.matmul(features_a[i], fa_all, transpose_b=True))
            dist_b_b.append(1. - tf.matmul(features_b[i], fb_all, transpose_b=True))
            dist_a_b.append(1. - tf.matmul(features_a[i], fb_all, transpose_b=True))

    # combine results + add a bit to the diagonal to prevent self-matches
    distances = [tf.concat(dist_a_a, 0) + 999. * tf.eye(ngpu * batch_size),
                 tf.concat(dist_b_b, 0) + 999. * tf.eye(ngpu * batch_size),
                 tf.concat(dist_a_b, 0)]

    # use Sinkhorn algorithm to do soft assignment
    assignments = []
    entropy = []
    for i in range(len(distances)):
        with tf.device('/gpu:%d' % (i % ngpu)):
            log_a = -sinkhorn_lambda * distances[i]

            for it in range(nr_sinkhorn_iter):
                log_a -= tf.reduce_logsumexp(log_a, axis=1, keep_dims=True)
                log_a -= tf.reduce_logsumexp(log_a, axis=0, keep_dims=True)

            assignments.append(tf.nn.softmax(log_a))
            entropy.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=assignments[-1], logits=log_a)))

    assignment_a_a, assignment_b_b, assignment_a_b = assignments
    entropy = sum(entropy) / len(entropy)

    # get matched features
    features_a_a = tf.split(tf.matmul(assignment_a_a, fa_all), ngpu, 0)
    features_b_b = tf.split(tf.matmul(assignment_b_b, fb_all), ngpu, 0)
    features_a_b = tf.split(tf.matmul(assignment_a_b, fb_all), ngpu, 0)
    features_b_a = tf.split(tf.matmul(assignment_a_b, fa_all, transpose_a=True), ngpu, 0)

    return features_a_a, features_b_b, features_a_b, features_b_a, entropy


def calc_distance(features_a, features_b, matched_features):
    ngpu = len(features_a)
    batch_size = features_a[0].get_shape().as_list()[0]
    features_a_a, features_b_b, features_a_b, features_b_a, _ = matched_features

    dist = []
    for i in range(ngpu):
        with tf.device('/gpu:%d' % i):
            nd_a_a = tf.reduce_sum(features_a[i] * features_a_a[i])
            nd_b_b = tf.reduce_sum(features_b[i] * features_b_b[i])
            nd_a_b = tf.reduce_sum(features_a[i] * features_a_b[i])
            dist.append(nd_b_b + nd_a_a - 2. * nd_a_b)
    
    total_dist = sum(dist) / (2 * batch_size * ngpu)
    return total_dist
