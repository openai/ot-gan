import tensorflow as tf


def get_matched_features(features_a, features_b, sinkhorn_lambda, nr_sinkhorn_iter):

    n = features_a.get_shape().as_list()[1]
    fa_batch1, fa_batch2 = tf.split(features_a, 2, axis=0)
    fb_batch1, fb_batch2 = tf.split(features_b, 2, axis=0)

    # calculate all distances
    dist_a1_a2 = []
    dist_b2_b1 = []
    dist_a1_b1 = []
    dist_a1_b2 = []
    dist_a2_b1 = []
    dist_a2_b2 = []
    asq = 0.5 * tf.reduce_mean(tf.square(fa_batch1), axis=1, keep_dims=True)
    dist_a1_a2.append(
        asq + 0.5 * tf.reshape(tf.reduce_mean(tf.square(fa_batch2), axis=1), [1, -1]) - tf.matmul(fa_batch1,
                                                                                                  fa_batch2,
                                                                                                  transpose_b=True) / n)
    dist_a1_b1.append(
        asq + 0.5 * tf.reshape(tf.reduce_mean(tf.square(fb_batch1), axis=1), [1, -1]) - tf.matmul(fa_batch1,
                                                                                                  fb_batch1,
                                                                                                  transpose_b=True) / n)
    dist_a1_b2.append(
        asq + 0.5 * tf.reshape(tf.reduce_mean(tf.square(fb_batch2), axis=1), [1, -1]) - tf.matmul(fa_batch1,
                                                                                                  fb_batch2,
                                                                                                  transpose_b=True) / n)

    asq = 0.5 * tf.reduce_mean(tf.square(fa_batch2), axis=1, keep_dims=True)
    bsq = 0.5 * tf.reduce_mean(tf.square(fb_batch2), axis=1, keep_dims=True)

    dist_a2_b1.append(
        asq + 0.5 * tf.reshape(tf.reduce_mean(tf.square(fb_batch1), axis=1), [1, -1]) - tf.matmul(fa_batch2,
                                                                                                  fb_batch1,
                                                                                                  transpose_b=True) / n)
    dist_a2_b2.append(
        asq + 0.5 * tf.reshape(tf.reduce_mean(tf.square(fb_batch2), axis=1), [1, -1]) - tf.matmul(fa_batch2,
                                                                                                  fb_batch2,
                                                                                                  transpose_b=True) / n)
    dist_b2_b1.append(
        bsq + 0.5 * tf.reshape(tf.reduce_mean(tf.square(fb_batch1), axis=1), [1, -1]) - tf.matmul(fb_batch2,
                                                                                                  fb_batch1,
                                                                                                  transpose_b=True) / n)

    distances = [tf.concat(dist_a1_a2, 0), tf.concat(dist_b2_b1, 0),
                 tf.concat(dist_a1_b1, 0), tf.concat(dist_a1_b2, 0),
                 tf.concat(dist_a2_b1, 0), tf.concat(dist_a2_b2, 0)]

    # use Sinkhorn algorithm to do soft assignment
    assignments = []
    entropy = []
    for i in range(len(distances)):
        log_a = -sinkhorn_lambda * distances[i]
        for it in range(nr_sinkhorn_iter):
            log_a -= tf.reduce_logsumexp(log_a, axis=1, keep_dims=True)
            log_a -= tf.reduce_logsumexp(log_a, axis=0, keep_dims=True)

        assignments.append(tf.nn.softmax(log_a))
        entropy.append(
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=assignments[-1], logits=log_a)))

    assignment_a1_a2, assignment_b2_b1, assignment_a1_b1, assignment_a1_b2, \
    assignment_a2_b1, assignment_a2_b2 = assignments
    entropy = sum(entropy) / len(entropy)

    # get matched features
    features_a1_a2_matched = tf.matmul(assignment_a1_a2, fa_batch2)
    features_b1_b2_matched = tf.matmul(assignment_b2_b1, fb_batch2, transpose_a=True)
    features_a1_b1_matched = tf.matmul(assignment_a1_b1, fb_batch1)
    features_a1_b2_matched = tf.matmul(assignment_a1_b2, fb_batch2)
    features_a2_b1_matched = tf.matmul(assignment_a2_b1, fb_batch1)
    features_a2_b2_matched = tf.matmul(assignment_a2_b2, fb_batch2)
    features_a2_a1_matched = tf.matmul(assignment_a1_a2, fa_batch1, transpose_a=True)
    features_b2_b1_matched = tf.matmul(assignment_b2_b1, fb_batch1)
    features_b1_a1_matched = tf.matmul(assignment_a1_b1, fa_batch1, transpose_a=True)
    features_b2_a1_matched = tf.matmul(assignment_a1_b2, fa_batch1, transpose_a=True)
    features_b1_a2_matched = tf.matmul(assignment_a2_b1, fa_batch2, transpose_a=True)
    features_b2_a2_matched = tf.matmul(assignment_a2_b2, fa_batch2, transpose_a=True)


    features_a_a = tf.concat([features_a1_a2_matched, features_a2_a1_matched], axis=0)
    features_b_b = tf.concat([features_b1_b2_matched, features_b2_b1_matched], axis=0)

    features_a_b = tf.concat([features_a1_b1_matched, features_a2_b1_matched], axis=0) + \
                   tf.concat([features_a1_b2_matched, features_a2_b2_matched], axis=0)
    features_a_b = features_a_b * 0.5

    features_b_a = tf.concat([features_b1_a1_matched, features_b2_a1_matched], axis=0) + \
                   tf.concat([features_b1_a2_matched, features_b2_a2_matched], axis=0)

    features_b_a = features_b_a * 0.5

    return features_a_a, features_b_b, features_a_b, features_b_a, entropy


def get_matched_features_single_batch(features_a, features_b, sinkhorn_lambda, nr_sinkhorn_iter, batch_size):
    """ simplified, more efficient, but slightly wrong, version of the original (two-batch) matching code """

    ngpu = len(features_a)
    # batch_size = features_a[0].get_shape().as_list()[0]
    n = features_a[0].get_shape().as_list()[1]

    # gather all features
    fa_all = tf.concat(features_a, axis=0)
    fa_all_sq = 0.5 * tf.reshape(tf.reduce_mean(tf.square(fa_all), axis=1), [1, -1])
    fb_all = tf.concat(features_b, axis=0)
    fb_all_sq = 0.5 * tf.reshape(tf.reduce_mean(tf.square(fb_all), axis=1), [1, -1])

    # calculate all distances
    dist_a_a = []
    dist_b_b = []
    dist_a_b = []
    for i in range(ngpu):
        with tf.device('/gpu:%d' % i):
            asq = 0.5 * tf.reduce_mean(tf.square(features_a[i]), axis=1, keep_dims=True)
            bsq = 0.5 * tf.reduce_mean(tf.square(features_b[i]), axis=1, keep_dims=True)
            dist_a_a.append(asq + fa_all_sq - tf.matmul(features_a[i], fa_all, transpose_b=True) / n)
            dist_b_b.append(bsq + fb_all_sq - tf.matmul(features_b[i], fb_all, transpose_b=True) / n)
            dist_a_b.append(asq + fb_all_sq - tf.matmul(features_a[i], fb_all, transpose_b=True) / n)

    # combine results + add a bit to the diagonal to prevent self-matches
    distances = [tf.concat(dist_a_a, 0) + 999. * tf.eye(batch_size),
                 tf.concat(dist_b_b, 0) + 999. * tf.eye(batch_size),
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
            entropy.append(
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=assignments[-1], logits=log_a)))

    assignment_a_a, assignment_b_b, assignment_a_b = assignments
    entropy = sum(entropy) / len(entropy)

    # get matched features
    features_a_a = tf.split(tf.matmul(assignment_a_a, fa_all), ngpu, 0)
    features_b_b = tf.split(tf.matmul(assignment_b_b, fb_all), ngpu, 0)
    features_a_b = tf.split(tf.matmul(assignment_a_b, fb_all), ngpu, 0)
    features_b_a = tf.split(tf.matmul(assignment_a_b, fa_all, transpose_a=True), ngpu, 0)

    return features_a_a, features_b_b, features_a_b, features_b_a, entropy


def calc_distance(features_a, features_b, matched_features):
    features_a_a, features_b_b, features_a_b, features_b_a, _ = matched_features

    nd_a_a = tf.reduce_mean(features_a * features_a_a)
    nd_b_b = tf.reduce_mean(features_b * features_b_b)
    nd_a_b = tf.reduce_mean(features_a * features_a_b)
    total_dist = nd_b_b + nd_a_a - 2. * nd_a_b

    total_dist = total_dist / (2.)
    return total_dist
