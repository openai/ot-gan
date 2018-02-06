import tensorflow as tf
import horovod.tensorflow as hvd
from horovod.tensorflow import allgather, allreduce
rnk = hvd.rank()

def minibatch_energy_distance(features_a, features_b, sinkhorn_inv_lambda, nr_sinkhorn_iter):
    n,k = features_a.get_shape().as_list()
    half_n = n//2

    # split features into two batches, gather, calculate transport costs
    fa1 = features_a[:half_n]
    fa2 = features_a[half_n:]
    fb1 = features_b[:half_n]
    fb2 = features_b[half_n:]
    all_fa2 = allgather(fa2)
    all_fb1 = allgather(fb1)
    all_fb2 = allgather(fb2)
    costs = {}
    costs['a1_a2'] = 1. - (1. / k) * tf.matmul(fa1, all_fa2, transpose_b=True)
    costs['a1_b1'] = 1. - (1. / k) * tf.matmul(fa1, all_fb1, transpose_b=True)
    costs['a1_b2'] = 1. - (1. / k) * tf.matmul(fa1, all_fb2, transpose_b=True)
    costs['a2_b1'] = 1. - (1. / k) * tf.matmul(fa2, all_fb1, transpose_b=True)
    costs['a2_b2'] = 1. - (1. / k) * tf.matmul(fa2, all_fb2, transpose_b=True)
    costs['b2_b1'] = 1. - (1. / k) * tf.matmul(fb2, all_fb1, transpose_b=True)

    # use Sinkhorn algorithm to do soft assignment
    assignments = {}
    assignments_transposed = {}
    entropies = []
    summed_costs = {}
    for k, C in costs.items():
        log_a = -sinkhorn_inv_lambda * C

        for it in range(nr_sinkhorn_iter):
            # rows
            log_a -= tf.reduce_logsumexp(log_a, axis=1, keepdims=True)

            # cols
            log_sum_col = tf.reduce_logsumexp(log_a, axis=0, keepdims=True)
            all_log_sum_col = allgather(log_sum_col)
            log_a -= tf.reduce_logsumexp(all_log_sum_col, axis=0, keepdims=True)

        assignments[k] = tf.nn.softmax(log_a) # normalized rows
        assignments_transposed[k] = tf.exp(log_a) # normalized cols
        entropies.append(allreduce(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=assignments[k], logits=log_a)), average=True))
        summed_costs[k] = allreduce(tf.reduce_sum(assignments[k]*C), average=False)

    med_loss = 0.5 * (summed_costs['a1_b1'] + summed_costs['a1_b2'] + summed_costs['a2_b1'] + summed_costs['a2_b2']) \
               - summed_costs['a1_a2'] - summed_costs['b2_b1'] # ignoring the entropies in the sinkhorn distance

    # get all matched features for calculating gradients
    features_a1_a2_matched = tf.matmul(assignments['a1_a2'], all_fa2)
    features_a1_b1_matched = tf.matmul(assignments['a1_b1'], all_fb1)
    features_a1_b2_matched = tf.matmul(assignments['a1_b2'], all_fb2)
    features_a2_b1_matched = tf.matmul(assignments['a2_b1'], all_fb1)
    features_a2_b2_matched = tf.matmul(assignments['a2_b2'], all_fb2)
    features_b2_b1_matched = tf.matmul(assignments['b2_b1'], all_fb1)

    # horovod does not support reduce, so being a bit inefficient here
    features_b1_b2_matched = allreduce(tf.matmul(assignments_transposed['b2_b1'], fb2, transpose_a=True),
                                       average=False)[rnk*half_n:(rnk+1)*half_n]
    features_a2_a1_matched = allreduce(tf.matmul(assignments_transposed['a1_a2'], fa1, transpose_a=True),
                                       average=False)[rnk*half_n:(rnk+1)*half_n]
    features_b1_a1_matched = allreduce(tf.matmul(assignments_transposed['a1_b1'], fa1, transpose_a=True),
                                       average=False)[rnk*half_n:(rnk+1)*half_n]
    features_b2_a1_matched = allreduce(tf.matmul(assignments_transposed['a1_b2'], fa1, transpose_a=True),
                                       average=False)[rnk*half_n:(rnk+1)*half_n]
    features_b1_a2_matched = allreduce(tf.matmul(assignments_transposed['a2_b1'], fa2, transpose_a=True),
                                       average=False)[rnk*half_n:(rnk+1)*half_n]
    features_b2_a2_matched = allreduce(tf.matmul(assignments_transposed['a2_b2'], fa2, transpose_a=True),
                                       average=False)[rnk*half_n:(rnk+1)*half_n]

    features_a_a = tf.concat([features_a1_a2_matched, features_a2_a1_matched],axis=0)
    features_b_b = tf.concat([features_b1_b2_matched, features_b2_b1_matched],axis=0)
    features_a_b = 0.5 * (tf.concat([features_a1_b1_matched, features_a2_b1_matched],axis=0)
                          + tf.concat([features_a1_b2_matched, features_a2_b2_matched],axis=0))
    features_b_a = 0.5 * (tf.concat([features_b1_a1_matched, features_b2_a1_matched],axis=0)
                          + tf.concat([features_b1_a2_matched, features_b2_a2_matched],axis=0))

    return med_loss, entropies, features_a_b, features_a_a, features_b_a, features_b_b
