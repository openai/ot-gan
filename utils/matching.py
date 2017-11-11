import tensorflow as tf

def safe_sqrt(x):
    return tf.sqrt(tf.maximum(x, 1e-8))

def minibatch_energy_distance(features_a, features_b, target_entropy, nr_sinkhorn_iter, wasserstein_p=1):
    ngpu = len(features_a)
    half_ngpu = ngpu // 2
    n,k = features_a[0].get_shape().as_list()
    sinkhorn_inv_lambda = [tf.Variable(500., dtype=tf.float32, trainable=False) for i in range(6)]

    # calculate norms of features
    fa_norm = []
    fb_norm = []
    for i in range(ngpu):
        with tf.device('/gpu:%d' % i):
            fa_norm.append(tf.reduce_mean(tf.square(features_a[i]), axis=1, keep_dims=True))
            fb_norm.append(tf.reduce_mean(tf.square(features_b[i]), axis=1, keep_dims=True))

    # gather features, split into two batches
    with tf.device('/gpu:0'):
        fa_batch2 = tf.concat(features_a[half_ngpu:], axis=0)
        fa_batch2_norm = tf.reshape(tf.concat(fa_norm[half_ngpu:], axis=0), (1,half_ngpu*n))
        fb_batch1 = tf.concat(features_b[:half_ngpu], axis=0)
        fb_batch1_norm = tf.reshape(tf.concat(fb_norm[:half_ngpu], axis=0), (1,half_ngpu*n))
        fb_batch2 = tf.concat(features_b[half_ngpu:], axis=0)
        fb_batch2_norm = tf.reshape(tf.concat(fb_norm[half_ngpu:], axis=0), (1,half_ngpu*n))

    # calculate all transport costs
    cost_a1_a2 = []
    cost_a1_b1 = []
    cost_a1_b2 = []
    cost_a2_b1 = []
    cost_a2_b2 = []
    cost_b2_b1 = []

    for i in range(half_ngpu):
        with tf.device('/gpu:%d' % i):
            cost_a1_a2.append(fa_norm[i] + fa_batch2_norm - (2./k)*tf.matmul(features_a[i],fa_batch2,transpose_b=True))
            cost_a1_b1.append(fa_norm[i] + fb_batch1_norm - (2./k)*tf.matmul(features_a[i],fb_batch1,transpose_b=True))
            cost_a1_b2.append(fa_norm[i] + fb_batch2_norm - (2./k)*tf.matmul(features_a[i],fb_batch2,transpose_b=True))

    for i in range(half_ngpu,ngpu):
        with tf.device('/gpu:%d' % i):
            cost_a2_b1.append(fa_norm[i] + fb_batch1_norm - (2./k)*tf.matmul(features_a[i],fb_batch1,transpose_b=True))
            cost_a2_b2.append(fa_norm[i] + fb_batch2_norm - (2./k)*tf.matmul(features_a[i],fb_batch2,transpose_b=True))
            cost_b2_b1.append(fb_norm[i] + fb_batch1_norm - (2./k)*tf.matmul(features_b[i],fb_batch1,transpose_b=True))

    transport_costs = [tf.concat(cost_a1_a2, 0), tf.concat(cost_b2_b1, 0), tf.concat(cost_a1_b1, 0),
                       tf.concat(cost_a1_b2, 0), tf.concat(cost_a2_b1, 0), tf.concat(cost_a2_b2, 0)]

    if wasserstein_p == 1:
        transport_costs = [safe_sqrt(c) for c in transport_costs]

    # use Sinkhorn algorithm to do soft assignment
    entropies = []
    sinkhorn_distances = []
    for i,C in enumerate(transport_costs):
        with tf.device('/gpu:%d' % (i % ngpu)):
            inv_lam = sinkhorn_inv_lambda[i]
            log_a = -inv_lam * C

            for it in range(nr_sinkhorn_iter):
                log_a -= tf.reduce_logsumexp(log_a, axis=1, keep_dims=True)
                log_a -= tf.reduce_logsumexp(log_a, axis=0, keep_dims=True)

                # adjust lambda
                if it > 5 and it < nr_sinkhorn_iter - 5 and it % 3 == 0:
                    M = tf.nn.softmax(log_a)
                    H = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=M, logits=log_a))
                    delta_H = H - target_entropy
                    delta_H *= (10. / tf.maximum(abs(delta_H), 10.))
                    new_inv_lam = tf.minimum(inv_lam * tf.exp(0.05 * delta_H), 10000.)
                    log_a *= new_inv_lam / inv_lam
                    inv_lam = new_inv_lam

            # final matching, entropy, sinkhorn distance
            with tf.control_dependencies([sinkhorn_inv_lambda[i].assign(inv_lam)]):
                M = tf.stop_gradient(tf.nn.softmax(log_a))
                H = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=M, logits=log_a))
                W = tf.reduce_sum(M * C) / (half_ngpu*n)
                if wasserstein_p == 2:
                    W = safe_sqrt(W)

            entropies.append(H)
            sinkhorn_distances.append(W)

    # get minibatch energy distance
    w_a1_a2, w_b2_b1, w_a1_b1, w_a1_b2, w_a2_b1, w_a2_b2 = sinkhorn_distances
    med_loss = 0.5*(w_a1_b1 + w_a1_b2 + w_a2_b1 + w_a2_b2) - w_a1_a2 - w_b2_b1
    
    return med_loss, entropies

# alternative implementation. very slightly faster and more memory efficient with >= 8 GPUs and large batch size
def minibatch_energy_distance2(features_a, features_b, target_entropy, nr_sinkhorn_iter, wasserstein_p=1):
    ngpu = len(features_a)
    half_ngpu = ngpu // 2
    n, k = features_a[0].get_shape().as_list()
    sinkhorn_inv_lambda = [tf.Variable(500., dtype=tf.float32, trainable=False) for i in range(6)]

    # calculate norms of features
    fa_norm = []
    fb_norm = []
    for i in range(ngpu):
        with tf.device('/gpu:%d' % i):
            fa_norm.append(tf.reduce_mean(tf.square(features_a[i]), axis=1, keep_dims=True))
            fb_norm.append(tf.reduce_mean(tf.square(features_b[i]), axis=1, keep_dims=True))

    # gather features, split into two batches
    with tf.device('/gpu:0'):
        fa_batch2 = tf.concat(features_a[half_ngpu:], axis=0)
        fa_batch2_norm = tf.reshape(tf.concat(fa_norm[half_ngpu:], axis=0), (1, half_ngpu * n))
        fb_batch1 = tf.concat(features_b[:half_ngpu], axis=0)
        fb_batch1_norm = tf.reshape(tf.concat(fb_norm[:half_ngpu], axis=0), (1, half_ngpu * n))
        fb_batch2 = tf.concat(features_b[half_ngpu:], axis=0)
        fb_batch2_norm = tf.reshape(tf.concat(fb_norm[half_ngpu:], axis=0), (1, half_ngpu * n))

    # calculate all transport costs
    cost_a1_a2 = []
    cost_a1_b1 = []
    cost_a1_b2 = []
    cost_a2_b1 = []
    cost_a2_b2 = []
    cost_b2_b1 = []

    for i in range(half_ngpu):
        with tf.device('/gpu:%d' % i):
            cost_a1_a2.append(fa_norm[i] + fa_batch2_norm - (2. / k) * tf.matmul(features_a[i], fa_batch2, transpose_b=True))
            cost_a1_b1.append(fa_norm[i] + fb_batch1_norm - (2. / k) * tf.matmul(features_a[i], fb_batch1, transpose_b=True))
            cost_a1_b2.append(fa_norm[i] + fb_batch2_norm - (2. / k) * tf.matmul(features_a[i], fb_batch2, transpose_b=True))

    for i in range(half_ngpu, ngpu):
        with tf.device('/gpu:%d' % i):
            cost_a2_b1.append(fa_norm[i] + fb_batch1_norm - (2. / k) * tf.matmul(features_a[i], fb_batch1, transpose_b=True))
            cost_a2_b2.append(fa_norm[i] + fb_batch2_norm - (2. / k) * tf.matmul(features_a[i], fb_batch2, transpose_b=True))
            cost_b2_b1.append(fb_norm[i] + fb_batch1_norm - (2. / k) * tf.matmul(features_b[i], fb_batch1, transpose_b=True))

    transport_costs = [cost_a1_a2, cost_a1_b1, cost_a1_b2, cost_b2_b1, cost_a2_b1, cost_a2_b2]

    if wasserstein_p == 1:
        transport_costs = [[safe_sqrt(c) for c in cl] for cl in transport_costs]

    # use Sinkhorn algorithm to do soft assignment
    sinkhorn_distances = []
    entropies = []
    for l, cl in enumerate(transport_costs):
        if l < 3:  # use first half of GPUs
            base_gpu = 0
            gpu_nrs = range(half_ngpu)
        else:  # use second half of GPUs
            base_gpu = half_ngpu
            gpu_nrs = range(half_ngpu, ngpu)

        inv_lam = sinkhorn_inv_lambda[l]

        log_a = []
        for i in gpu_nrs:
            with tf.device('/gpu:%d' % i):
                log_a.append(-inv_lam * cl[i % half_ngpu])

        for it in range(nr_sinkhorn_iter):

            ls_rows = []
            for i in gpu_nrs:
                with tf.device('/gpu:%d' % i):
                    ls_rows.append(tf.reduce_logsumexp(log_a[i % half_ngpu], axis=0, keep_dims=True))
            with tf.device('/gpu:%d' % base_gpu):
                ls_rows = tf.reduce_logsumexp(tf.concat(ls_rows, axis=0), axis=0, keep_dims=True)
            for i in gpu_nrs:
                with tf.device('/gpu:%d' % i):
                    log_a[i % half_ngpu] -= ls_rows

            for i in gpu_nrs:
                with tf.device('/gpu:%d' % i):
                    log_a[i % half_ngpu] -= tf.reduce_logsumexp(log_a[i % half_ngpu], axis=1, keep_dims=True)

            # adjust lagrange multiplier
            if it > 5 and it < nr_sinkhorn_iter - 5 and it % 3 == 0:
                Hs = []
                for i in gpu_nrs:
                    with tf.device('/gpu:%d' % i):
                        Mi = tf.nn.softmax(log_a[i % half_ngpu])
                        Hs.append(tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(labels=Mi, logits=log_a[i % half_ngpu])))
                H = sum(Hs) / half_ngpu
                delta_H = H - target_entropy
                delta_H *= (10. / tf.maximum(abs(delta_H), 10.))
                new_inv_lam = tf.minimum(inv_lam * tf.exp(0.05 * delta_H), 10000.)
                mult = new_inv_lam / inv_lam
                for i in gpu_nrs:
                    with tf.device('/gpu:%d' % i):
                        log_a[i % half_ngpu] *= mult
                inv_lam = new_inv_lam

        # update lambda and calculate matchings
        Ws = []
        Hs = []
        with tf.control_dependencies([sinkhorn_inv_lambda[l].assign(inv_lam)]):
            for i in gpu_nrs:
                with tf.device('/gpu:%d' % i):
                    Mi = tf.stop_gradient(tf.nn.softmax(log_a[i % half_ngpu]))
                    Hs.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Mi, logits=log_a[i % half_ngpu])))
                    Ws.append(tf.reduce_sum(Mi * cl[i % half_ngpu]))
        H = sum(Hs) / half_ngpu
        W = sum(Ws) / (half_ngpu * n)
        if wasserstein_p == 2:
            W = safe_sqrt(W)
        entropies.append(H)
        sinkhorn_distances.append(W)

    # get minibatch energy distance
    w_a1_a2, w_a1_b1, w_a1_b2, w_b2_b1, w_a2_b1, w_a2_b2 = sinkhorn_distances
    med_loss = 0.5 * (w_a1_b1 + w_a1_b2 + w_a2_b1 + w_a2_b2) - w_a1_a2 - w_b2_b1

    return med_loss, entropies

