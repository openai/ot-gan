import tensorflow as tf

def minibatch_energy_distance(features_a, features_b, target_entropy, nr_sinkhorn_iter, wasserstein_p=1):
    ngpu = len(features_a)
    half_ngpu = ngpu // 2
    n,k = features_a[0].get_shape().as_list()
    sinkhorn_inv_lambda = [tf.Variable(500., dtype=tf.float32, trainable=False) for i in range(6)]
    sil = [s for s in sinkhorn_inv_lambda]

    # calculate norms of features
    fa_norm = []
    fb_norm = []
    for i in range(ngpu):
        with tf.device('/gpu:%d' % i):
            fa_norm.append(tf.reshape(tf.reduce_mean(tf.square(features_a[i]), axis=1), (n,1)))
            fb_norm.append(tf.reshape(tf.reduce_mean(tf.square(features_b[i]), axis=1), (n,1)))

    # gather features, split into two batches
    with tf.device('/gpu:0'):
        fa_batch2 = tf.concat(features_a[half_ngpu:], axis=0)
        fa_batch2_norm = tf.reshape(tf.concat(fa_norm[half_ngpu:], axis=0), (1,half_ngpu*n))
        fb_batch1 = tf.concat(features_b[:half_ngpu], axis=0)
        fb_batch1_norm = tf.reshape(tf.concat(fb_norm[:half_ngpu], axis=0), (1,half_ngpu*n))
        fb_batch2 = tf.concat(features_b[half_ngpu:], axis=0)
        fb_batch2_norm = tf.reshape(tf.concat(fb_norm[half_ngpu:], axis=0), (1,half_ngpu*n))

    # calculate all distances
    dist_a1_a2 = []
    dist_a1_b1 = []
    dist_a1_b2 = []
    dist_a2_b1 = []
    dist_a2_b2 = []
    dist_b2_b1 = []

    for i in range(half_ngpu):
        with tf.device('/gpu:%d' % i):
            dist_a1_a2.append(fa_norm[i] + fa_batch2_norm - (2./k)*tf.matmul(features_a[i],fa_batch2,transpose_b=True))
            dist_a1_b1.append(fa_norm[i] + fb_batch1_norm - (2./k)*tf.matmul(features_a[i],fb_batch1,transpose_b=True))
            dist_a1_b2.append(fa_norm[i] + fb_batch2_norm - (2./k)*tf.matmul(features_a[i],fb_batch2,transpose_b=True))

    for i in range(half_ngpu,ngpu):
        with tf.device('/gpu:%d' % i):
            dist_a2_b1.append(fa_norm[i] + fb_batch1_norm - (2./k)*tf.matmul(features_a[i],fb_batch1,transpose_b=True))
            dist_a2_b2.append(fa_norm[i] + fb_batch2_norm - (2./k)*tf.matmul(features_a[i],fb_batch2,transpose_b=True))
            dist_b2_b1.append(fb_norm[i] + fb_batch1_norm - (2./k)*tf.matmul(features_b[i],fb_batch1,transpose_b=True))

    transport_costs = [dist_a1_a2, dist_a1_b1, dist_a1_b2, dist_b2_b1, dist_a2_b1, dist_a2_b2]

    if wasserstein_p == 1:
        transport_costs = [[tf.sqrt(1e-8 + c) for c in cl] for cl in transport_costs]

    # use Sinkhorn algorithm to do soft assignment
    distances = []
    entropies = []
    for l,cl in enumerate(transport_costs):
        if l < 3:  # use first half of GPUs
            base_gpu = 0
            gpu_nrs = range(half_ngpu)
        else:  # use second half of GPUs
            base_gpu = half_ngpu
            gpu_nrs = range(half_ngpu, ngpu)

        ls_cols = []
        for i in gpu_nrs:
            with tf.device('/gpu:%d' % i):
                ls_cols.append(tf.reduce_logsumexp(-sil[l] * cl[i % half_ngpu], axis=1, keep_dims=True)/sil[l])

        for it in range(nr_sinkhorn_iter):

            ls_rows = []
            for i in gpu_nrs:
                with tf.device('/gpu:%d' % i):
                    ls_rows.append(tf.reduce_logsumexp(-sil[l] * (cl[i % half_ngpu] + ls_cols[i % half_ngpu]), axis=0, keep_dims=True)/sil[l])
            with tf.device('/gpu:%d' % base_gpu):
                ls_rows = tf.reduce_logsumexp(tf.concat(ls_rows, axis=0), axis=0, keep_dims=True)

            ls_cols = []
            for i in gpu_nrs:
                with tf.device('/gpu:%d' % i):
                    ls_cols.append(tf.reduce_logsumexp(-sil[l] * (cl[i % half_ngpu] + ls_rows), axis=1, keep_dims=True)/sil[l])

            # calculate entropy and adjust lagrange multiplier
            Hs = []
            for i in gpu_nrs:
                with tf.device('/gpu:%d' % i):
                    li = -sil[l] * (cl[i % half_ngpu] + ls_cols[i % half_ngpu] + ls_rows)
                    Mi = tf.nn.softmax(li)
                    Hs.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Mi, logits=li)))
            H = sum(Hs)/half_ngpu
            delta_H = H-target_entropy
            delta_H *= (10./tf.maximum(abs(delta_H),10.))
            sil[l] = tf.minimum(sil[l] * tf.exp(0.03 * delta_H), 10000.)

        # update lambda and calculate the distance
        lambda_update = sinkhorn_inv_lambda[l].assign(sil[l])
        with tf.control_dependencies([lambda_update]):
            Ws = []
            for i in gpu_nrs:
                with tf.device('/gpu:%d' % i):
                    li = -sil[l] * (cl[i % half_ngpu] + ls_cols[i % half_ngpu] + ls_rows)
                    Mi = tf.stop_gradient(tf.nn.softmax(li))
                    Ws.append(tf.reduce_mean(Mi * cl[i % half_ngpu]))
            W = sum(Ws) / half_ngpu

        if wasserstein_p == 2:
            W = tf.sqrt(1e-8 + W)

        distances.append(W)
        entropies.append(H)

    # get loss
    w_a1_a2, w_a1_b1, w_a1_b2, w_b2_b1, w_a2_b1, w_a2_b2 = distances
    loss = 0.5*(w_a1_b1 + w_a1_b2 + w_a2_b1 + w_a2_b2) - w_a1_a2 - w_b2_b1
    
    return loss, entropies

