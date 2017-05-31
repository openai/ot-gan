import numpy as np
import tensorflow as tf
import hungarian
import multiprocessing as mp
num_cores = mp.cpu_count()

def tf_match(costs, reps=100, bs=100):
    assignment = tf.py_func(func = lambda c: match(c, reps, bs), inp=[costs], Tout=tf.int32, stateful=False)
    assignment.set_shape([costs.get_shape().as_list()[0]])
    return assignment

def inner_match(a, c):
    return a[hungarian.match(c)]

def match(costs, reps=100, bs=100):
    n,k = costs.shape
    assert n==k

    inds = np.arange(n)
    assignment = np.arange(n)
    num_batches = n // bs
    num_jobs = np.minimum(num_cores, num_batches)

    with mp.Pool(processes=num_jobs) as pool:

        for rep in range(reps):
            randperm = np.random.permutation(n)
            inds = inds[randperm]
            assignment = assignment[randperm]
            costs = costs[randperm]

            a = np.split(assignment, num_batches)
            c = [ci[:,ai] for ci,ai in zip(np.split(costs, num_batches), a)]

            new_a = pool.starmap(inner_match, zip(a, c))

            assignment = np.concatenate(new_a)

    assignment = assignment[np.argsort(inds)].astype(np.int32)

    return assignment
