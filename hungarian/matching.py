import numpy as np
import tensorflow as tf
import hungarian

def tf_match(costs, reps=100, bs=100):
    assignment = tf.py_func(func = lambda c: match(c, reps, bs), inp=[costs], Tout=tf.int32, stateful=False)
    assignment.set_shape([costs.get_shape().as_list()[0]])
    return assignment

def inner_match(a, c):
    return a[hungarian.match(c[:,a])]

def match(costs, reps=100, bs=100):
    n,k = costs.shape
    assert n==k

    inds = np.arange(n)
    assignment = np.arange(n)
    num_batches = n // bs

    for rep in range(reps):
        randperm = np.random.permutation(n)
        inds = inds[randperm]
        assignment = assignment[randperm]
        costs = costs[randperm]

        a = np.split(assignment, num_batches)
        c = np.split(costs, num_batches)

        new_a = map(inner_match, zip(a, c))

        assignment = np.concatenate(new_a)

    assignment = assignment[np.argsort(inds)].astype(np.int32)

    return assignment
