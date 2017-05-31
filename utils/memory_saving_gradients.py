import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from toposort import toposort
import contextlib
import time
from tensorflow.python.ops.control_flow_ops import MaybeCreateControlFlowState
import sys
sys.setrecursionlimit(10000)

# graph editor slowness work-around: https://github.com/tensorflow/tensorflow/issues/9925#issuecomment-302476302
from tensorflow.contrib.graph_editor import transform
def dummy_collections_handler(info, elem, elem_): pass
transform.assign_renamed_collections_handler = dummy_collections_handler

# tf.gradients slowness work-around: https://github.com/tensorflow/tensorflow/issues/9901
def _MyPendingCount(graph, to_ops, from_ops, colocate_gradients_with_ops):

    # get between ops, faster for large graphs than original implementation
    between_op_list = ge.get_backward_walk_ops(to_ops, stop_at_ts=[op.outputs[0] for op in from_ops], inclusive=False)
    between_op_list += to_ops + from_ops
    between_op_list = list(set(between_op_list))
    between_ops = [False] * (graph._last_id + 1)
    for op in between_op_list:
        between_ops[op._id] = True

    # 'loop_state' is None if there are no while loops.
    loop_state = MaybeCreateControlFlowState(between_op_list, between_ops, colocate_gradients_with_ops)

    # Initialize pending count for between ops.
    pending_count = [0] * (graph._last_id + 1)
    for op in between_op_list:
        for x in op.inputs:
            if between_ops[x.op._id]:
                pending_count[x.op._id] += 1

    return pending_count, loop_state

from tensorflow.python.ops import gradients_impl
gradients_impl._PendingCount = _MyPendingCount


def gradients(ys, xs, grad_ys=None, remember='collection', **kwargs):
    '''
    Authors: Tim Salimans & Yaroslav Bulatov, OpenAI

    memory efficient gradient implementation inspired by "Training Deep Nets with Sublinear Memory Cost"
    by Chen et al. 2016 (https://arxiv.org/abs/1604.06174)

    ys,xs,grad_ys,kwargs are the arguments to standard tensorflow tf.gradients

    'remember' can either be
        - a list consisting of tensors from the forward pass of the neural net
          that we should re-use when calculating the gradients in the backward pass
          all other tensors that do not appear in this list will be re-computed
        - a string specifying how this list should be determined. currently we support
            - 'speed':  remember all outputs of convolutions and matmuls. these ops are usually the most expensive,
                        so remembering them maximizes the running speed
                        (this is a good option if nonlinearities, concats, batchnorms, etc are taking up a lot of memory)
            - 'memory': try to minimize the memory usage
                        (currently using a very simple strategy that identifies a number of bottleneck tensors in the graph to remember)
            - 'collection': look for a tensorflow collection named 'remember', which holds the tensors to remember
    '''
    if not isinstance(ys, list):
        ys = [ys]
    if not isinstance(xs, list):
        xs = [xs]

    # get "forward" graph
    bwd_ops = ge.get_backward_walk_ops([y.op for y in ys], inclusive=True)
    fwd_ops = ge.get_forward_walk_ops([x.op for x in xs], inclusive=True, within_ops=bwd_ops)

    # remove all placeholders, variables and assigns
    fwd_ops = [op for op in fwd_ops if op._inputs]
    fwd_ops = [op for op in fwd_ops if not '/read' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/Assign' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/assign' in op.name]

    # construct list of tensors to remember from forward pass, if not given as input
    if type(remember) is not list:
        if remember == 'collection':
            remember = tf.get_collection('remember')
            remember = list(set(remember).intersection(set(ge.filter_ts(fwd_ops, True))))

        elif remember == 'speed':
            # remember all expensive ops to maximize running speed
            remember = ge.filter_ts_from_regex(fwd_ops, 'Conv|MatMul')

        elif remember == 'memory':

            # get all tensors in the fwd graph
            ts = ge.filter_ts(fwd_ops, True)

            # filter out all tensors that are inputs of the backward graph
            with capture_ops() as bwd_ops:
                gs = tf.gradients(ys, xs, grad_ys=grad_ys, **kwargs)
            bwd_inputs = [t for op in bwd_ops for t in op.inputs]
            ts_filtered = list(set(bwd_inputs).intersection(ts))

            for rep in range(2):  # try two slightly different ways of getting bottlenecks tensors to remember

                if rep == 0:
                    ts_rep = ts_filtered  # use only filtered candidates
                else:
                    ts_rep = ts  # if not enough bottlenecks found on first try, use all tensors as candidates

                # get all bottlenecks in the graph
                bottleneck_ts = []
                for t in ts_rep:
                    b = set(ge.get_backward_walk_ops(t.op, inclusive=True, within_ops=fwd_ops))
                    f = set(ge.get_forward_walk_ops(t.op, inclusive=False, within_ops=fwd_ops))

                    # check that there are not shortcuts
                    b_inp = [inp for op in b for inp in op.inputs]
                    f_inp = [inp for op in f for inp in op.inputs]
                    if not set(b_inp).intersection(f_inp):  # we have a bottleneck!
                        bottleneck_ts.append(t)

                # success? or try again without filtering?
                if len(bottleneck_ts) >= np.sqrt(len(ts_filtered)):  # yes, enough bottlenecks found!
                    break

            if not bottleneck_ts:
                raise (
                'unable to find bottleneck tensors! please provide remember nodes manually, or use remember="speed".')

            # sort the bottlenecks
            bottlenecks_sorted_lists = tf_toposort(bottleneck_ts)
            sorted_bottlenecks = [t for ts in bottlenecks_sorted_lists for t in ts]

            # save an approximately optimal number ~ sqrt(N)
            N = len(ts_filtered)
            k = np.minimum(int(np.floor(np.sqrt(N))), len(sorted_bottlenecks) // 2)
            remember = sorted_bottlenecks[k:N:k]

        else:
            raise ('unsupported input for "remember"')

    # remove initial and terminal nodes from remember list if present
    remember = list(set(remember) - set(ys) - set(xs))

    # check that we have some nodes to remember
    if not remember:
        raise ('no remember nodes found or given as input!')

    # disconnect dependencies between remembered tensors
    remember_disconnected = {x: tf.stop_gradient(x) for x in remember}

    # partial derivatives to the remembered tensors and xs
    ops_to_copy = ge.get_backward_walk_ops(seed_ops=[y.op for y in ys], stop_at_ts=remember, within_ops=fwd_ops)
    copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
    copied_ops = info._transformed_ops.values()
    ge.reroute_ts(remember_disconnected.values(), remember_disconnected.keys(), can_modify=copied_ops)
    for op in copied_ops:
        ge.add_control_inputs(op, [y.op for y in ys])

    # get gradients with respect to current boundary + original x's
    copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]
    boundary = list(remember_disconnected.values())
    dv = tf.gradients(copied_ys, boundary + xs, grad_ys=grad_ys, **kwargs)

    # extract partial derivatives to the remembered nodes
    d_remember = {r: dr for r, dr in zip(remember_disconnected.keys(), dv[:len(remember_disconnected)])}
    # extract partial derivatives to xs (usually the params of the neural net)
    d_xs = dv[len(remember_disconnected):]

    # incorporate derivatives flowing through the remembered nodes
    remember_sorted_lists = tf_toposort(remember, within_ops=fwd_ops)
    for ts in remember_sorted_lists[::-1]:
        remember_other = [r for r in remember if r not in ts]
        remember_disconnected_other = [remember_disconnected[r] for r in remember_other]

        # copy part of the graph below current remember node, stopping at other remember nodes
        ops_to_copy = ge.get_backward_walk_ops(seed_ops=[r.op for r in ts], stop_at_ts=remember_other, within_ops=fwd_ops)
        if not ops_to_copy:  # we're done!
            break
        copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
        copied_ops = info._transformed_ops.values()
        ge.reroute_ts(remember_disconnected_other, remember_other, can_modify=copied_ops)
        for op in copied_ops:
            ge.add_control_inputs(op, [d_remember[r].op for r in ts])

        # gradient flowing through the remembered node
        boundary = [info._transformed_ops[r.op]._outputs[0] for r in ts]
        substitute_backprops = [d_remember[r] for r in ts]
        dv = tf.gradients(boundary, remember_disconnected_other + xs, grad_ys=substitute_backprops, **kwargs)

        # partial derivatives to the remembered nodes
        for r, dr in zip(remember_other, dv[:len(remember_other)]):
            if dr is not None:
                if d_remember[r] is None:
                    d_remember[r] = dr
                else:
                    d_remember[r] += dr

        # partial derivatives to xs (usually the params of the neural net)
        d_xs_new = dv[len(remember_other):]
        for j in range(len(xs)):
            if d_xs_new[j] is not None:
                if d_xs[j] is None:
                    d_xs[j] = d_xs_new[j]
                else:
                    d_xs[j] += d_xs_new[j]

    return d_xs


def tf_toposort(ts, within_ops=None):
    all_ops = ge.get_forward_walk_ops([x.op for x in ts], within_ops=within_ops)
    deps = {}
    for op in all_ops:
        for o in op.outputs:
            deps[o] = set(op.inputs)
    sorted_ts = toposort(deps)

    # only keep the tensors from our original list
    ts_sorted_lists = []
    for l in sorted_ts:
        keep = list(set(l).intersection(ts))
        if keep:
            ts_sorted_lists.append(keep)

    return ts_sorted_lists


@contextlib.contextmanager
def capture_ops():
    """Decorator to capture ops created in the block.
  with capture_ops() as ops:
    # create some ops
  print(ops) # => prints ops created.
  """

    micros = int(time.time() * 10 ** 6)
    scope_name = str(micros)
    op_list = []
    with tf.name_scope(scope_name):
        yield op_list

    g = tf.get_default_graph()
    op_list.extend(ge.select_ops(scope_name + "/.*", graph=g))
