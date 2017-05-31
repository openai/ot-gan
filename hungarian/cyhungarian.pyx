# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


cdef extern from "hungarian.h":
    ssize_t** kuhn_match(long** table, size_t n, size_t m)


@cython.boundscheck(False)
def match(np.ndarray[np.float32_t, ndim=2] costs):
    cdef float ratio = 2147483647.0 / max(abs(costs.min()), abs(costs.max()))  # for scaling the costs to integers

    # Convert costs (by scaling them up) to integers
    cdef size_t n = costs.shape[0]
    cdef size_t m = costs.shape[1]
    cdef long** table = <long**> malloc(n * sizeof(long*))
    cdef size_t i, j
    for i in range(n):
        table[i] = <long*> malloc(m * sizeof(long))
        for j in range(m):
            table[i][j] = <long> (costs[i, j] * ratio)

    # Do it
    cdef ssize_t** assignment = kuhn_match(table, n, m)

    # Save outputs
    cdef np.ndarray[np.int_t, ndim=1] out = np.empty(n, dtype=int)
    for i in range(n):
        # assignment[i][0] seems to be just i, so ignore it
        out[i] = assignment[i][1]

    # Free stuff
    for i in range(n):
        free(table[i])
        free(assignment[i])
    free(table)
    free(assignment)

    return out

