cdef extern from "_short_path.h":
    void dijkstra_impl(size_t start, size_t stop, size_t T, size_t n, const float * x, double * dist, char * visited, int * pred, float param);

import numpy as np
cimport numpy as np

def path(x, start, stop, param):
    r'''Compute the shortest path through data points using Dijkstra's algorithm.

    Paramaters
    ----------
    x : np.array((T, n), np.float32)
        Array of T data points, each point having dimension x.
    start : int
        index of desired starting point of the path
    stop : int
        index of desired end point of the path
    param : float
        Parameter for the distance computation, d(x,y) = exp(...)

    Resturns
    --------
    list of intergers : indices that encode the order of the path 
    '''
    T = x.shape[0]
    cdef int[:] pred = np.zeros((T,), np.intc)
    cdef char[:] visited = np.zeros((T,), np.int8)
    cdef double[:] dist = np.zeros((T,), np.float64)
    cdef float[:, :] y = np.require(x, dtype=np.float32, requirements=('C', 'A'))
    if start < 0 or start >= T:
        raise ValueError('start must be in the range of 0 to len(x)-1.')
    if stop < 0 or stop >= T:
        raise ValueError('stop must be in the range of 0 to len(x)-1.')
    dijkstra_impl(start, stop, T, x.shape[1], &y[0, 0], &dist[0], &visited[0], &pred[0], param)
    path = [stop]
    u = stop
    while pred[u] != -1:
        u = pred[u]
        path.append(u)
    if path[-1] != start:
        raise OverflowError('Cost computation yielded disconnected network. Try decreasing the exponential scaling parameter.')
    return path[::-1]