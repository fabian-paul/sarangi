import numpy as np

__all__ = ['bottlenecks']


def dijkstra(matrix, start, stop):
    n = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape[1] != n:
        raise ValueError('matrix must be a square matrix')
    pred = np.zeros(n) - 1
    dist = np.empty(n)
    dist[:] = np.inf
    visited = np.zeros(n, dtype=np.int8)
    dist[start] = 0.
    u = start
    for _ in range(n):
        visited[u] = 1
        if u == stop:
            return pred
        else:
            for v in range(n):
                if not visited[v]:
                    q = max(matrix[u, v], dist[u])
                    if q < dist[v]:
                        dist[v] = q
                        pred[v] = u
        u = np.argmin(dist)
    return pred


def short_path(matrix, start, stop):
    pred = dijkstra(matrix, start, stop)
    path = [stop]
    u = stop
    while pred[u] != -1:
        u = pred[u]
        path.append(u)
    if path[-1] != start:
        raise OverflowError('Graph is disconnected, should not happen.')
    return np.array(path[::-1])


def bottlenecks(overlap_matrix):
    r'''Finds the path with the smallest bottleneck starting in image 0 and ending the last image.

    Parameters
    -----------
    overlap_matrix: ndarray((n, n))
        Symmetric matrix of edge weights.

    Returns
    -------
    path: list of indices
        indices of the min-bottleneck path
    gaps: list of float
        edge weigths along edges of the paths
    pairs: list of pairs of indices
        edges of the path, in the same order as gaps
    '''
    # 0 overlap actually means infinite distance!
    # 1 overlap means
    path = short_path(overlap_matrix, 0, overlap_matrix.shape[0] - 1)
    pairs = list(zip(path[0:-1], path[1:]))
    gaps = [overlap_matrix[a, b] for a, b in pairs]
    return path, gaps, pairs


class Network(object):
    def __init__(self, strings):
        self.strings = strings

    def count_matrix(self):
        pass


