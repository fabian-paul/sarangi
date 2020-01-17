import numpy as np
import warnings

# TODO: rename module to geometry?

def point_at_arc_length_in_segment(p, q, s0, d):
    r'''Get point at arc length s + d on the segment [p, q].

    :param p: point 1 of the segment
    :param q: point 2 of the segment
    :param s0: initial arc length from which to measure all length
    :param d: point returned will be at arc length s0 + d
    :return:
        s: arc length of point x (in the current segment)
        x: the next point x (d away from s)
        z: s - l, the negative distance from x to the segment end
    '''
    s = s0 + d
    assert s >= 0
    l = np.linalg.norm(p - q)
    if s > l:
        return s0, None, s0 - l
    else:
        x = (l - s)/l * p + s/l * q
        return s, x, s - l


def compute_equidistant_nodes(old_nodes, d):
    '''Convert old list of nodes to new list of roughly equidistant nodes by following the polyline through the old nodes.

        Notes
        -----
        New nodes are equidistant in the subspace formed by the polyline that goes through the old nodes.
        New nodes are not equidistant in Cartesian space.
        Use compute_equidistant_nodes_2 to generate nodes equidistant in Cartesian space.
    '''

    pts = []
    s = 0
    for i in range(len(old_nodes) - 1):
        end = False
        while not end:
            s, x, z = point_at_arc_length_in_segment(p=old_nodes[i], q=old_nodes[i + 1], s0=s, d=d)
            #print('s, x, z =', s, x, z)
            if x is None:
                end = True
                s = z
            else:
                pts.append(x)
    return pts


def next_at_distance(p, q, o, r, s0=0.):
    'Compute intersection of sphere with radius r around o and the segment from p to q. Return point and arc length.'
    n = q - p
    x = p - o
    a = np.dot(n, n)
    b = np.dot(x, n)
    c = np.dot(x, x) - r * r
    det = b * b - a * c
    if det < 0:
        return None, 0.
    t1 = (-b + det ** 0.5) / a
    t2 = (-b - det ** 0.5) / a
    l = a ** 0.5
    if s0 <= l * t1 and t1 <= 1.:
        return p + t1 * n, l * t1
    elif s0 <= l * t2 and t2 <= 1.:
        return p + t2 * n, l * t2
    else:
        return None, 0.


def find_intersecting_segment(nodes, o, d, i_edge0, s0, direction=-1):
    r'''For all segments between nodes starting from nodes[i_edge0], find intersection with point at distance d from o.

        Parameters
        ----------
        direction: int, default = -1
           One of 1 or -1. Where to start searching for segments.
           1 means to start searching towards the beginning of the list
           of nodes (i.e. close to the last node that was added to the
           reparametrized string being built).
           -1 means to start searching at the last node. -1 makes faster
           progress towards the terminal node, bypassing some detours.
           This is in the spirit of a very simplified variant of dynamic
           programming.

        Notes
        -----
        In first segment, arc length of intersecting point must be larger then s0. Return point and arc length
    '''
    st = direction
    # print('i_edge0', i_edge0)
    nodes = np.array(nodes)[i_edge0:, :]
    for i_edge, p, q in zip(np.arange(len(nodes) - 1)[::st], nodes[0:-1, :][::st, :], nodes[1:, :][::st, :]):
        # s0 only refers to i_edge0; once we have passed that, set to 0
        intersection, s = next_at_distance(p, q, o, d, s0 if i_edge == 0 else 0.)
        # print('test', i_edge, s0 if i_edge==0 else 0., p, q)
        if intersection is not None:
            # print('return', i_edge + i_edge0)
            return i_edge + i_edge0, intersection, s

    return -1, None, 0.


def compute_equidistant_nodes_2(old_nodes, d, direction=-1, d_skip=None, do_warn=True):
    r'Convert old list of nodes to new list of exactly equidistant nodes (equidistant in Cartesian space).'
    nodes = np.array(old_nodes)
    if nodes.ndim != 2:
        raise ValueError('old_nodes must be 2-D')
    if nodes.dtype not in [np.float64, np.float32]:
        raise ValueError('old_node must be array of floating point numbers')
    i_edge0 = 0
    res = []
    point = nodes[0, :]
    s0 = 0.
    while point is not None:
        res.append(point)
        i_edge0, point, s0 = find_intersecting_segment(nodes, point, d, i_edge0, s0, direction)
        # print(i_edge0)
    # done, finally check if the last newly interpolated node is too close to the end node and maybe remove it
    if d_skip is not None and np.linalg.norm(res[-1] - nodes[-1, :]) < d_skip:
        res.pop()
        percentage = 100 * np.linalg.norm(res[-1] - nodes[-1, :]) / d
        if do_warn:
            warnings.warn('During reparametrization, the last interpolated image is closer than %f to the fixed end of the '
                          'string. Did not add that interpolating image. This leads to a image distance at the end of the '
                          'string that is %d%% of the normal distance.' % (d_skip, percentage))
    return np.concatenate((res, [nodes[-1, :]]))


def reorder_nodes(nodes, return_indices=False):
    'Reorder the nodes in the list given in the argument. nodes[0] and nodes[-1] are guaranteed to be unchanged.'
    nodes = np.array(nodes)
    if nodes.ndim != 2:
        raise ValueError('nodes must be 2-D and not %d' % nodes.ndim)
    if nodes.dtype not in [np.float64, np.float32]:
        raise ValueError('nodes must be array of floating point numbers')
    last = len(nodes) - 1
    available = np.arange(1, len(nodes))
    x = nodes[0]
    res = [x]
    order = [0]
    end = False
    while not end:
        index_in_available = np.argmin(np.linalg.norm(nodes[available, :] - x[np.newaxis, :], axis=1))
        index = available[index_in_available]
        available = np.setdiff1d(available, [index])
        #print(available)
        x = nodes[index]
        res.append(x)
        order.append(index)
        if index == last:
            end = True

    if len(res) < len(nodes):
        warnings.warn(
            'String became shorter on reordering, looks like we deleted one (or more) meander(s) '
            'consisting of nodes ' + ', '.join([str(i) for i in available]) + '.', RuntimeWarning)

    if return_indices:
        return res, order
    else:
        return res


def curvatures(nodes):
    n = nodes.shape[0] - 2
    chi = np.zeros(n)
    for i in range(n):
        y_m = nodes[i, :]
        y_i = nodes[i + 1, :]
        y_p = nodes[i + 2, :]
        t_p = y_p - y_i
        t_i = y_i - y_m
        l_p = np.linalg.norm(t_p)
        l_i = np.linalg.norm(t_i)
        chi_i = np.vdot(t_p, t_i) / (l_p * l_i)
        chi[i] = np.abs(chi_i)
    return chi


def smooth(nodes, filter_width):
    'Smooth nodes with a moving average filter.'
    smoothed = np.zeros_like(nodes)
    for i in range(nodes.shape[1]):
        padded = np.pad(nodes[:, i], (filter_width // 2, filter_width - 1 - filter_width // 2), mode='edge')
        smoothed[:, i] = np.convolve(padded, np.ones((filter_width,)) / filter_width, mode='valid')
    # preserve initial and final nodes
    smoothed[0, :] = nodes[0, :]
    smoothed[-1, :] = nodes[-1, :]
    return smoothed
