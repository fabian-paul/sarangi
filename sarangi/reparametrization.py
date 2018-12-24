import numpy as np

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
    r'Convert old list of nodes to new list of equidistant nodes'
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
    'compute intersection of sphere with radius r around o and the segment from p to q. Return point and arc length'
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
    'For all segments between nodes starting from node i_edge0, find intersection with point at distance d from o.'
    'In first segment, arc length of intersecting point must be larger then s0. Retrun point and arc length'
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


def compute_equidistant_nodes_2(old_nodes, d, direction=-1):
    nodes = np.array(old_nodes)
    i_edge0 = 0
    res = []
    point = nodes[0, :]
    s0 = 0.
    while point is not None:
        res.append(point)
        i_edge0, point, s0 = find_intersecting_segment(nodes, point, d, i_edge0, s0, direction)
        # print(i_edge0)
    return np.concatenate((res, [nodes[-1, :]]))


def reorder_nodes(nodes):
    'Reorder the nodes in the list given in the argument. nodes[0] and nodes[-1] are guaranteed to be unchanged.'
    nodes = np.array(nodes)
    last = len(nodes) - 1
    available = np.arange(1, len(nodes))
    x = nodes[0]
    res = [x]
    end = False
    while not end:
        index_in_available = np.argmin(np.linalg.norm(nodes[available, :] - x[np.newaxis, :], axis=1))
        index = available[index_in_available]
        available = np.setdiff1d(available, [index])
        #print(available)
        x = nodes[index]
        res.append(x)
        if index == last:
            end = True

    if len(res) < len(nodes):
        warnings.warn('String became shorter on reordering, looks like we deleted some omega.', RuntimeWarning)

    return res

