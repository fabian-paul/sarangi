import numpy as np
import warnings
import errno
import os


__all__ =['find', 'mkdir', 'dict_to_structured', 'structured_to_dict', 'structured_to_flat', 'flat_to_structured',
          'recarray_average', 'recarray_difference', 'recarray_vdot', 'recarray_norm', 'recarray_allclose', 'argmin_dict',
          'AllType', 'All', 'root', 'is_sim_id', 'pairing', 'bisect_decreasing', 'IDEAL_GAS_CONSTANT_KCAL_MOL_K',
          'DEFAULT_TEMPERATURE']


IDEAL_GAS_CONSTANT_KCAL_MOL_K = 1.985877534E-3  # in kcal/mol/K
DEFAULT_TEMPERATURE = 303.15

def abspath_with_symlinks(p: str):
    if 'PWD' in os.environ:
       curr_path = os.environ['PWD']
       return os.path.normpath(os.path.join(curr_path, p))
    else:
       return os.path.abspath(p)


def root():
    r'''Return absolute path to the root directory of the project.

        Notes
        -----
        When tracing up the directory tree, this routine will respect the
        environment variable $PWD which may contain a path that contains
        symlinks to directories. Tracing up the directory tree might therefore
        follow the symlinks backwards, if you navigated to the current
        directory by following a symlink.
    '''
    if 'STRING_SIM_ROOT' in os.environ:
        return os.environ['STRING_SIM_ROOT']
    else:  # follow CWD up directory by directory and search for the .sarangirc file
        folder = abspath_with_symlinks('.')
        while not os.path.exists(os.path.join(folder, '.sarangirc')) and folder != '/':
            # print('looking at', folder)
            folder = abspath_with_symlinks(os.path.join(folder, '..'))
        if os.path.exists(os.path.join(folder, '.sarangirc')):
            return folder
        else:
            raise RuntimeError('Could not locate the project root. Environment variable STRING_SIM_ROOT is not set and'
                               ' no .sarangirc file was found.')


def is_sim_id(s: str):
    fields = s.split('_')
    if len(fields) != 4:
        return False
    if not fields[0][0].isalpha():
        return False
    if not fields[1].isnumeric() or not fields[2].isnumeric() or not fields[3].isnumeric():
        return False
    return True


def find(items, keys):
    try:
        return True, next(x for x in items if x in keys)
    except StopIteration:
        return False, None


def mkdir(folder):
    if len(folder) == 0:
        return
    try:
        os.makedirs(folder)
    except (OSError, FileNotFoundError) as e:
        if e.errno != errno.EEXIST:
            raise


def length_along_segment(a, b, x, clamp=True):
    s = 1. - np.vdot(x - b, a - b) / np.vdot(a - b, a - b)  # x = a => 0; x = b => 1
    if clamp:
        s = min(max(s, 0.), 1.)
    return s


def dict_to_structured(config: dict, allow_numpy=False):
    'Convert dictionary with numerical values to numpy structured array'
    dtype_def = []
    for name, value in config.items():
        if isinstance(value, list) or isinstance(value, tuple):
            dtype_def.append((name, np.float64, len(value)))
        elif isinstance(value, float):
            dtype_def.append((name, np.float64))
        elif isinstance(value, int):
            dtype_def.append((name, int))
        elif isinstance(value, np.ndarray) and allow_numpy:
            dtype_def.append((name, value.dtype))
        else:
            raise RuntimeError('unrecognized type %s of %s', (type(value), value))
    dtype = np.dtype(dtype_def)
    array = np.zeros(1, dtype=dtype)
    for name, value in config.items():
        array[name] = value  # TODO: do we need special handling of input tuple?
    return array


def structured_to_dict(array: np.ndarray, keep_fat_scalars=False):
    'Convert numpy structured array to python dictionary'
    if isinstance(array, np.void):
        raise ValueError('structured_to_dict does not yet support conversion of single elements of structured arrays')
    if not isinstance(array, (np.ndarray, np.recarray)):
        raise ValueError('Unsupported type of array parameter.')
    if array.dtype.names is None:
        raise ValueError('Array must be structured and not flat.')
    config = {}
    for n in array.dtype.names:
        if len(array.dtype.fields[n][0].shape) == 1:  # vector type
            if array.dtype.fields[n][0].shape[0] == 1 and not keep_fat_scalars:  # fat scalar
                config[n] = float(array[n][0])
            else:
                config[n] = [float(x) for x in array[n][0]]  # true vector
        elif len(array.dtype.fields[n][0].shape) == 0:  # scalar type
            config[n] = float(array[n])
        else:
            raise ValueError('unsupported dimension')
    return config


def exactly_2d(ary: np.ndarray, add_frame_axis):
    ary = np.asanyarray(ary)
    if ary.ndim == 0:
        return ary.reshape(1, 1)
    elif ary.ndim == 1:
        if add_frame_axis:
            return ary[np.newaxis, :]
        else:
            return ary[:, np.newaxis]
    elif ary.ndim == 2:
        return ary 
    else:
        raise ValueError('Cannot convert array to 2D. ary.ndim is ' + str(ary.ndim))


def structured_to_flat(recarray: np.ndarray, fields=None):
    r'''Convert numpy structured array to a flat numpy ndarray

    :param recarray: numpy recarray
    :param fields: list of string
        If given, fields will be collected in the same order as they are given in the list.
    :return: numpy ndarray
        If the input recarray contains multiple time steps, the return value
        will be two-dimensional.
    '''

    if recarray.dtype.names is None:
        warnings.warn('Calling structured_to_flat on an array that is already flat. This is deprecated.')
        # conventional ndarray
        if fields is not None and not isinstance(fields, AllType):
            warnings.warn('User requested fields in a particular order %s, but input is already flat. '
                          'Continuing without selection/ordering and hoping for the best.' % str(fields))
        #n = recarray.shape[0]
        #return recarray.reshape((n, -1))
        return exactly_2d(recarray, add_frame_axis=False)
    else:
        if len(recarray.shape) == 0:
            n = 1
            warnings.warn('A single element of a structured array was passed to structured_to_flat, this is deprecated.')
        else:
            n = recarray.shape[0]
        if fields is None or isinstance(fields, AllType):
            fields = recarray.dtype.names
        idx = np.cumsum([0] +
                        [np.prod(recarray.dtype.fields[name][0].shape, dtype=int) for name in fields])
        m = idx[-1]
        x = np.zeros((n, m), dtype=float)
        for name, start, stop in zip(fields, idx[0:-1], idx[1:]):
            x[:, start:stop] = exactly_2d(recarray[name], add_frame_axis=False)
        return x


def flat_to_structured(array, fields, dims):
    if array.ndim != 2:
        raise RuntimeError('flat array must be 2D.')
    dtype = np.dtype([(name, np.float64, dim) if dim > 1 else (name, np.float64) for name, dim in zip(fields, dims)])
    indices = np.concatenate(([0], np.cumsum(dims)))
    if array.dtype.names is not None:
        raise ValueError('array parameter must be flat and not structured.')
    structured = np.zeros((len(array),), dtype=dtype)
    if indices[-1] != array.shape[1]:
        raise ValueError('Shape of array is not compatible with dims.')
    if len(fields) != len(dims):
        raise ValueError('Number of fields and dims does not match.')
    for name, start, stop in zip(fields, indices[0:-1], indices[1:]):
        structured[name] = array[:, start:stop]
    return structured
    #print('colgroups', colgroups)
    #print('dtype', dtype)
    # colgroups = [array[:, start:stop] for name, start, stop in zip(fields, indices[0:-1], indices[1:])]
    #return np.core.records.fromarrays(colgroups, dtype=dtype)


def recarray_average(a, b):
    if not all(n in b.dtype.names for n in a.dtype.names) and all(n in b.dtype.names for n in a.dtype.names):
        raise ValueError('a and b must have the same fields')
    c = np.zeros_like(a)
    for name in a.dtype.names:
        c[name] = (a[name] + b[name]) * 0.5
    return c


def recarray_difference(a, b):
    'a - b'
    if not all(n in b.dtype.names for n in a.dtype.names) and all(n in b.dtype.names for n in a.dtype.names):
        raise ValueError('a and b must have the same fields')
    c = np.zeros_like(a)
    for name in a.dtype.names:
        c[name] = a[name] - b[name]
    return c


def recarray_vdot(a, b, allow_broadcasting=False):
    if not all(n in b.dtype.names for n in a.dtype.names) and all(n in b.dtype.names for n in a.dtype.names):
        raise ValueError('a and b must have the same fields')
    if allow_broadcasting:  # TODO: do further testing
        s = None
        for name in a.dtype.names:
            if s is None:
                s = np.einsum('ti,ti->t', a[name], b[name])
            else:
                s += np.einsum('ti,ti->t', a[name], b[name])
            return s
    else:
        s = 0.0
        for name in a.dtype.names:
            s += np.vdot(a[name], b[name])
        return s


def recarray_norm(a, rmsd=False):
    s = 0.0
    for name in a.dtype.names:
        s += np.sum(a[name]**2)
    if rmsd:
        return (s / len(a.dtype.names))**0.5
    else:
        return s**0.5


def recarray_dims(x):
    res = []
    for n in x.dtype.names:
        shape = x.dtype.fields[n][0].shape
        if len(shape) == 0:
            res.append(1)
        else:
            res.append(shape[0])
    return res


def recarray_allclose(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return a==b
    if sorted(a.dtype.names) != sorted(b.dtype.names):
        return False
    return all([np.allclose(a[field], b[field]) for field in a.dtype.names])


def argmin_dict(d):
    'key with minimum value'
    minimum = float('inf')
    argminimum = None
    for k, v in d.items():
        if v < minimum:
            minimum = v
            argminimum = k
    return argminimum


class AllType(object):
    'x in All == True for any x'
    def __contains__(self, key):
        return True

    def __repr__(self):
        return 'All'


All = AllType()


def pairing(i, j, ordered=True):
    'Cantor\'s pairing function.'
    return (i + j) * (i + j + 1) // 2 + i
    #if not tri:
    #    return (i + j) * (i + j + 1) // 2 + i
    #else:  # TODO: think about more compact pairing
    #    return i*(i + 1)//2 + j  # TODO: correct for the ordering of states


def nodes_to_trajs(string, fname='nodes.pdb', fields=All):
    pdb_fmt = '{ATOM:<6}{serial_number:>5} {atom_name:<4}{alt_loc_indicator:<1}{res_name:<3} ' \
              '{chain_id:<1}{res_seq_number:>4}{insert_code:<1}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}'

    frames = []
    for i_node, image in enumerate(string.images_ordered):
        frame = 'MODEL      {model:>3}\n'.format(model=i_node)
        node = image.node
        for i, field in enumerate(node.dtype.names):
            if field in fields:
                frame += (pdb_fmt.format(
                        ATOM='ATOM',
                        serial_number=i,
                        atom_name='C',
                        alt_loc_indicator=' ',
                        res_name=field[0:3],
                        chain_id='A',
                        res_seq_number=i,
                        insert_code=' ',
                        x=node[field][0, 0],
                        y=node[field][0, 1],
                        z=node[field][0, 2],
                        occupancy=1.0,
                        temp_factor=0.0) + '\n')
        frame += 'ENDMDL\n'
        frames.append(frame)

    with open(fname, 'w') as f:
        for frame in frames:
            f.write(frame)

    return frames


def shortest_path(cost_matrix, start, stop):
    'Compute the shortest path (minimum summed cost path) from start to stop.'
    import scipy.sparse.csgraph
    _, pred = scipy.sparse.csgraph.dijkstra(cost_matrix, directed=False, indices=start, return_predecessors=True)
    path = [stop]
    u = stop
    while pred[u] != start:
        u = pred[u]
        if u < 0:
            raise ValueError('Graph is disconneted, could not find shortest path from %s to %d.' % (start, stop))
        path.append(u)
    return [start] + path[::-1]


def widest_path(matrix, start=0, stop=-1):
    'Compute the widest path (max-capacity path or min-bottleneck path) form start to stop.'
    import scipy.sparse.csgraph
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Matrix must be square.')
    mst = -scipy.sparse.csgraph.minimum_spanning_tree(-matrix)
    _, pred = scipy.sparse.csgraph.depth_first_order(mst, i_start=start, directed=False, return_predecessors=True)
    # convert list of predecessors to path
    stop = np.arange(matrix.shape[0])[stop]
    path = [stop]
    u = stop
    while pred[u] != start:
        u = pred[u]
        if u < 0:
            raise ValueError('Graph is disconneted, could not find shortest path from %s to %d.' % (start, stop))
        path.append(u)
    path = [start] + path[::-1]
    return path


def bisect_decreasing(func, a, b, level=0., max_iter=50, accuracy=1.E-5, precision=1.E-5, return_y=False):
    r'''Find a root x* of func(x*)==level using the bisection method, assuming that func is decreasing. Algorithm will extend [a, b] if [f(a), f(b)] does not bracket level.

    Parameters
    ----------
    a: float
        Guess for one bound of a root-bracketing interval.

    b: float
        Guess for one bound of a root-bracketing interval.

    level: float
        The right-hand-side of the equation  func(x*) = level

    accuracy: float, default = 1.E-5
        threshold on the approximation error in the independent variable
        Function succeeds if either of accuracy or precision thresholds have been met.

    precision: float, default = 1.E-5
        threshold on the spread of approximation error (error in the independent variable)
        Function succeeds if either of accuracy or precision thresholds have been met.
    '''
    a_, b_ = a, b
    a = min(a_, b_)
    b = max(a_, b_)
    # if (a, b) does not bracket a root (assuming monotonic decreasing behaviour of func), extend the interval.
    for _ in range(max_iter):
        fa = func(a) - level
        if fa > 0:
            break
        else:
            a *= 0.5
    for _ in range(max_iter):
        fb = func(b) - level
        if fb < 0:
            break
        else:
            b *= 2
    if not fa > 0 > fb: # (fa < 0 < fb): #or
        raise RuntimeError('Initial interval for bisection does not bracket the level that you are searching for.')
    for _ in range(max_iter):
        c = 0.5*(a + b)
        y = func(c) - level
        #print('y', y)
        if np.abs(y) < accuracy or 0.5*(b - a) <= precision:
            if return_y:
                return c, y + level
            else:
                return c
        if np.sign(y) == np.sign(func(a) - level):
            a = c
        else:
            b = c
    raise RuntimeError('Bisection could not find x such that {func}(x)={level}'.format(func=func, level=level))
