import numpy as np
import warnings
import errno
import os


__all__ =['find', 'mkdir', 'load_structured', 'dump_structured', 'structured_to_flat', 'flat_to_structured',
          'recarray_average', 'recarray_difference', 'recarray_norm', 'AllType', 'All']


def find(items, keys):
    try:
        return True, next(x for x in items if x in keys)
    except StopIteration:
        return False, None


def mkdir(folder):
    try:
        os.mkdir(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_structured(config):
    'Convert dictionary with numerical values to numpy structured array'
    dtype_def = []
    for name, value in config.items():
        if isinstance(value, list) or isinstance(value, tuple):
            dtype_def.append((name, np.float64, len(value)))
        elif isinstance(value, float):
            dtype_def.append((name, np.float64))
        elif isinstance(value, int):
            dtype_def.append((name, int))
        else:
            raise RuntimeError('unrecognized type %s', type(value))
    dtype = np.dtype(dtype_def)
    array = np.zeros(1, dtype=dtype)
    for name, value in config.items():
        array[name] = value  # TODO: do we need special handling of input tuple?
    return array


def dump_structured(array):
    'Convert numpy structured array to python dictionary'
    config = {}
    for n in array.dtype.names:
        # TODO: what about lists with a single element?
        if len(array.dtype.fields[n][0].shape) == 1:  # vector type
            config[n] = [float(x) for x in array[n][0]]
        elif len(array.dtype.fields[n][0].shape) == 0:  # scalar type
            config[n] = float(array[n])
        else:
            raise ValueError('unsupported dimension')
    return config


def structured_to_flat(recarray, fields=None):
    r'''Convert numpy structured array to a flat numpy ndarray

    :param recarray: numpy recarray
    :param fields: list of string
        If given, fields will be collected in the same order as they are given in the list.
    :return: numpy ndarray
        If the input recarray contains multiple time steps, the return value
        will be two-dimensional.
    '''

    if recarray.dtype.names is None:
        # conventional ndarray
        if fields is not None and not isinstance(fields, AllType):
            warnings.warn('User requested fields in a particular order %s, but input is already flat. Continuing and hoping for the best.' % str(fields))
        n = recarray.shape[0]
        return recarray.reshape((n, -1))
    else:
        if len(recarray.shape) == 0:
            n = 1
        else:
            n = recarray.shape[0]
        if fields is None or isinstance(fields, AllType):
            fields = recarray.dtype.names
        idx = np.cumsum([0] +
                        [np.prod(recarray.dtype.fields[name][0].shape, dtype=int) for name in fields])
        m = idx[-1]
        x = np.zeros((n, m), dtype=float)
        for name, start, stop in zip(fields, idx[0:-1], idx[1:]):
            x[:, start:stop] = recarray[name]
        return x


def flat_to_structured(array, fields, dims):
    dtype = np.dtype([(name, np.float64, dim) for name, dim in zip(fields, dims)])
    indices = np.concatenate(([0], np.cumsum(dims)))
    # TODO: create simple structured array instead of recarray?
    colgroups = [array[:, start:stop] for name, start, stop in zip(fields, indices[0:-1], indices[1:])]
    return np.core.records.fromarrays(colgroups, dtype=dtype)


def recarray_average(a, b):
    if a.dtype.names != b.dtype.names:
        raise ValueError('a and b must have the same fields')
    c = np.zeros_like(a)
    for name in a.dtype.names:
        c[name] = (a[name] + b[name]) * 0.5
    return c


def recarray_difference(a, b):
    if a.dtype.names != b.dtype.names:
        raise ValueError('a and b must have the same fields')
    c = np.zeros_like(a)
    for name in a.dtype.names:
        c[name] = a[name] - b[name]
    return c


def recarray_norm(a, rmsd=True):
    s = 0.0
    for name in a.dtype.names:
        s += np.sum(a[name]**2)
    if rmsd:
        return (s / len(a.dtype.names))**0.5
    else:
        return s**0.5


class AllType(object):
    'x in All == True for any x'
    def __contains__(self, key):
        return True

    def __repr__(self):
        return 'All'


All = AllType()