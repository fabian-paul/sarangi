import numpy as np
import os
from .util import All, AllType, structured_to_flat, length_along_segment

__all__ = ['Colvars']


class Colvars(object):
    def __init__(self, folder, base, fields=All):
        self._mean = None
        self._var = None
        self._cov = None
        fname = folder + '/' + base
        if os.path.exists(fname + '.colvars.traj'):
            self._colvars = Colvars.load_colvar(fname + '.colvars.traj', selection=fields)
        elif os.path.exists(fname + '.npy'):
            self._colvars = np.load(fname + '.npy')
            if not isinstance(fields, AllType):
                #print('Hello fields!', fields, type(fields))
                self._colvars = self._colvars[fields]
        # TODO: support pdb format for later use for gap detection!
        else:
            raise FileNotFoundError('No progress coordinates / colvar file %s found.' % fname)

    @staticmethod
    def parse_line(tokens):
        'Convert line from colvsars.traj.file to a list of values and a list of vector dimensions'
        in_vector = False
        dims = []
        dim = 1
        values = list()
        for t in tokens:
            if t == '(':
                assert not in_vector
                in_vector = True
                dim = 1
            elif t == ')':
                assert in_vector
                in_vector = False
                dims.append(dim)
                dim = 1
            elif t == ',':
                assert in_vector
                dim += 1
            else:
                values.append(float(t))
                if not in_vector:
                    dims.append(dim)
        return values, dims

    @staticmethod
    def load_colvar(fname, selection=All):
        'Load a colvars.traj file and convert to numpy recarray (with field names taken form the file header)'
        rows = []
        with open(fname) as f:
            for line in f:
                tokens = line.split()
                if tokens[0] == '#':
                    var_names = tokens[1:]
                else:
                    values, dims = Colvars.parse_line(tokens)
                    rows.append(values)
        assert len(var_names) == len(dims)
        assert len(rows[0]) == sum(dims)
        data = np.array(rows[1:])  # ignore the zero'th row, since this is just the initial condition (in NAMD)

        if selection is None:
            selection = All
        elif isinstance(selection, str):
            selection = [selection]

        # convert to structured array
        dtype_def = []
        for name, dim in zip(var_names, dims):
            # print(name, 'is in selection?', name in selection)
            if name in selection:
                if dim == 1:
                    dtype_def.append((name, np.float64, 1))
                else:
                    dtype_def.append((name, np.float64, dim))
        #print('dtype_def is', dtype_def)
        dtype = np.dtype(dtype_def)

        indices = np.concatenate(([0], np.cumsum(dims)))
        colgroups = [np.squeeze(data[:, start:stop]) for name, start, stop in zip(var_names, indices[0:-1], indices[1:])
                        if name in selection]
        # for c,n in zip(colgroups, dtype.names):
        #    print(c.shape, n, dtype.fields[n][0].shape)
        # TODO: can't we create the structured array directly from the rows?
        return np.core.records.fromarrays(colgroups, dtype=dtype)

    def _compute_moments(self):
        if self._mean is None or self._var is None:
            pcoords = self._colvars
            if pcoords.dtype.names is None:
                self._mean = np.mean(pcoords, axis=0)
                self._var = np.var(pcoords, axis=0)
            else:
                self._mean = np.zeros(1, pcoords.dtype)
                for n in pcoords.dtype.names:
                    self._mean[n] = np.mean(pcoords[n], axis=0)
                self._var = np.zeros(1, pcoords.dtype)
                for n in pcoords.dtype.names:
                    self._var[n] = np.var(pcoords[n], axis=0)
            #mean_free = pcoords - mean[np.newaxis, :]
            #cov = np.dot(mean_free.T, mean_free) / pcoords.shape[0]
            #self._mean = mean
            #self._cov = cov

    def __getitem__(self, items):
        return self._colvars[items]

    def __len__(self):
        return self._colvars.shape[0]

    def as2D(self, fields):
        'Convert to plain 2-D numpy array where the first dimension is time'
        return structured_to_flat(self._colvars, fields=fields)

    @property
    def mean(self):
        self._compute_moments()
        return self._mean

    @property
    def var(self):
        self._compute_moments()
        return self._var

    @property
    def cov(self):
        self._compute_moments()
        return self._cov

    def overlap_plane(self, other, indicator='max'):
        'Compute overlap between two distributions from assignment error of a support vector machine trained on the data from both distributions.'
        import sklearn.svm
        clf = sklearn.svm.LinearSVC()
        X_self = self.as2D(fields=All)
        X_other = other.as2D(fields=All)
        X = np.vstack((X_self, X_other))
        n_self = X_self.shape[0]
        n_other = X_other.shape[0]
        labels = np.zeros(n_self + n_other, dtype=int)
        labels[n_self:] = 1
        clf.fit(X, labels)
        c = np.zeros((2, 2), dtype=int)
        p_self = clf.predict(X_self)
        p_other = clf.predict(X_other)
        c[0, 0] = np.count_nonzero(p_self == 0)
        c[0, 1] = np.count_nonzero(p_self == 1)
        c[1, 0] = np.count_nonzero(p_other == 0)
        c[1, 1] = np.count_nonzero(p_other == 1)
        c_sum = c.sum(axis=1)
        c_norm = c / c_sum[:, np.newaxis]
        if indicator == 'max':
            return max(c_norm[0, 1], c_norm[1, 0])
        elif indicator == 'min':
            return min(c_norm[0, 1], c_norm[1, 0])
        elif indicator == 'down':
            return c_norm[1, 0]  # other -> self
        elif indicator == 'up':
            return c_norm[0, 1]  # self -> other
        else:
            return c_norm

    def overlap_Bhattacharyya(self, other):
        # https://en.wikipedia.org/wiki/Bhattacharyya_distance
        #if self.endpoint or other.endpoint:
        #    return 0
        half_log_det_s1 = np.sum(np.log(np.diag(np.linalg.cholesky(self.cov))))
        half_log_det_s2 = np.sum(np.log(np.diag(np.linalg.cholesky(other.cov))))
        s = 0.5*(self.cov + other.cov)
        half_log_det_s = np.sum(np.log(np.diag(np.linalg.cholesky(s))))
        delta = self.mean - other.mean
        return 0.125*np.dot(delta, np.dot(np.linalg.inv(s), delta)) + half_log_det_s - 0.5*half_log_det_s1 - 0.5*half_log_det_s2

    @staticmethod
    def arclength_projection(points, nodes, order=0, return_z=False):
        if order not in [0, 1, 2]:
            raise ValueError('order must be 0 or 2, other orders are not implemented')
        nodes = np.array(nodes)
        if nodes.ndim != 2:
            raise NotImplementedError('Nodes with ndim > 1 are not supported.')
        results_s = []
        results_z = []
        ## TODO: this is kind of a hack, replace by the correct soltution later
        #if order in [1, 2]:
        #    delta =
        for x in points:
            i = np.argmin(np.linalg.norm(x[np.newaxis, :] - nodes, axis=1))
            if order == 0:
                results_s.append(i)
            elif order == 1:
                if i == 0:
                    i0 = 0
                elif i == len(nodes) - 1:
                    i0 = len(nodes) - 2
                else:
                    if np.linalg.norm(x - nodes[i + 1, :]) < np.linalg.norm(x - nodes[i - 1, :]):
                        i0 = i
                    else:
                        i0 = i - 1
                results_s.append(i0 + length_along_segment(a=nodes[i0], b=nodes[i0 + 1], x=x, clamp=False))
            elif order == 2:  # equation from Grisell Díaz Leines and Bernd Ensing. Phys. Rev. Lett., 109:020601, 2012
                i = np.argmin(np.linalg.norm(x[np.newaxis, :] - nodes, axis=1))
                if i == 0 or i == len(nodes) - 1:  # do orthogonal projection in the first and last segments
                    #print('end node', i)
                    if i == 0:
                        i0 = 0
                    else:
                        i0 = len(nodes) - 2
                    results_s.append(i0 + length_along_segment(a=nodes[i0], b=nodes[i0 + 1], x=x, clamp=False))
                    #if return_z:
                    #    results_z.append(np.linalg.norm(x - (a + s*(b - a))))
                else:  # for all other segments, continue with Leines and Ensing
                    mid = nodes[i, :]
                    plus = nodes[i + 1, :]
                    minus = nodes[i - 1, :]
                    #print('mid point', i, i+1, i-1)
                    v3 = plus - mid
                    v3v3 = np.vdot(v3, v3)
                    di = 1. #np.sign(i2 - i1)
                    v1 = mid - x
                    v2 = x - minus
                    v1v3 = np.vdot(v1, v3)
                    v1v1 = np.vdot(v1, v1)
                    v2v2 = np.vdot(v2, v2)
                    # TODO: test these expressions
                    f = ((v1v3**2 - v3v3*(v1v1 - v2v2))**0.5 - v1v3) / v3v3  # minus is the correct sign here!
                    s = (f - 1.)*0.5
                    results_s.append(i + di*s)
                    if return_z:
                        results_z.append(np.linalg.norm(x - f*v3 + mid))  # TODO: test
        if return_z:
            return results_s, results_z
        else:
            return results_s

    def closest_point(self, x):
        'Find the replica which is closest to x in order parameters space.'
        #print('input shapes', self.as2D.shape, recscalar_to_vector(x).shape)
        dist = np.linalg.norm(self.as2D(fields=self.fields) - structured_to_flat(x, fields=self.fields), axis=1)
        #print('dist shape', dist.shape)
        i = np.argmin(dist)
        return {'i': int(i),
                'd': dist[i],
                'x': self._colvars[i]}

    def distance_of_mean(self, other, rmsd=True):
        'Compute the distance between the mean of this window and the mean of some other window.'
        if rmsd:
            n_atoms = len(self._colvars.dtype.names)
            # TODO: assert the each atoms entry has dimension 3
        else:
            n_atoms = 1
        return np.linalg.norm(structured_to_flat(self.mean) - structured_to_flat(other.mean, fields=self.fields)) * (n_atoms ** -0.5)

    @property
    def fields(self):
        'Variable names (from colvars.traj header or npy file)'
        return self._colvars.dtype.names

    @property
    def dims(self):
        res = []
        for n in self.fields:
            shape = self._colvars.dtype.fields[n][0].shape
            if len(shape) == 0:
                res.append(1)
            else:
                res.append(shape[0])
        return res