import numpy as np
import os
import warnings
from .util import All, AllType, structured_to_flat, length_along_segment

__all__ = ['Colvars']


class Colvars(object):
    def __init__(self, folder, base, fields=All):
        self._mean = None
        #self._var = None
        self._cov = None
        fname = folder + '/' + base
        if os.path.exists(fname + '.npy'):
            self._colvars = np.load(fname + '.npy')
            if not isinstance(fields, AllType):
                self._colvars = self._colvars[fields]
        elif os.path.exists(fname + '.colvars.traj'):
            self._colvars = Colvars.load_colvar(fname + '.colvars.traj', selection=fields)
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
        if self._mean is None or self._cov is None:
            pcoords = self._colvars
            if pcoords.dtype.names is None:
                warnings.warn('Colvars that were loaded form file are not in structured format. This case is not fully supported. Expect things to fail.')
                self._mean = np.mean(pcoords, axis=0)
                self._var = np.var(pcoords, axis=0)
                self._cov = np.dot(pcoords.T, pcoords) / pcoords.shape[0]
            else:
                self._mean = np.zeros(1, pcoords.dtype)
                mean_flat = np.zeros(sum(self.dims))
                k = 0
                for n in pcoords.dtype.names:
                    column_mean = np.mean(pcoords[n], axis=0)
                    self._mean[n] = column_mean
                    mean_flat[k:k + column_mean.shape[0]] = column_mean
                    k += column_mean.shape[0]
                self._var = np.zeros(1, pcoords.dtype)
                for n in pcoords.dtype.names:
                    self._var[n] = np.var(pcoords[n], axis=0)
                X = self.as2D(self.fields) - mean_flat[np.newaxis, :]
                self._cov = np.dot(X.T, X) / len(self)

    def __getitem__(self, items):
        return self._colvars[items]

    def __len__(self):
        return self._colvars.shape[0]

    def as2D(self, fields):
        'Convert to plain 2-D numpy array where the first dimension is time'
        return structured_to_flat(self._colvars, fields=fields)

    def as3D(self, fiels):
        'For set of Cartesian coordinates, convert to (n_time_steps, n_atoms, 3) ndarray.'
        raise NotImplementedError()

    @property
    def mean(self):
        self._compute_moments()
        return self._mean

    def bootstrap_mean(self):
        'Perform bootstrap and return a sample for the mean.'
        indices = np.random.randint(low=0, high=len(self)-1, size=len(self))
        mean = np.zeros(1, self._colvars.dtype)
        for n in self._colvars.dtype.names:
            mean[n] = np.mean(self._colvars[n][indices], axis=0)
        return mean

    @property
    def var(self):
        self._compute_moments()
        return self._var

    @property
    def cov(self):
        self._compute_moments()
        return self._cov

    @property
    def error_matrix(self):
        'Squared standard error of the mean (full matrix, including covariance structure)'
        self._compute_moments()
        return self._cov / len(self)

    def overlap_plane(self, other, indicator='max'):
        'Compute overlap between two distributions from assignment error of a support vector machine trained on the data from both distributions.'
        import sklearn.svm
        clf = sklearn.svm.LinearSVC(max_iter=10000)
        if set(self.fields) != set(other.fields):
            raise ValueError('Attempted to compute the overlap of two sets with different dimensions. Giving up.')
            # TODO: have option that selects the overlap automatically
        X_self = self.as2D(fields=All)
        X_other = other.as2D(fields=All)
        X = np.vstack((X_self, X_other))
        n_self = X_self.shape[0]
        n_other = X_other.shape[0]
        labels = np.zeros(n_self + n_other, dtype=int)
        labels[n_self:] = 1
        mean = np.mean(X, axis=0)
        std = np.maximum(np.std(X, axis=0), 1.E-6)
        clf.fit((X - mean) / std, labels)
        c = np.zeros((2, 2), dtype=int)
        p_self = clf.predict((X_self - mean)/std)
        p_other = clf.predict((X_other - mean)/std)
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
        half_log_det_s1 = np.sum(np.log(np.diag(np.linalg.cholesky(self.cov))))
        half_log_det_s2 = np.sum(np.log(np.diag(np.linalg.cholesky(other.cov))))
        s = 0.5*(self.cov + other.cov)
        half_log_det_s = np.sum(np.log(np.diag(np.linalg.cholesky(s))))
        delta = self.mean - other.mean
        return 0.125*np.dot(delta, np.dot(np.linalg.inv(s), delta)) + half_log_det_s - 0.5*half_log_det_s1 - 0.5*half_log_det_s2

    @staticmethod
    def arclength_linear(x, curve, return_foot=False):
        r'''Arc length of x along the polyline the goes through the points given parameter curve.

            Parameters
            ----------
            x : ndarray((d,))
                Point x

            curve : ndarray((n, d))
                Points that define the polyline

            return_foot : boolean, optional, default=False
                Also return the full coordinates of x projected on the polyline

            Returns
            -------
            Project x onto the polyline and return the arc lengh of the projected point along the polyline.
            Result is a floating point number in the range 0 <= l <= len(curve) - 1.
            If return_points is True, also return full coordinates of projected x.
        '''
        distances = []
        sigmas = []
        feet = []
        for i in range(len(curve) - 1):
            a = curve[i]
            b = curve[i + 1]
            sigma = max(min(np.vdot(x - a, b - a) / np.vdot(b - a, b - a), 1.0), 0.0)
            foot = (1. - sigma) * a + sigma * b
            distances.append(np.linalg.norm(x - foot))
            sigmas.append(sigma)
            feet.append(foot)
        i_min = np.argmin(distances)
        if return_foot:
            return i_min + sigmas[i_min], feet[i_min]
        else:
            return i_min + sigmas[i_min]

    @staticmethod
    def arclength_projection(points, nodes, order=0, return_z=False):
        if return_z:
            raise NotImplementedError('return_z not yet implemented')
        if order not in [0, 1, 2]:
            raise ValueError('order must be 0 or 2, other orders are not implemented')
        nodes = np.array(nodes)
        if nodes.ndim != 2:
            raise NotImplementedError('Nodes with ndim > 1 are not supported.')
        results_s = []
        results_z = []
        if order == 1:
            return [Colvars.arclength_linear(x, curve=nodes) for x in points]
        elif order == 0:
            return [np.argmin(np.linalg.norm(x[np.newaxis, :] - nodes, axis=1)) for x in points]
        else:
            pass  # use legacy version below

        # TODO: this (identifying the closest segment by looking for the two closest points) is kind of a hack (it's generally wrong in fact),
        # TODO: replace by the correct solution later
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
                    if np.linalg.norm(x - nodes[i + 1, :]) < np.linalg.norm(x - nodes[i - 1, :]):  # based on the closest two nodes
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
                    s = (f - 1.)*0.5  # TODO: minus correct here?
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
        return list(self._colvars.dtype.names)

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