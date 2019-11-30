import numpy as np
import os
import warnings
from .util import All, AllType, structured_to_flat, length_along_segment, recarray_dims

__all__ = ['Colvars', 'overlap_svm']


# TODO: move to some other subpackage
def overlap_svm(X_self, X_other, indicator='max', return_classifier=False):
    import sklearn.svm
    if X_self.ndim != 2:
        raise ValueError('X_self must be 2-D')
    if X_self.dtype not in [np.float64, np.float32]:
        raise ValueError('X_self must be an array of floating point numbers')
    if X_other.ndim != 2:
        raise ValueError('X_other must be 2-D')
    if X_other.dtype not in [np.float64, np.float32]:
        raise ValueError('X_other must be an array of floating point numbers')
    if X_self.shape[1] != X_other.shape[1]:
        raise ValueError('X_self and X_other must have the same number of dimensions.')
    X = np.vstack((X_self, X_other))
    n_self = X_self.shape[0]
    n_other = X_other.shape[0]
    labels = np.zeros(n_self + n_other, dtype=int)
    labels[n_self:] = 1
    mean = np.mean(X, axis=0)
    std = np.maximum(np.std(X, axis=0), 1.E-6)
    clf = sklearn.svm.LinearSVC(max_iter=10000)
    clf.fit((X - mean) / std, labels)
    c = np.zeros((2, 2), dtype=int)
    p_self = clf.predict((X_self - mean) / std)
    p_other = clf.predict((X_other - mean) / std)
    c[0, 0] = np.count_nonzero(p_self == 0)
    c[0, 1] = np.count_nonzero(p_self == 1)
    c[1, 0] = np.count_nonzero(p_other == 0)
    c[1, 1] = np.count_nonzero(p_other == 1)
    c_sum = c.sum(axis=1)
    c_norm = c / c_sum[:, np.newaxis]
    if indicator == 'max':
        return_value = max(c_norm[0, 1], c_norm[1, 0])
    elif indicator == 'min':
        return_value = min(c_norm[0, 1], c_norm[1, 0])
    elif indicator == 'down':
        return_value = c_norm[1, 0]  # other -> self
    elif indicator == 'up':
        return_value = c_norm[0, 1]  # self -> other
    else:
        return_value = c_norm
    if return_classifier:
        return return_value, clf
    else:
        return return_value


class Colvars(object):
    # TODO: should we unify this with Pyemma?
    def __init__(self, folder, base, fields=All, ignore_step_column=True):
        'Load file from folder/base(.npy|.colvars.traj|.pdb|.pdb.gz)'
        self._mean = None
        #self._var = None
        self._cov = None
        fname = folder + '/' + base
        if os.path.exists(fname + '.npy'):
            self._colvars = np.load(fname + '.npy')
            if self._colvars.dtype.names is None:
                raise ValueError('File %s.npy does not contain a structured array with CV name information. Giving up.' % fname)
            if len(self._colvars.shape) != 1:
                raise ValueError('File %s.npy has an incorrect shape and does not seem to contain frame data.' % fname)
            self._type = 'npy'
            if not isinstance(fields, AllType):
                self._colvars = self._colvars[fields]
        elif os.path.exists(fname + '.colvars.traj'):
            self._colvars = Colvars.load_colvar(fname + '.colvars.traj', selection=fields,
                                                ignore_step_column=ignore_step_column)
            self._type = 'colvars.traj'
        elif os.path.exists(fname + '.pdb'):
            self._colvars = Colvars.load_pdb_traj(fname + '.pdb', selection=fields)
            self._type = 'pdb'
        elif os.path.exists(fname + '.pdb.gz'):
            self._colvars = Colvars.load_pdb_traj(fname + '.pdb.gz', selection=fields)
            self._type = 'pdb'
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
    def load_pdb_traj(fname, selection=All):
        'Load PDB file or PDB trajectory and convert to numpy recarray.'
        # TODO: should we support mdtraj selection strings?
        import mdtraj
        traj = mdtraj.load(fname)
        if isinstance(selection, AllType):
            fields = ['%s_%s%d' % (a.name, a.residue.name[0:3], a.residue.resSeq) for a in traj.top.atoms]
            indices = slice(0, None, None)
        else:
            fields = []
            indices = []
            for i, a in enumerate(traj.top.atoms):
                field = '%s_%s%d' % (a.name, a.residue.name[0:3], a.residue.resSeq)
                if field in selection:
                    fields.append(field)
                    indices.append(i)
        dtype = np.dtype([(name, np.float64, 3) for name in fields])
        data = np.core.records.fromarrays(np.transpose(traj.xyz[:, indices, :], axes=(1, 0, 2)) * 10, dtype=dtype)
        return data

    @staticmethod
    def load_colvar(fname, selection=All, ignore_step_column=True):
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

        if var_names[0] == 'step':
            if len(np.unique(data[:, 0])) != data.shape[0]:
                warnings.warn('Colvar file %s contains repeated (non-unique) time steps. Proceed with caution.' % fname)

        if ignore_step_column and var_names[0] == 'step':
            data = data[:, 1:]  # delete first column
            var_names = var_names[1:]
            dims = dims[1:]

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
                    dtype_def.append((name, np.float64))
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
                for n, dim in zip(self.fields, self.dims):
                    column_mean = np.mean(pcoords[n], axis=0)
                    self._mean[n] = column_mean
                    mean_flat[k:k + dim] = column_mean
                    k += dim
                self._var = np.zeros(1, pcoords.dtype)
                for n in self.fields:
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

    def overlap_3D_units(self, other, indicator='max'):
        r'''Compute the minimum of the overlap scores for each Cartesian subunit.
        '''
        from .util import exactly_2d
        if set(self.fields) != set(other.fields):
            raise ValueError('Attempted to compute the overlap of two sets with different dimensions. Giving up.')
        overlap_1D = []
        for field in self.fields:
            o = overlap_svm(exactly_2d(self._colvars[field], add_frame_axis=False),
                            exactly_2d(other._colvars[field], add_frame_axis=False), indicator=indicator)
            overlap_1D.append(o)
        return np.min(overlap_1D)  # or return product? / sum log?

    def overlap_plane(self, other, indicator='max', return_plane=False):
        r'''Computes overlap between two distributions from classifcation error of a support vector machine trained on the data from both distributions.

            Parameters
            ----------
            other : Colvars
               Other colvars object to compute overlap with. Fields must match.

            indicator : string, one of "max", "min", "down", "up"
                * up: fraction of self misclassified as self as other
                * down: fraction of other misclassified as self
                * min: mimimum of up and down
                * max: maximum of up and down

            return_plane: bool
                Whether to return the parameters for the separating hyperplane

            Returns
            -------
            If return_plane is False, just returns the scalar overlap score.
            If return_plane is True, return the triple (score, normal, b)
            score: float
                overlap score between 0 and 1. The higher, the better the overlap.
            normal: ndarray(ndim)
                normal vector of the hyperplane
            b : float
                intercept parameter b of the hyperplane
         '''

        if set(self.fields) != set(other.fields):
            raise ValueError('Attempted to compute the overlap of two sets with different dimensions. Giving up.')
            # TODO: have option that selects the overlap (what?) automatically
        X_self = self.as2D(fields=All)
        X_other = other.as2D(fields=All)
        overlap, clf = overlap_svm(X_self, X_other, indicator='max', return_classifier=True)
        if return_plane:
            return overlap, clf.coef_[0], clf.intercept_[0]
        else:
            return overlap

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
        if order not in [0, 0.5, 1, 2]:
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
        elif order == 0.5:
            # This is a fast approximate first order method
            for x in points:
                i = np.argmin(np.linalg.norm(x[np.newaxis, :] - nodes, axis=1))
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
            return results_s
        elif order == 2:
            for x in points:
                i = np.argmin(np.linalg.norm(x[np.newaxis, :] - nodes, axis=1))
                if i == 0 or i == len(nodes) - 1:  # do orthogonal projection in the first and last segments
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
        r'''Find the replica which is closest to x in order parameters space.

        Returns
        -------
        the optimal search result, encoded as a dict with keys:
        * i: the frame index (discrete time step) of the optimal result
        * d: the distance to the optimal point
        * x: the order parameters of the optimal point
        '''
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
        return recarray_dims(self._colvars)
        #res = []
        #for n in self.fields:
        #    shape = self._colvars.dtype.fields[n][0].shape
        #    if len(shape) == 0:
        #        res.append(1)
        #    else:
        #        res.append(shape[0])
        #return res
