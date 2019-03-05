import numpy as np
import os
import glob
import warnings
import collections
import concurrent.futures
import subprocess
import yaml  # replace with json (more portable and future proof)
import tempfile
import shutil
import errno
import time
from pyemma.util.annotators import deprecated
from .reparametrization import *


__all__ = ['String', 'Image', 'root', 'load', 'main', 'init', 'is_sim_id', 'All']
__author__ = 'Fabian Paul <fab@physik.tu-berlin.de>'


def root():
    'Return absolute path to the root directory of the project.'
    if 'STRING_SIM_ROOT' in os.environ:
        return os.environ['STRING_SIM_ROOT']
    else:
        folder = os.path.realpath('.')
        while not os.path.exists(os.path.join(folder, '.sarangirc')) and folder != '/':
            # print('looking at', folder)
            folder = os.path.realpath(os.path.join(folder, '..'))
        if os.path.exists(os.path.join(folder, '.sarangirc')):
            return folder
        else:
            raise RuntimeError('Could not locate the project root. Environment variable STRING_SIM_ROOT is not set and no .sarangirc file was found.')

# TODO: provide a set of functions/methods that abstract away id and file name generation

# TODO: provide some plotting functions that takes a dictionary as input with entries image_id -> value

def is_sim_id(s):
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


class Universe(object):
    'x in All == True for any x'
    def __contains__(self, key):
        return True

    def __repr__(self):
        return 'All'


All = Universe()


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
        if isinstance(value, list):
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
        array[name] = value
    return array


def dump_structured(array):
    'Convert numpy structured array to python dictionary'
    config = {}
    for n in array.dtype.names:
        if len(array.dtype.fields[n][0].shape) == 1:  # vector type
            config[n] = [float(x) for x in array[n][0]]
        elif len(array.dtype.fields[n][0].shape) == 0:  # scalar type
            config[n] = float(array[n])
        else:
            raise RuntimeError('unsupported dimension')
    return config


def recscalar_to_vector(recarray):
    'Convert numpy recarray with on entry ("scalar type") to a numpy vector (1-D array)'
    # TODO: also handle the case where the input is already a numpy vector?
    n_cols = sum(np.prod(recarray.dtype.fields[name][0].shape, dtype=int) for name in recarray.dtype.names)
    return recarray.view((float, n_cols))


def recarray_average(a, b):
    if a.dtype.names != b.dtype.names:
        raise ValueError('a and b must have the same fields')
    c = np.zeros_like(a)
    for name in a.dtype.names:
        c[name] = (a[name] + b[name]) * 0.5
    return c


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
        data = np.array(rows)

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
        dtype = np.dtype(dtype_def)

        indices = np.concatenate(([0], np.cumsum(dims)))
        # print(dtype)
        # print(indices)
        colgroups = [np.squeeze(data[:, start:stop]) for name, start, stop in zip(var_names, indices[0:-1], indices[1:])
                        if name in selection]
        # for c,n in zip(colgroups, dtype.names):
        #    print(c.shape, n, dtype.fields[n][0].shape)
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

    @property
    def as2D(self):
        'Convert to plain 2-D numpy array where the first dimension is time'
        if self._colvars.dtype.names is None:
            n = self._colvars.shape[0]
            return self._colvars.reshape((n, -1))
        else:
            n = self._colvars.shape[0]
            idx = np.cumsum([0] +
                            [np.prod(self._colvars.dtype.fields[name][0].shape, dtype=int) for name in self._colvars.dtype.names])
            m = idx[-1]
            x = np.zeros((n, m), dtype=float)
            for name, start, stop in zip(self._colvars.dtype.names, idx[0:-1], idx[1:]):
                #columns = self._colvars[name]
                #x[:, start:stop] = np.reshape(columns, (len(columns),-1))
                #print(name, ':', self._colvars[name].shape, '->', x[:, start:stop].shape)
                x[:, start:stop] = self._colvars[name]
            return x

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
        X_self = self.as2D
        X_other = other.as2D
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
    def arclength_projection(points, nodes, order=0):
        if order not in [0, 2]:
            raise ValueError('order must be 0 or 2, other orders are not implemented')
        nodes = np.array(nodes)
        if nodes.ndim != 2:
            raise NotImplementedError('Nodes with ndim > 1 are not supported.')
        results = []
        for x in points:
            i = np.argmin(np.linalg.norm(x[np.newaxis, :]-nodes, axis=1))
            if order == 0:
                results.append(i)
            elif order == 2:
                if i == 0 or i == len(nodes) - 1:
                    continue
                mid = nodes[i, :]
                plus = nodes[i + 1, :]
                minus = nodes[i - 1, :]
                v3 = plus - mid
                v3v3 = np.vdot(v3, v3)
                di = 1. #np.sign(i2 - i1)
                v1 = mid - x
                v2 = x - minus
                v1v3 = np.vdot(v1, v3)
                v1v1 = np.vdot(v1, v1)
                v2v2 = np.vdot(v2, v2)
                results.append(
                     i + (di*(v1v3**2 - v3v3*(v1v1 - v2v2))**0.5 - v1v3 - v3v3) / (2*v3v3))
        return results

    def closest_point(self, x):
        'Find the replica which is closest to x in order parameters space.'
        #print('input shapes', self.as2D.shape, recscalar_to_vector(x).shape)
        dist = np.linalg.norm(self.as2D - recscalar_to_vector(x), axis=1)
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
        return np.linalg.norm(recscalar_to_vector(self.mean) - recscalar_to_vector(other.mean)) * (n_atoms ** -0.5)

    @property
    def fields(self):
        'Variable names (from colvars.traj header or npy file)'
        return self._colvars.dtype.names


class Image(object):
    def __init__(self, image_id, previous_image_id, previous_frame_number,
                 node, spring, endpoint, atoms_1):
        self.image_id = image_id
        self.previous_image_id = previous_image_id
        self.previous_frame_number = previous_frame_number
        self.node = node
        self.spring = spring
        self.endpoint = endpoint
        self.atoms_1 = atoms_1
        self._pcoords = {}
        self._x0 = {}

    def copy(self, image_id=None, previous_image_id=None, previous_frame_number=None,
             node=None, spring=None, endpoint=None, atoms_1=None):
        'Copy the Image object, allowing parameter changes'
        if image_id is None:
            image_id = self.image_id
        if previous_image_id is None:
            previous_image_id = self.previous_image_id
        if previous_frame_number is None:
            previous_frame_number = self.previous_frame_number
        if node is None:
            node = self.node.copy()
        if spring is None:
            spring = self.spring.copy()
        if atoms_1 is None:
            atoms_1 = self.atoms_1

        if endpoint is None:
            endpoint = self.endpoint
        #image_id = '{branch}_{iteration:03d}_{major_id:03d}_{minor_id:03d}'.format(
        #                branch = branch, iteration=iteration, major_id=major_id, minor_id=minor_id
        #            )
        return Image(image_id=image_id, previous_image_id=previous_image_id,
                     previous_frame_number=previous_frame_number,
                     node=node, spring=spring, endpoint=endpoint, atoms_1=atoms_1)

    @classmethod
    def load(cls, config):
        'Load file from dictionary. This function is called by String.load'
        image_id = config['id']
        previous_image_id = config['prev_image_id']
        previous_frame_number = config['prev_frame_number']
        if 'node' in config:
            node = load_structured(config['node'])
        else:
            node = None
        if 'spring' in config:
            spring = load_structured(config['spring'])
        else:
            spring = None
        if 'atoms_1' in config:
            atoms_1 = config['atoms_1']
        else:
            atoms_1 = None
        if 'endpoint' in config:
            endpoint = config['endpoint']
        else:
            endpoint = False
        return Image(image_id=image_id, previous_image_id=previous_image_id,
                     previous_frame_number=previous_frame_number,
                     node=node, spring=spring, endpoint=endpoint, atoms_1=atoms_1)

    def dump(self):
        'Dump state of object to dictionary. Called by String.dump'
        config = {'id': self.image_id, 'prev_image_id': self.previous_image_id,
                  'prev_frame_number': self.previous_frame_number}
        if self.node is not None:
            config['node'] = dump_structured(self.node)
        if self.spring is not None:
            config['spring'] = dump_structured(self.spring)
        if self.atoms_1 is not None:
            config['atoms_1'] = self.atoms_1
        if self.endpoint is not None and self.endpoint:
            config['endpoint'] = self.endpoint
        return config

    @property
    def propagated(self):
        return os.path.exists(self.base + '.dcd') #and os.path.exists(self.base + '.colvars.traj')

    def _make_job_file(self, env):
        'Created a submission script for the job on the local file system.'
        with open('%s/setup/jobfile.template' % root()) as f:
            template = ''.join(f.readlines())
            environment = '\n'.join(['export %s=%s' % (k, v) for k, v in env.items()])
        with tempfile.NamedTemporaryFile(suffix='.sh', delete=False) as f:
            f.write(template.format(job_name=self.job_name, environment=environment).encode(encoding='UTF-8'))
            job_file_name = f.name
        return job_file_name

    @property
    def branch(self):
        return self.image_id.split('_')[0]

    @property
    def iteration(self):
        return int(self.image_id.split('_')[1])

    @property
    def id_major(self):
        return int(self.image_id.split('_')[2])

    @property
    def id_minor(self):
        return int(self.image_id.split('_')[3])

    @property
    def seq(self):
        'id_major.in_minor as a floating point number'
        return float(str(self.id_major) + '.' + str(self.id_minor))

    #def id_str(self, arclength):
    #    return '%3s_%03d_%03d_%03d' % (self.branch, self.iteration, self.id_major, self.id_minor)

    @property
    def job_name(self):
        return 'im_' + self.image_id

    @property
    def base(self):
        'Base path of the image. base+".dcd" are the replicas, base+".dat" are the order parameters'
        return '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
               root=root(), branch=self.branch, iteration=self.iteration, id_minor=self.id_minor, id_major=self.id_major)

    @property
    def previous_base(self):
        branch, iteration, id_major, id_minor = self.previous_image_id.split('_')
        return '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
               root=root(), branch=branch, iteration=int(iteration), id_minor=int(id_minor), id_major=int(id_major))

    def submitted(self, queued_jobs):
        return self.job_name in queued_jobs

    def _make_env(self, random_number):
        env = dict()
        root_ = root()
        env['STRING_SIM_ROOT'] = root_
        env['STRING_ITERATION'] = str(self.iteration)
        env['STRING_IMAGE_ID'] = self.image_id
        env['STRING_PLAN'] = '{root}/strings/{branch}_{iteration:03d}/plan.yaml'.format(root=root_, branch=self.branch, iteration=self.iteration)
        env['STRING_PREV_IMAGE_ID'] = self.previous_image_id
        env['STRING_PREV_FRAME_NUMBER'] = str(self.previous_frame_number)
        env['STRING_RANDOM'] = str(random_number)
        env['STRING_PREV_ARCHIVE'] = self.previous_base
        env['STRING_ARCHIVE'] = self.base
        env['STRING_ARCHIVIST'] = os.path.dirname(__file__) + '/string_archive.py'
        env['STRING_SARANGI_SCRIPTS'] = os.path.dirname(__file__) + '/../scripts'
        return env

    def propagate(self, random_number, wait, queued_jobs, run_locally=False, dry=False):
        'Generic propagation command. Submits jobs for the intermediate points. Copies the end points.'
        if self.propagated:
            #print(self.job_name, 'already completed')
            return self

        #  if the job is already queued, return or wait and return then
        if self.job_name in queued_jobs:
            print('skipping submission of', self.job_name, 'because already queued')
            if wait:  # poll the results file
                while not self.propagated:
                    time.sleep(30)
            else:
                return self

        env = self._make_env(random_number=random_number)

        if run_locally:
            job_file = self._make_job_file(env)
            print('run', job_file, '(', self.job_name, ')')
            if not dry:
                subprocess.run('bash ' + job_file, shell=True)  # directly execute the job file
        else:
            job_file = self._make_job_file(env)
            if wait:
                command = 'qsub --wait ' + job_file  # TODO: slurm (sbatch)
                print('run', command, '(', self.job_name, ')')
                if not dry:
                    subprocess.run(command, shell=True)  # debug
            else:
                command = 'qsub ' + job_file  # TODO: slurm (sbatch)
                print('run', command, '(', self.job_name, ')')
                if not dry:
                    subprocess.run(command, shell=True)  # debug

        return self

    @deprecated
    def closest_replica(self, x):  # TODO: rewrite!!!
        # TODO: offer option to search for a replica that is closest to a given plane
        'Find the replica which is closest to x in order parameters space.'
        assert self.propagated
        dist = np.linalg.norm(self.colvars - x[np.newaxis, :], axis=1)
        i = np.argmin(dist)
        return int(i), dist[i]

    def colvars(self, subdir='colvars', fields=All, memoize=True):
        'Return Colvars object for the set of collective variables saved in a given subdir and limited to given fields'
        if isinstance(fields, list):
            fields = tuple(fields)
        if (subdir, fields) in self._pcoords:
            return self._pcoords[(subdir, fields)]
        else:
            folder = '{root}/observables/{branch}_{iteration:03d}/'.format(
                   root=root(), branch=self.branch, iteration=self.iteration)
            base = '{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
                   branch=self.branch, iteration=self.iteration, id_minor=self.id_minor, id_major=self.id_major)
            pcoords = Colvars(folder=folder + subdir, base=base, fields=fields)
            if memoize:
                self._pcoords[(subdir, fields)] = pcoords
            return pcoords

    def overlap_plane(self, other, subdir='colvars', fields=All, indicator='max'):
        'Compute overlap between two distributions from assignment error of a support vector machine trained on the data from both distributions.'
        return self.colvars(subdir=subdir, fields=fields).overlap_plane(other.colvars(subdir=subdir, fields=fields),
                                                                        indicator=indicator)

    def x0(self, subdir='colvars', fields=All):  # TODO: have this as a Colvars object?
        'Get the initial position for the simualtion in colvar space.'
        if subdir not in self._x0:
            branch, iteration, id_major, id_minor = self.previous_image_id.split('_')
            folder = '{root}/observables/{branch}_{iteration:03d}/'.format(
                   root=root(), branch=branch, iteration=int(iteration))
            base = '{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
                   branch=branch, iteration=int(iteration), id_minor=int(id_minor), id_major=int(id_major))
            self._x0[subdir] = Colvars(folder=folder + subdir, base=base, fields=fields)[self.previous_frame_number]
        return self._x0[subdir]

    @deprecated
    def bar_RMSD(self, other, T=303.15):
        import pyemma
        RT = 1.985877534E-3 * T  # kcal/mol
        id_self = '%03d_%03d' % (self.id_major, self.id_minor)
        id_other = '%03d_%03d' % (other.id_major, other.id_minor)
        p_self = self.colvars(subdir='RMSD')
        p_other = other.colvars(subdir='RMSD')
        btrajs = [np.zeros((len(p_self), 2)), np.zeros((len(p_other), 2))]
        btrajs[0][:, 0] = p_self[id_self][:]**2 * 5.0 / RT
        btrajs[0][:, 1] = p_self[id_other][:]**2 * 5.0 / RT
        btrajs[1][:, 0] = p_other[id_self][:]**2 * 5.0 / RT
        btrajs[1][:, 1] = p_other[id_other][:]**2 * 5.0 / RT
        ttrajs = [np.zeros(len(p_self), dtype=int), np.ones(len(p_other), dtype=int)]
        mbar = pyemma.thermo.MBAR()
        mbar.estimate((ttrajs, ttrajs, btrajs))
        return mbar.f_therm[0] - mbar.f_therm[1]

    @staticmethod
    def potential(x, node, spring):
        'Compute the bias potential parametrized by node and spring evaluated along the possibly multidimensional order parameter x.'
        u = None
        for name in x.fields:
            #print('x', x[name].shape)
            #print('spring', spring[name].shape)
            #print('node', node[name].shape)
            u_part = spring[name] * np.linalg.norm(x[name] - node[name], axis=1)**2
            assert u_part.ndim == 1
            #print('pot', u_part.shape, u_part.ndim)
            if u is None:
                u = u_part
            else:
                u += u_part
        return u
    
    def bar(self, other, subdir='colvars', T=303.15):
        'Compute thermodynamic free energy difference between this window (self) and other using BAR.'
        import pyemma
        RT = 1.985877534E-3 * T  # kcal/mol
        fields = self.node.dtype.names
        my_x = self.colvars(subdir=subdir, fields=fields)
        other_x = other.colvars(subdir=subdir, fields=fields)
        btrajs = [np.zeros((len(my_x), 2)), np.zeros((len(other_x), 2))]
        btrajs[0][:, 0] = Image.potential(my_x, self.node, self.spring) / RT
        btrajs[0][:, 1] = Image.potential(my_x, other.node, other.spring) / RT
        btrajs[1][:, 0] = Image.potential(other_x, self.node, self.spring) / RT
        btrajs[1][:, 1] = Image.potential(other_x, other.node, other.spring) / RT
        ttrajs = [np.zeros(len(my_x), dtype=int), np.ones(len(other_x), dtype=int)]
        mbar = pyemma.thermo.MBAR()
        mbar.estimate((ttrajs, ttrajs, btrajs))
        return mbar.f_therm[0] - mbar.f_therm[1]

    def displacement(self, subdir='colvars', fields=All, norm='rmsd', origin='node'):
        'Compute the difference between the windows\'s mean and its initial position in order parameter space'
        if origin == 'x0':
            o = self.x0(subdir=subdir, fields=fields)
        elif origin == 'node' or origin == 'center':
            if isinstance(fields, Universe):
                o = self.node
            else:
                o = self.node[fields]  # TODO: FIXME
        else:
            raise ValueError('origin must be either "node" or "x0"')
        mean = self.colvars(subdir=subdir, fields=fields).mean
        if norm=='rmsd':
            n_atoms = len(o.dtype.names)
        else:
            n_atoms = 1
        # TODO: have option to report the biggest displacement of individual atoms
        if norm != 'rmsd':
            ord = norm
        else:
            ord = None
        return np.linalg.norm(recscalar_to_vector(mean) - recscalar_to_vector(o), ord=ord) * (n_atoms ** -0.5)


def load_jobs_PBS():
    from subprocess import Popen, PIPE
    import xml.etree.ElementTree as ET
    process = Popen(['qstat', '-x',], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    jobs = []
    root = ET.fromstring(stdout)
    for node in root:
       job = {}
       for pair in node:
           if pair.tag in ['Job_Name', 'Job_Owner', 'job_state']:
               job[pair.tag] = pair.text
       jobs.append(job)
    return jobs


def get_queued_jobs_PBS():
    import getpass
    user = getpass.getuser()
    names = []
    for job in load_jobs_PBS():
        if job['Job_Owner'][0:len(user)] == user:
            names.append(job['Job_Name'])
    return names


def get_queued_jobs_SLURM():
    from subprocess import Popen, PIPE    
    import getpass
    user = getpass.getuser()
    process = Popen(['squeue', '-o', '%j', '-h', '-u', user], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()    
    return [j.strip() for j in stdout]


def get_queued_jobs():
    try:
        return get_queued_jobs_SLURM()
    except FileNotFoundError:
        try:
            return get_queued_jobs_PBS()
        except FileNotFoundError:
            return None  #_All


_Bias = collections.namedtuple('_Bias', ['ri', 'spring', 'rmsd_simids', 'bias_simids'], verbose=False)


def interpolate_id(s, z, others):
    'Create new simulation ID between s and z, preferentially leaving low digits zero.'
    others_int = []
    for id_ in others:
        _, _, a, b = id_.split('_')
        others_int.append(int(a + b))

    branch, iter_, a, b = s.split('_')
    _, _, c, d = z.split('_')
    x = a + b
    y = c + d
    upper = max(int(x), int(y))
    lower = min(int(x), int(y))
    # print('searching', upper, lower)
    result = between(0, 6, upper=upper, lower=lower, others=others_int)
    # print('searching', upper, lower)
    if not result:
        raise RuntimeError('Could not generate an unique id.')
    return '%s_%03s_%03d_%03d' % (branch, iter_, result // 1000, result % 1000)


def between(a, e, upper, lower, others):
    if e < 0:
        return False
    for i in [5, 4, 6, 3, 7, 2, 8, 9, 1, 0]:  # TODO: start half between upper and lower instead with 5
        x = a + 10 ** e * i
        if lower < x < upper and x not in others:
            # print('success', x)
            return x
    for i in [5, 4, 6, 3, 7, 2, 8, 9, 1, 0]:
        l = a + 10 ** e * i
        r = a + 10 ** e * (i + 1)
        if lower < r and upper > l:
            # print('recursing', a, l, r, e, i)
            result = between(l, e - 1, upper, lower, others)
            if result:
                return result
    # print('not found')
    return False


class String(object):
    def __init__(self, branch, iteration, images, image_distance, previous, opaque):
        self.branch = branch
        self.iteration = iteration
        self.images = images
        self.image_distance = image_distance
        self.previous = previous
        self.opaque = opaque

    def empty_copy(self, iteration=None, images=None, previous=None):
        'Creates a copy of the current String object. Image array is left empty.'
        if iteration is None:
            iteration = self.iteration
        if previous is None:
            previous = self.previous
        if images is None:
            images = dict()
        return String(branch=self.branch, iteration=iteration, images=images,
                      image_distance=self.image_distance, previous=previous, opaque=self.opaque)

    def add_image(self, image):
        'Add nwe image to string'
        if image.seq in self.images:
            raise ValueError('String already contains an image with sequence id %f, aborting operation.' % image.seq)
        else:
            self.images[image.seq] = image

    def __len__(self):
        'Number of images'
        return len(self.images)

    @property
    def images_ordered(self):
        'Images ordered by ID, where id_major.id_minor is interpreted as a flaoting point number'
        return [self.images[key] for key in sorted(self.images.keys())]

    @classmethod
    def from_scratch(cls, image_distance=1.0, branch='AZ', iteration_id=1):
        'Initialized a String from a folder of *.nc files.'
        n_images = len(glob.glob(root() + '/strings/AZ_000/*.dcd'))
        # TODO sort files and set endpoint properly!
        images = dict()
        for i in range(n_images):
            endpoint = (i==0 or i==n_images - 1)
            images[i] = \
                 Image(iteration_id=iteration_id, image_id=i, previous_iteration_id=0, previous_image_id=i,
                        previous_replica_id=0, node=None, endpoint=endpoint)
        return String(branch=branch, iteration=iteration_id, images=images, image_distance=image_distance, previous=None)

    def _launch_simulation(self, image, random_number, wait, queued_jobs, run_locally, dry):
        return image.propagate(random_number=random_number, wait=wait,
                               queued_jobs=queued_jobs, run_locally=run_locally, dry=dry)

    def propagate(self, wait=False, run_locally=False, dry=False):
        'Propagated the String (one iteration). Returns a modified copy.'
        if self.propagated:
            return self  # TODO: warn???
        # Does not increase the iteration number

        if run_locally:
            queued_jobs = []
        else:
            queued_jobs = get_queued_jobs()
            if queued_jobs is None:
                raise RuntimeError('queued jobs is undefined, I don\'t know which jobs are currently running. Giving up propagation.')

        #print('queued jobs', queued_jobs)

        l = len(self.images)
        propagated_images = dict()

        #print('propagating string, iteration =', self.iteration)

        mkdir('%s/strings/%s_%03d/' % (root(), self.branch, self.iteration))

        max_workers=1#l

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._launch_simulation, image, np.random.randint(0, high=np.iinfo(np.int64).max, size=1, dtype=np.int64)[0], wait, queued_jobs, run_locally, dry)
                       for image in self.images_ordered]
            for future in concurrent.futures.as_completed(futures):
                image = future.result()
                propagated_images[image.image_id] = image

        return self.empty_copy(images=propagated_images)

    def connected(self, threshold=0.1):
        'Test if all images are overlapping'
        if not self.propagated:
            raise RuntimeError('Trying to find connectedness of string that has not been (fully) propagated. Giving up.')
        return all(p.overlap_plane(q) >= threshold for p, q in zip(self.images_ordered[0:-1], self.images_ordered[1:]))

    def bisect_at(self, i, j=None, subdir='colvars', where='node', search='string', fields=All):
        r'''

        example
        -------
            ov, ids = string.overlap(return_ids=True)
            for pair, o in zip(ids, ov):
                if o < 0.05:
                     new_image = string.bisect_at(i=pair[0], j=pair[1])
                     string.add_image(new_image)
        '''
        if j is None:
            j += i + 1
        if isinstance(i, int):
            p = self.images_ordered[i]
        elif isinstance(i, float):
            p = self.images[i]
        else:
            raise ValueError('parameter i has incorrect type')
        if isinstance(j, int):
            q = self.images_ordered[j]
        elif isinstance(j, float):
            q = self.images[j]
        else:
            raise ValueError('parameter j has incorrect type')
        if where == 'mean':
            x = recarray_average(p.colvars(subdir=subdir, fields=fields).mean, q.colvars(subdir=subdir, fields=fields).mean)
        elif where == 'x0':
            x = recarray_average(p.x0(subdir=subdir, fields=fields), q.x0(subdir=subdir, fields=fields))
        elif where == 'node':
            x = recarray_average(p.node, q.node)
        elif where == 'plane':
            raise NotImplementedError('bisection at SVM plane not implemented yet')
        else:
            raise ValueError('Unrecognized value "%s" for option "where"' % where)

        if search == 'points':
            query_p = p.colvars(subdir=subdir, fields=fields).closest_point(x)
            query_q = q.colvars(subdir=subdir, fields=fields).closest_point(x)
            if query_p['d'] < query_q['d']:
                print('best distance is', query_p['d'])
                best_image = p
                best_step = query_p['i']
            else:
                print('best distance is', query_q['d'])
                best_image = q
                best_step = query_q['i']
        elif search == 'string':
            responses = []
            try:
                for im in self.images.values():
                    responses.append((im, im.colvars(subdir=subdir, fields=fields).closest_point(x)))
            except FileNotFoundError as e:
                warnings.warn(str(e))
            best_idx = np.argmin([r[1]['d'] for r in responses])
            best_image = responses[best_idx][0]
            best_step = responses[best_idx][1]['i']
            print('best distance is', responses[best_idx][1]['d'])
        else:
            raise ValueError('Unrecognized value "%s" of parameter "search"' % search)

        # TODO: make alternative where we only attempt to change the most significant digit
        #new_seq = (p.seq + q.seq) * 0.5
        #new_major = int(new_seq)
        #new_minor = int((new_seq - new_major)*1000)
        #new_image_id = '%s_%03d_%03d_%03d' %(best_image.branch, best_image.iteration, new_major, new_minor)
        new_image_id = interpolate_id(p.image_id, q.image_id, others=[im.image_id for im in self.images.values()])
        if new_image_id in (im.image_id for im in self.images.items):
            raise RuntimeError('Bisection produced new image id which is not unique. This should not happen.')

        new_image = Image(image_id=new_image_id, previous_image_id=best_image.image_id, previous_frame_number=best_step,
                          node=x, spring=best_image.spring, endpoint=False, atoms_1=best_image.atoms_1)

        #   self.images[new_image.seq] = new_image
        return new_image

    def bisect(self, ids):

        raise NotImplementedError('This is broken')
        subdir = 'colvars'
        ov_max, idx = self.overlap(subdir=subdir, indicator='max', return_ids=True)
        gaps = np.array(idx)[ov_max < 0.10]
        new_imgs = []
        new_string = self.empty_copy(images=string.images)  # keep current images, just add to them
        for gap in gaps:
            new_img = string.bisect_at(i=gap[0], j=gap[1], subdir=subdir, where='node')
            new_imgs.append(new_img)
            new_string.images[new_img.seq] = new_img
        return new_string  # new_string.write_yaml()



    def bisect_and_propagate_util_connected(self, run_locally=False):
        s = self
        while not s.connected():
            s = s.bisect().propagate(wait=True, run_locally=run_locally)
        return s

    @property
    def propagated(self):
        return all(image.propagated for image in self.images.values())

    def ribbon(self, run_locally):
        'String that show the status of all simulations graphically.'
        if run_locally:
            queued_jobs = []
        else:
            queued_jobs = get_queued_jobs()
        rib = ''
        for image in self.images_ordered:
            if image.propagated:
                rib += 'C'  # completed
            elif queued_jobs is None:
                rib += '?'
            elif image.submitted(queued_jobs):
                rib += 's'  # submitted
            else:
                rib += '.'  # not submitted, only defined
        return rib

    def write_yaml(self, backup=True, message=None):  # TODO: rename to save_status
        'Save the full status of the String to yaml file in directory $STRING_SIM_ROOT/strings/<branch>_<iteration>'
        # TODO: think of saving this to the commits folder in general...
        import shutil
        string = {}
        for key, image in self.images.items():
            assert key==image.seq
        string['images'] = [image.dump() for image in self.images_ordered]
        string['branch'] = self.branch
        string['iteration'] = self.iteration
        string['image_distance'] = self.image_distance
        config = {}
        config.update(self.opaque)
        config['strings'] = [string]
        if message is not None:
            config['message'] = message
        mkdir('%s/strings/%s_%03d/' % (root(), self.branch, self.iteration))
        fname_base = '%s/strings/%s_%03d/plan' % (root(), self.branch, self.iteration)
        if backup and os.path.exists(fname_base + '.yaml'):
            attempt = 0
            fname_backup = fname_base + '.bak'
            while os.path.exists(fname_backup):
                fname_backup = fname_base + '.bak%d' % attempt
                attempt += 1
            shutil.move(fname_base + '.yaml', fname_backup)
        with open(fname_base + '.yaml', 'w') as f:
            yaml.dump(config, f, width=1000)  # default_flow_style=False,

    def reparametrize(self, subdir='colvars'):
        'Created a copy of the String where the images are reparametrized. The resulting string in an unpropagated String.'

        fields = self.images_ordered[0].mean.dtype.names

        # collect all means and bring them into order ("direct" connection from 0 to len(self)-1)
        means = reorder_nodes(nodes=[image.colvars(subdir=subdir, fields=fields).mean for image in self.images_ordered])  #  TODO: fix me!

        # do the string reparametrization
        nodes = compute_equidistant_nodes_2(old_nodes=means, d=self.image_distance)  #  TODO: fix me!

        iteration = self.iteration + 1
        new_string = self.empty_copy(iteration=iteration, previous=self)

        for i_node, x in enumerate(nodes):
            # we have to convert the unrealized nodes to realized frames
            responses = []
            try:
                for im in self.images.values():
                    responses.append((im, im.colvars(subdir=subdir, fields=fields).closest_point(x)))
            except FileNotFoundError as e:
                warnings.warn(str(e))
            best_idx = np.argmin([r[1]['d'] for r in responses])
            best_image = responses[best_idx][0]
            best_step = responses[best_idx][1]['i']

            new_image = Image(image_id='%s_%03d_%03d_%03d' % (self.branch, iteration, i_node, 0),
                              previous_image_id=best_image.image_id, previous_frame_number=best_step,
                              node=x, spring=best_image.spring.copy(), endpoint=False, atoms_1=None)

            new_string.images[new_image.seq] = new_image

        return new_string

    def find(self, x):
        'Find the Image and a replica that is closest to point x in order parameter space.'
        # return image, replica_id (in that image)
        images =  [image for image in self.images.values() if not image.endpoint]
        results = [image.closest_replica(x) for image in images]
        best = int(np.argmin([r[1] for r in results]))
        return images[best], results[best][0], results[best][1]

    @classmethod
    def load(cls, branch='AZ', iteration=0):
        'Created a String object by recovering the information form the yaml file in the folder that is given as the argument.'
        folder = '%s/strings/%s_%03d' % (root(), branch, iteration)
        with open(folder + '/plan.yaml') as f:
            config = yaml.load(f)
        string = config['strings'][0]
        branch = string['branch']
        iteration = string['iteration']
        image_distance = string['image_distance']
        images_arr = [Image.load(config=img_cfg) for img_cfg in string['images']]
        images = {image.seq:image for image in images_arr}
        opaque = {key : config[key] for key in config.keys() if key!='strings'}
        return String(branch=branch, iteration=iteration, images=images, image_distance=image_distance, previous=None,
                      opaque=opaque)

    @property
    def previous_string(self):
        'Attempt to load previous iteration of the string'
        if self.previous is None:
            if self.iteration > 1:
                print('loading -1')
                self.previous = String.load(branch=self.branch, iteration=self.iteration - 1)
            else:
                print('loading *')  # TODO: this seems currently broken
                self.previous = String.from_scratch(branch=self.branch, iteration_id=0)  # TODO: find better solution
        return self.previous

    def overlap(self, subdir='colvars', fields=All, indicator='max', matrix=False, return_ids=False):
        'Compute the overlap (SVM) between images of the string'
        ids = []
        if not matrix:
            o = np.zeros(len(self.images_ordered) - 1) + np.nan
            for i, (a, b) in enumerate(zip(self.images_ordered[0:-1], self.images_ordered[1:])):
                try:
                    o[i] = a.overlap_plane(b, subdir=subdir, fields=fields, indicator=indicator)
                    ids.append((a.seq, b.seq))
                except FileNotFoundError as e:
                    warnings.warn(str(e))
            if return_ids:
                return o, ids
            else:
                return o
        else:
            o = np.zeros((len(self.images_ordered), len(self.images_ordered))) + np.nan
            for i, a in enumerate(self.images_ordered[0:-1]):
                o[i, i] = 0.
                for j, b in enumerate(self.images_ordered[i+1:]):
                    try:
                       o[i, i + j + 1] = a.overlap_plane(b, subdir=subdir, fields=fields, indicator=indicator)
                       o[i + j + 1, i] = o[i, i + j + 1]
                    except FileNotFoundError as e:
                       warnings.warn(str(e))
            o[-1, -1] = 0.
            return o

    def fel(self, subdir='colvars', T=303.15):
        'Compute an estimate of the free energy along the string, by running BAR for adjacent images'
        f = np.zeros(len(self.images) - 1) + np.nan
        for i, (a, b) in enumerate(zip(self.images_ordered[0:-1], self.images_ordered[1:])):
            try:
                f[i] = a.bar(b, subdir=subdir, T=T)
            except FileNotFoundError as e:
                warnings.warn(str(e))
        return f

    def arclength_projections(self, subdir='colvars', order=0, x0=False):
        support_points = [recscalar_to_vector(image.node[0]) for image in self.images_ordered]  # TODO: use node instead?
        # remove duplicate points https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        x = []
        for r in support_points:
            if tuple(r) not in x:
                x.append(tuple(r))
        support_points = np.array(x)

        fields = self.images_ordered[0].node.dtype.names
        results = []
        for image in self.images_ordered:  # TODO: return as dictionary instead
            try:
                if x0:
                    x = image.x0(subdir=subdir, fields=fields)
                    results.append(Colvars.arclength_projection([recscalar_to_vector(x)], support_points, order=order))
                else:
                    x = image.colvars(subdir=subdir, fields=fields)
                    results.append(Colvars.arclength_projection(x.as2D, support_points, order=order))
            except FileNotFoundError as e:
                warnings.warn(str(e))
        return results

    def mbar(self, subdir='colvars', T=303.15, disc_subdir='colvars', disc_fields=All, disc_centers=None):
        'Estimate all free energies using MBAR (when running with conventional order parameters, not RMSD)'
        import pyemma
        import pyemma.thermo

        RT = 1.985877534E-3 * T  # kcal/mol

        fields = self.images_ordered[0].node.dtype.names
        for image in self.images.values():
            if image.node.dtype.names != fields:
                raise RuntimeError('Images have varying node dimensions, cannot use this MBAR wrapper.')
            if image.spring.dtype.names != fields:
                raise RuntimeError('Images have varying spring dimensions, cannot use this MBAR wrapper.')

        # collect all possible biases
        unique_biases = []
        for im in self.images_ordered:
            bias = (im.node, im.spring)
            if bias not in unique_biases:
                unique_biases.append(bias)
        print('found', len(unique_biases), 'unique biases')

        btrajs = []
        ttrajs = []
        dtrajs = []
        K = len(unique_biases)
        for i_im, image in enumerate(self.images.values()):
            try:
                x = image.colvars(subdir=subdir, fields=fields, memoize=False)
                btraj = np.zeros((len(x), K))
                ttraj = np.zeros(len(x), dtype=int) + i_im
                for k, bias in enumerate(unique_biases):
                    node, spring = bias
                    btraj[:, k] = Image.potential(x=x, node=node, spring=spring) / RT
                btrajs.append(btraj)
                ttrajs.append(ttraj)

                if disc_centers is not None:
                    y = image.colvars(subdir=disc_subdir, fields=disc_fields, memoize=False).as2D
                    dtrajs.append(pyemma.coordinates.assign_to_centers(y, centers=disc_centers)[0])
                else:
                    dtrajs.append(np.zeros(len(x), dtype=int))
            except FileNotFoundError as e:
                warnings.warn(str(e))

        print('running MBAR')
        mbar = pyemma.thermo.MBAR(direct_space=True, maxiter=100000, maxerr=1e-13)
        mbar.estimate((ttrajs, dtrajs, btrajs))

        return mbar

    def mbar_RMSD(self, T=303.15, subdir='rmsd'):
        'For RMSD-type bias: Estimate all free energies using MBAR'
        import pyemma.thermo

        RT = 1.985877534E-3 * T  # kcal/mol

        # collect unique RMSDs and biases
        bias_def_to_simid = collections.defaultdict(list)
        rmsd_def_to_simid = collections.defaultdict(list)
        for im in self.images.values():
            bias_def = (im.previous_image_id, im.previous_frame_number, tuple(im.atoms_1), float(im.spring['RMSD'][0]))
            bias_def_to_simid[bias_def].append(im.image_id)
            rmsd_def = (im.previous_image_id, im.previous_frame_number, tuple(im.atoms_1))
            rmsd_def_to_simid[rmsd_def].append(im.image_id)
        K = len(bias_def_to_simid)  # number of unique biases
        unique_biases = []
        simid_to_bias_index = {}
        for ri, (bias_def, bias_simids) in enumerate(bias_def_to_simid.items()):
            rmsd_def = bias_def[0:-1]
            unique_biases.append(_Bias(ri=ri, spring=bias_def[-1], rmsd_simids=rmsd_def_to_simid[rmsd_def],
                                       bias_simids=bias_simids))
            for simid in bias_simids:
                simid_to_bias_index[simid] = ri

        btrajs = []
        ttrajs = []
        for im in self.images.values():
            x = im.colvars(subdir=subdir, memoize=False)
            btraj = np.zeros((len(x), K)) + np.nan
            biases_computed = set()

            for bias in unique_biases:
                found, simid = find(keys=bias.rmsd_simids, items=x._pcoords.dtype.names)
                if found:
                     running_bias_index = bias.ri
                     spring_constant = bias.spring
                     btraj[:, running_bias_index] = 0.5 * spring_constant * x[simid] ** 2 / RT
                     biases_computed.add(bias.ri)
                else:
                    warnings.warn('Trajectory %s is missing bias with simid %s.' % (im.image_id, bias.simids[0]))

            if len(biases_computed) > K:
                raise ValueError('Image %s has too many biases' % im.image_id)
            if len(biases_computed) < K:
                raise ValueError('Image %s is missing some biases' % im.image_id)
            assert not np.any(np.isnan(btraj))
            btrajs.append(btraj)
            ttrajs.append(np.zeros(len(x), dtype=int) + simid_to_bias_index[im.image_id])

        print('running MBAR')  # TODO: overlap option????
        mbar = pyemma.thermo.MBAR(direct_space=True, maxiter=100000, maxerr=1e-13)
        mbar.estimate((ttrajs, ttrajs, btrajs))

        # prepare "xaxis" to use for plots
        xaxis = [-1] * len(simid_to_bias_index)
        for id_, number in simid_to_bias_index.items():
            _, _, major, minor = id_.split('_')
            xaxis[number] = float(major + '.' + minor)

        return mbar, xaxis

    @deprecated
    def mbar_RMSD_old(self, T=303.15, subdir='rmsd'):
        'For RMSD-type bias: Estimate all free energies using MBAR'
        # TODO: also implement the simpler case with a "simple" (possibly multidimensional) order parameter
        # TODO: implement some way to provide mbar with am order parameter that is binned (discretized) -> dtrajs
        # e.g. # discretize='com_distance' (how to select the number of bins?)
        import collections
        import pyemma.thermo

        RT = 1.985877534E-3 * T  # kcal/mol

        # Convention for ensemble IDs: sarangi does not use any explict ensemble IDs.
        # Ensembles are simply defined by the bias parameters and are not given any canonical label.
        # This is the same procedure as in Pyemma (Ch. Wehmeyer's harmonic US API).
        # For data exchange, biases are simply labeled with the ID of a simulations that uses the bias.
        # This is e.g. used in precomputed bias energies / precomputed spring extensions.
        # Data exchange labels are not unique (there can't be any unique label that's not arbitrary).
        # So we have to go back to the actual bias definitions and map them to their possible IDs.
        parameters_to_names = collections.defaultdict(list)  # which bias IDs point to the same bias?
        for im in self.images.values():  # TODO: have a full universal convention for defining biases (e.g. type + params)
            bias_def = (im.previous_image_id, im.previous_frame_number, tuple(im.atoms_1), im.spring[0][0])  # convert back from numpy?
            parameters_to_names[bias_def].append(im.image_id)
        # TODO: what if the defintions span multiple string iterations or multiple branches?
        K = len(parameters_to_names)  # number of (unique) ensembles
        print('number of unique biases is', K)
        # generate running indices for all the different biases; generate map from bias IDs to running indices
        names_to_indices = {}  # index = running index
        names_to_spring_constants = {}  # index = running index
        for i, (bias_def, names) in enumerate(parameters_to_names.items()):
            for name in names:
                names_to_indices[name] = i
                names_to_spring_constants[name] = bias_def[-1]

        btrajs = []
        ttrajs = []
        for im in self.images.values():
            # print('loading', im.image_id)
            x = im.colvars(subdir=subdir, memoize=False)
            # print('done loading')
            btraj = np.zeros((len(x), K)) + np.nan  # shape??
            biases_defined = set()

            for name in x._pcoords.dtype.names:
                if name in names_to_indices:
                    # loop over all indices
                    btraj[:, names_to_indices[name]] = 0.5 * names_to_spring_constants[name] * x[name] ** 2 / RT
                    biases_defined.add(names_to_indices[name])
                else:
                    warnings.warn('Trajectory %s contains unused observable %s.' % (im.image_id, name))
            if len(biases_defined) > K:
                raise ValueError('Image %s has too many biases' % im.image_id)
            if len(biases_defined) < K:
                raise ValueError('Image %s is missing some biases' % im.image_id)
            assert not np.any(np.isnan(btraj))
            btrajs.append(btraj)
            ttrajs.append(np.zeros(len(x), dtype=int) + names_to_indices[im.image_id])

        print('running MBAR')
        mbar = pyemma.thermo.MBAR(direct_space=True, maxiter=100000, maxerr=1e-13)
        mbar.estimate((ttrajs, ttrajs, btrajs))

        # prepare "xaxis" to use for plots
        xaxis = [-1] * len(names_to_indices)
        for id_, number in names_to_indices.items():
            _, _, major, minor = id_.split('_')
            xaxis[number] = float(major + '.' + minor)

        return mbar, xaxis

    @staticmethod
    def overlap_gaps(matrix, threshold=0.99):
        'Indentify the major gaps in the (thermodynamic) overlap matrix'
        import msmtools
        n = matrix.shape[0]
        if matrix.shape[1] != n:
            raise ValueError('matrix must be square')
        di = np.diag_indices(n)
        matrix = matrix.copy()
        matrix[di] = 0
        c = matrix.sum(axis=1)
        matrix[di] = c
        T = matrix / (2 * c[:, np.newaxis])
        m = np.count_nonzero(msmtools.analysis.eigenvalues(T) > threshold)
        return msmtools.analysis.pcca(T, m)

    def displacement(self, subdir='colvars', fields=All, norm='rmsd', origin='node'):
        drift = {}
        for seq, im in self.images.items():
            try:
                drift[seq] = im.displacement(subdir=subdir, fields=fields, norm=norm, origin=origin)
            except FileNotFoundError as e:
                warnings.warn(str(e))
            #x = im.x0(subdir=subdir, fields=fields)
            #y = im.colvars(subdir=subdir, fields=fields).mean
            #if normalize and x.ndim == 2 and x.shape[1] == 3:
            #    n = x.shape[0]
            #else:
            #    n = 1.
            #drift[seq] = np.linalg.norm(x - y) / np.sqrt(n) #* 10
        return drift




def parse_commandline(argv=None):
    import argparse
    import sys
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', help='wait for job completion', default=False, action='store_true')
    parser.add_argument('--dry', help='dry run', default=False, action='store_true')
    parser.add_argument('--local', help='run locally (in this machine)', default=False, action='store_true')
    #parser.add_argument('--distance', help='distance between images', default=1.0)
    #parser.add_argument('--boot', help='bootstrap computation', default=False, action='store_true')
    args = parser.parse_args(argv)

    return {'wait':args.wait, 'run_locally':args.local, 'dry':args.dry}

def init(image_distance=1.0, argv=None):
    String.from_scratch(image_distance=image_distance).write_yaml()


def load(branch='AZ', offset=0):
    'Find the latest iteration of the string in $STRING_SIM_ROOT/strings/ and recover it from the yaml file.'
    folder = root() + '/strings/'
    iteration = -1
    for entry in os.listdir(folder):
        splinters =entry.split('_')
        if len(splinters)==2:
            folder_branch, folder_iteration = splinters 
            if folder_branch==branch and folder_iteration.isdigit():
                iteration = max([iteration, int(folder_iteration)])
    print('highest current iteration is', iteration)
    return String.load(branch=branch, iteration = iteration + offset)


def main(argv=None):
    options = parse_commandline(argv)

    #if args.boot:
    #    string = String.from_scratch()
    #else:
    string = load()
    print(string.branch, string.iteration, ':', string.ribbon(run_locally=options['run_locally']))

    # do at least one iteration
    if not string.propagated:
        string = string.propagate(wait=options['wait'], run_locally=options['run_locally'], dry=options['dry'])  # finish propagating
    #else:  # reparametrize and propagate at least once
    #    string = string.bisect_and_propagate_util_connected(run_locally=options['run_locally']).reparametrize().propagate(wait=options['wait'], run_locally=options['run_locally'])

    #if options['wait']:  # keep looping
    #    while True:
    #        print(string.iteration, ':', string.ribbon(run_locally=options['run_locally']))
    #        string = string.bisect_and_propagate_util_connected(run_locally=options['run_locally']).reparametrize().propagate(wait=options['wait'], run_locally=options['run_locally'])


if __name__ == '__main__':
    import sys
    sys.exit(main())

