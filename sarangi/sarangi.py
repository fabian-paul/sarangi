import numpy as np
import os
import glob
import warnings
import concurrent.futures
import subprocess
import yaml  # replace with json (more portable and future proof)
import tempfile
import shutil
import errno
import time
from .reparametrization import *


def parse_line(tokens):
    #print(tokens)
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


class _Universe(object):
    def __contains__(self, key):
        return True


def load_colvar(fname, selection=None):
    rows = []
    with open(fname) as f:
        for line in f:
            tokens = line.split()
            if tokens[0] == '#':
                var_names = tokens[1:]
            else:
                values, dims = parse_line(tokens)
                rows.append(values)
    assert len(var_names) == len(dims)
    assert len(rows[0]) == sum(dims)
    data = np.array(rows)

    if selection is None:
        selection = _Universe()
    elif isinstance(selection, str):
        selection = [selection]

    # convert to structured array
    dtype_def = []
    for name, dim in zip(var_names, dims):
        if name in selection:
            if dim == 1:
                dtype_def.append((name, np.float64, 1))
            else:
                dtype_def.append((name, np.float64, dim))
    dtype = np.dtype(dtype_def)

    indices = np.concatenate(([0], np.cumsum(dims)))
    # print(dtype)
    # print(indices)
    colgroups = [np.squeeze(data[:, start:stop]) for name, start, stop in zip(var_names, indices[0:-1], indices[1:]) if
                 name in selection]
    # for c,n in zip(colgroups, dtype.names):
    #    print(c.shape, n, dtype.fields[n][0].shape)
    return np.core.records.fromarrays(colgroups, dtype=dtype)


def mkdir(folder):
    try:
        os.mkdir(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_structured(config):
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
    config = {}
    for n in array.dtype.names:
        if len(array.dtype.fields[n][0].shape) == 1:  # vector type
            config[n] = [float(x) for x in array[n]]
        elif len(array.dtype.fields[n][0].shape) == 0:  # scalar type
            config[n] = float(array[n])
        else:
            raise RuntimeError('unsupported dimension')
    return config


def overlap_gaps(matrix, threshold=0.99):
    import msmtools
    c = matrix.sum(axis=1)
    T = matrix / c[:, np.newaxis]
    n = np.count_nonzero(msmtools.analysis.eigenvalues(T) > threshold)
    return msmtools.analysis.pcca(T, n)


class Pcoord(object):
    def __init__(self, folder, base, fields=None):
        # TODO: in principle pcoords can be recomputed from the full trajectories if they are missing
        # TODO: implement this (or steps toward this). How can the computation of observables be automatized in a good way?
        self._mean = None
        self._var = None
        self._cov = None
        fname = folder + '/' + base
        if os.path.exists(fname + '.colvars.trajs'):
            self._pcoords = load_colvar(fname + '.colvars.trajs', selection=fields)
        elif os.path.exists(fname + '.npy'):
            self._pcoords = np.load(fname + '.npy')
        else:
            raise RuntimeError('No progress coordinates / colvar file %s found.' % fname)

    def _compute_moments(self):
        if self._mean is None or self._var is None:
            pcoords = self._pcoords
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
        return self._pcoords[items]

    def __len__(self):
        return self._pcoords.shape[0]

    @property
    def as2D(self):
        if self._pcoords.dtype.names is None:
            n = self._pcoords.shape[0]
            return self._pcoords.reshape((n, -1))
        else:
            raise NotImplementedError('as2D not yet implemented for structures array')

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

    # TODO: implement overlap comutation here
    # usage Pcoord.overlap(im_1.pcoord('rmsd'), im_2.pcoord('rmsd'))
    # im_1.pcoord('xyz').cov() ...
    # maybe allow some memoization ?
    #

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


    def arclength_projection(self, nodes, order=0):
        # TODO: also implement a simpler version of this algorithm
        nodes =  np.array(nodes)
        if nodes.ndim == 2:
            axis = 1
        elif nodes.ndim == 3:
            axis = (1, 2)
        else:
            raise NotImplementedError('Nodes with ndim > 2 are not supported.')
        results = []
        for x in self._pcoords:
            i = np.argmin(np.linalg.norm(x[np.newaxis, ...]-nodes, axis=axis))
            if order == 0:
                results.append(i)
            elif order == 2:
                if i == 0 or i==len(nodes) - 1:
                    continue
                mid = nodes[i]
                plus = nodes[i + 1]
                minus = nodes[i - 1]
                v3 = plus - mid
                v3v3 = np.vdot(v3, v3)
                di = 1. #np.sign(i2 - i1)
                v1 = mid - x
                v2 = x - minus
                #v1v2 = np.sum(v1*v2)
                v1v3 = np.vdot(v1, v3)
                v1v1 = np.vdot(v1, v1)
                v2v2 = np.vdot(v2, v2)
                results.append(
                     i + (di*(v1v3**2 - v3v3*(v1v1 - v2v2))**0.5 - v1v3 - v3v3) / (2*v3v3))
        return results

    def closest_point(self, x):
        'Find the replica which is closest to x in order parameters space.'
        if self._pcoords.ndim == 2:
            axis = 1
        elif self._pcoords.ndim == 3:
            axis = (1, 2)
        else:
            raise NotImplementedError('Nodes with ndim > 2 are not supported.')
        dist = np.linalg.norm(self._pcoords - x[np.newaxis, ...], axis=axis)
        i = np.argmin(dist)
        return {'i':int(i), 'd':dist[i], 'x':self._pcoords[i]}

    #def arclength_projection(self, plus, minus):
    #    mid = self.mean
    #    v3 = plus - mid
    #    v3v3 = np.vdot(v3, v3)
    #    di = 1. #np.sign(i2 - i1)
    #    results = []
    #    for x in self._pcoords:
    #        v1 = mid - x
    #        v2 = x - minus
    #        #v1v2 = np.sum(v1*v2)
    #        v1v3 = np.vdot(v1, v3)
    #        v1v1 = np.vdot(v1, v1)
    #        v2v2 = np.vdot(v2, v2)
    #        results.append(
    #             (di*(v1v3**2 - v3v3*(v1v1 - v2v2))**0.5 - v1v3 - v3v3) / (2*v3v3))
    #    return results

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

    def copy(self, branch=None, iteration=None, major_id=None, minor_id=None, node=None,
             spring=None, previous_image_id=None, previous_frame_number=None, atoms_1=None):
        'Copy the Image object, allowing parameter changes'
        if iteration is None:
            iteration = self.iteration
        if branch is None:
            branch = self.branch
        if major_id is None:
            major_id = self.major_id
        if minor_id is None:
            minor_id = self.minor_id
        if node is None:
            node = self.node
        if spring is None:
            spring = self.spring
        if atoms_1 is None:
            atoms_1 = self.atoms_1
        image_id = '{branch}_{iteration:03d}_{major_id:03d}_{minor_id:03d}'.format(
                        branch = branch, iteration=iteration, major_id=major_id, minor_id=minor_id
                    )
        if previous_image_id is None:
            previous_image_id=self.previous_image_id
        if previous_frame_number is None:
            previous_frame_number=self.previous_frame_number,
        return Image(image_id=image_id, previous_image_id=previous_image_id, 
                     previous_frame_number=previous_frame_number,
                     node=node, spring=spring, endpoint=self.endpoint, atoms_1=atoms_1)

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
        with open(os.path.expandvars('$STRING_SIM_ROOT/setup/jobfile.template')) as f:
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
        return float(str(self.id_major) + '.' + str(self.id_minor))

    #def id_str(self, arclength):
    #    return '%3s_%03d_%03d_%03d' % (self.branch, self.iteration, self.id_major, self.id_minor)

    @property
    def job_name(self):
        return 'im_' + self.image_id

    @property
    def base(self):
        'Base path of the image. base+".dcd" are the replicas, base+".dat" are the order parameters'
        root = os.environ['STRING_SIM_ROOT']
        return '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
               root=root, branch=self.branch, iteration=self.iteration, id_minor=self.id_minor, id_major=self.id_major)

    @property
    def previous_base(self):
        root = os.environ['STRING_SIM_ROOT']
        branch, iteration, id_major, id_minor = self.previous_image_id.split('_')
        return '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
               root=root, branch=branch, iteration=int(iteration), id_minor=int(id_minor), id_major=int(id_major))

    def submitted(self, queued_jobs):
        return self.job_name in queued_jobs

    def _make_env(self, random_number):
        env = dict()
        root = os.environ['STRING_SIM_ROOT']
        env['STRING_SIM_ROOT'] = root
        env['STRING_ITERATION'] = str(self.iteration)
        env['STRING_IMAGE_ID'] = self.image_id
        env['STRING_PLAN'] = '{root}/strings/{branch}_{iteration:03d}/plan.yaml'.format(root=root, branch=self.branch, iteration=self.iteration)
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
        root = os.environ['STRING_SIM_ROOT']

        if self.endpoint:
            source = self.previous_base
            dest = self.base
            shutil.copy(source + '.dcd', dest + '.dcd')
            if os.path.exists(source + '.colvars.traj'):  # if order were already computed
                shutil.copy(source + '.colvars.traj', dest + '.colvars.traj')  # copy them
            #else:  # compute the order parameters
            #    default_env = dict(os.environ)
            #    default_env.update(env)
            #    subprocess.run('python $STRING_SIM_ROOT/string_scripts/pcoords.py $STRING_ARCHIVE.nc > $STRING_ARCHIVE.dat',
            #                   shell=True, env=default_env)  # TODO: abstract this as a method

        else:  # normal (intermediate string point)
            if run_locally:
                job_file = self._make_job_file(env)
                print('run', job_file, '(', self.job_name, ')')
                if not dry:
                    subprocess.run('bash ' + job_file, shell=True)  # directly execute the job file
            else:
                job_file = self._make_job_file(env)
                if wait:
                    command = 'qsub --wait ' + job_file  # TODO: slurm
                    print('run', command, '(', self.job_name, ')')
                    if not dry:
                        subprocess.run(command, shell=True)  # debug
                        # TODO: delete the job file
                else:
                    command = 'qsub ' + job_file  # TODO: slurm
                    print('run', command, '(', self.job_name, ')')
                    if not dry:
                        subprocess.run(command, shell=True)  # debug

        return self


    def closest_replica(self, x):  # TODO: rewrite!!!
        # TODO: offer option to search for a replica that is closest to a given plane
        'Find the replica which is closest to x in order parameters space.'
        assert self.propagated
        dist = np.linalg.norm(self.pcoords - x[np.newaxis, :], axis=1)
        i = np.argmin(dist)
        return int(i), dist[i]

    #def get_pcoords(self, selection=None):
    #    return load_colvar(self.base + '.colvars.traj', selection=selection)

    def pcoords(self, subdir='', fields=None, memoize=True):
        if subdir in self._pcoords:
            return self._pcoords[subdir]
        else:
            root = os.environ['STRING_SIM_ROOT']
            folder = '{root}/observables/{branch}_{iteration:03d}/'.format(
                   root=root, branch=self.branch, iteration=self.iteration)
            base = '{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
                   branch=self.branch, iteration=self.iteration, id_minor=self.id_minor, id_major=self.id_major)
            pcoords = Pcoord(folder=folder + subdir, base=base, fields=fields)
            if memoize:
                self._pcoords[subdir] = pcoords
            return pcoords
    #    # TODO: sometime the relevant coordinates are not in the colvar file put elsewhere (pcoord file?)
    #    # TODO: come up with way of recomputing colvars on request (TODO: think about the network)
    #    #assert self.propagated
    #    if self._pcoords is None:
    #        self._pcoords = load_colvar(self.base + '.colvars.traj')
    #    return self._pcoords

    def overlap_plane(self, other, subdir='', indicator='max'):
        return self.pcoords(subdir=subdir).overlap_plane(other.pcoords(subdir=subdir), indicator=indicator)

    def x0(self, subdir='', fields=None):
        if subdir not in self._x0:
            root = os.environ['STRING_SIM_ROOT']
            branch, iteration, id_major, id_minor = self.previous_image_id.split('_')
            folder = '{root}/observables/{branch}_{iteration:03d}/'.format(
                   root=root, branch=branch, iteration=int(iteration))
            base = '{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
                   branch=branch, iteration=int(iteration), id_minor=int(id_minor), id_major=int(id_major))
            self._x0[subdir] = Pcoord(folder=folder + subdir, base=base, fields=fields)[self.previous_frame_number]
        return self._x0[subdir]

    @staticmethod
    def u(x, c, k, T=303.15):

        pot = np.zeros(x.shape[0])
        for n in k.dtype.names:
            pot += 0.5*k[n]*(x[n] - c[n])**2  # TODO: handle the case when x is a vector # TODO: correct units!!!
        return pot

    @staticmethod
    def bar_1D(x_a, c_a, k_a, x_b, c_b, k_b, T=303.15):
        import pyemma
        RT = 1.985877534E-3*T
        btrajs = [np.zeros((x_a.shape[0], 2)), np.zeros((x_b.shape[0], 2))]
        btrajs[0][:, 0] = Image.u(x_a, c_a, k_a)/RT
        btrajs[0][:, 1] = Image.u(x_a, c_b, k_b)/RT
        btrajs[1][:, 0] = Image.u(x_b, c_a, k_a)/RT
        btrajs[1][:, 1] = Image.u(x_b, c_b, k_b)/RT
        ttrajs = [np.zeros(x_a.shape[0], dtype=int), np.ones(x_b.shape[0], dtype=int)]
        mbar = pyemma.thermo.MBAR()
        mbar.estimate((ttrajs, ttrajs, btrajs))
        return mbar.f_therm[0] - mbar.f_therm[1]
        #return mbar.free_energies

    def bar(self, other, T=303.15):
        import pyemma
        RT = 1.985877534E-3 * T  # kcal/mol
        id_self = '%03d_%03d' % (self.id_major, self.id_minor)
        id_other = '%03d_%03d' % (other.id_major, other.id_minor)
        p_self = self.pcoords(subdir='RMSD')
        p_other = other.pcoords(subdir='RMSD')
        btrajs = [np.zeros((len(p_self), 2)), np.zeros((len(p_other), 2))]
        btrajs[0][:, 0] = p_self[id_self][:]**2 * 5.0 / RT
        btrajs[0][:, 1] = p_self[id_other][:]**2 * 5.0 / RT
        btrajs[1][:, 0] = p_other[id_self][:]**2 * 5.0 / RT
        btrajs[1][:, 1] = p_other[id_other][:]**2 * 5.0 / RT
        ttrajs = [np.zeros(len(p_self), dtype=int), np.ones(len(p_other), dtype=int)]
        mbar = pyemma.thermo.MBAR()
        mbar.estimate((ttrajs, ttrajs, btrajs))
        return mbar.f_therm[0] - mbar.f_therm[1]


    def fel(self, other, method='bar', T=303.15):
        assert method == 'bar'
        #if not self.propagated:
        #    raise RuntimeError('String not completely propagated. Can\'t compute free energies')
        # TODO: concatenate all repeats
        fields = self.spring.dtype.names
        # assert all(fields == im_2.spring.dtype.names)  # FIXME
        return Image.bar_1D(x_a=self.get_pcoords(selection=fields), c_a=self.node, k_a=self.spring,
                            x_b=other.get_pcoords(selection=fields), c_b=other.node,  k_b=other.spring, T=T)


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
    #with os.popen('squeue -o %j -h -u ' + user) as f:  #TODO qstat -u 
    #    jobs = f.readlines()
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
            return None  #_Universe()

class String(object):
    def __init__(self, branch, iteration, images, image_distance, previous):
        self.branch = branch
        self.iteration = iteration
        self.images = images
        self.image_distance = image_distance
        self.previous = previous

    def empty_copy(self, iteration=None, images=None, previous=None):
        'Creates a copy of the current String object. Image array is left empty.'
        if iteration is None:
            iteration = self.iteration
        if previous is None:
            previous = self.previous
        if images is None:
            images = dict()
        return String(branch=self.branch, iteration=iteration, images=images,
                      image_distance=self.image_distance, previous=previous)

    def __len__(self):
        return len(self.images)

    @property
    def images_ordered(self):
        return [self.images[key] for key in sorted(self.images.keys())]

    @classmethod
    def from_scratch(cls, image_distance=1.0, branch='AZ', iteration_id=1):
        'Initialized a String from a folder of *.nc files.'
        n_images = len(glob.glob(os.path.expandvars('$STRING_SIM_ROOT/strings/AZ_000/*.dcd')))
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
        '''Propagated the String (one iteration). Returns a modified copy.'''
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

        mkdir(os.path.expandvars('$STRING_SIM_ROOT/strings/%s_%03d/' % (self.branch, self.iteration)))

        max_workers=1#l

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._launch_simulation, image, np.random.randint(0, high=np.iinfo(np.int64).max, size=1, dtype=np.int64)[0], wait, queued_jobs, run_locally, dry)
                       for image in self.images_ordered]
            for future in concurrent.futures.as_completed(futures):
                image = future.result()
                propagated_images[image.image_id] = image

        return self.empty_copy(images=propagated_images)

    def connected(self, threshold=0.1):
        if not self.propagated:
            raise RuntimeError('Trying to find connectedness of string that has not been (fully) propagated. Giving up.')
        return all(p.overlap_plane(q) >= threshold for p, q in zip(self.images_ordered[0:-1], self.images_ordered[1:]))

    def bisect_at(self, i, j=None, subdir='', where='mean', search='string'):
        # TODO: think more about the parameters ...
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
            x = (p.pcoords(subdir=subdir).mean + q.pcoords(subdir=subdir).mean) * 0.5
        elif where == 'x0':
            x = (p.x0(subdir=subdir) + q.x0(subdir=subdir)) * 0.5
        elif where == 'plane':
            raise NotImplementedError('bisection at SVM plane not implemented yet')
        else:
            raise ValueError('Unrecognized value "%s" for option "where"' % where)

        if search == 'points':
            query_p = p.pcoords(subdir=subdir).closest_point(x)
            query_q = q.pcoords(subdir=subdir).closest_point(x)
            if query_p['d'] < query_q['d']:
                print('best distance is', query_p['d'])
                best_image = p
                best_step = query_p['i']
            else:
                print('best distance is', query_q['d'])
                best_image = q
                best_step = query_q['i']
        elif search == 'string':
            responses = [(im, im.pcoords(subdir=subdir).closest_point(x)) for im in self.images.values()]
            best_idx = np.argmin([r[1]['d'] for r in responses])
            best_image = responses[best_idx][0]
            best_step = responses[best_idx][1]['i']
            print('best distance is', responses[best_idx][1]['d'])
        else:
            raise ValueError('Unrecognized value "%s" of parameter "search"' % search)

        new_seq = (p.seq + q.seq) * 0.5
        new_major = int(new_seq)
        new_minor = int((new_seq - new_major)*1000)
        new_image_id = '%s_%03d_%03d_%03d' %(best_image.branch, best_image.iteration, new_major, new_minor)

        new_image = Image(image_id=new_image_id, previous_image_id=best_image.image_id, previous_frame_number=best_step,
                          node=best_image.node, spring=best_image.spring, endpoint=False, atoms_1=list(best_image.atoms_1))

        #   self.images[new_image.seq] = new_image
        return new_image
        # TODO: insert
        #new_image = Image(image_id=new_image_id,
        #                  previous_iteration_id=best_image.iteration_id,
        #                  previous_image_id=best_image.image_id, previous_replica_id=best_step,
        #                  node=x, endpoint=False)
        #
        #best = int(np.argmin([r[1] for r in results]))
        #return images[best], results[best][0], results[best][1]

    def bisect(self, subdir=''):
        raise NotImplementedError('This is broken')

        ov_max, idx = self.overlap(subdir=subdir, indicator='max', return_ids=True)
        gaps = np.array(idx)[ov_max < 0.10]
        new_imgs = []
        new_string = self.empty_copy()  # images=string.images  # TODO???
        for gap in gaps:
            new_img = string.bisect_at(i=gap[0], j=gap[1], subdir='cartesian', where='x0')
            new_imgs.append(new_img)
            new_string.images[new_img.seq] = new_img
        return new_string  # new_string.write_yaml()

        if not self.propagated:
            raise RuntimeError('Trying to bisect string that has not been (fully) propagated. Giving up.')
        # TODO
        new_string = self.empty_copy()  # we must make a new string which can later be propagated

        for p, q in zip(self.images_ordered[0:-1], self.images_ordered[1:]):
            new_string.images[p.image_id] = p  # copy p

            overlap = p.overlap_plane(q, subdir=subdir)
            print('overlap', overlap)
            if overlap < 0.1:
                print('besecting')
                x = (p.x0 + q.x0) / 2
                # TODO: use clf.support_vectors_
                # TODO: find the actual frame ... how? Answer: the point closest to the plane TODO
                # TODO: search in old string and not in current
                prev_image, prev_replica_id, prev_dist = self.previous_string.find(x)  # TODO: search in current and past strings!
                curr_image, curr_replica_id, curr_dist = self.find(x)
                if prev_dist < curr_dist:
                    old_image, old_replica_id = prev_image, prev_replica_id
                else:
                    old_image, old_replica_id = curr_image, curr_replica_id
                #assert old_image.iteration_id == self.iteration - 1, old_image.iteration_id
                new_image_id = (p.image_id + q.image_id) / 2  # go for the average now, fixme?
                new_image = Image(iteration_id=self.iteration, image_id=new_image_id,
                                  previous_iteration_id=old_image.iteration_id,
                                  previous_image_id=old_image.image_id, previous_replica_id=old_replica_id,
                                  node=x, endpoint=False)
                print(new_image.node)
                new_string.images[new_image_id] = new_image

        new_string.images[max(self.images)] = self.images[max(self.images)]  # do not change numbering

        print('done with string bisection')
        new_string.write_yaml(backup=True)

        return new_string

    def bisect_and_propagate_util_connected(self, run_locally=False):
        s = self
        while not s.connected():
            s = s.bisect().propagate(wait=True, run_locally=run_locally)
        return s

    @property
    def propagated(self):
        return all(image.propagated for image in self.images.values())

    def ribbon(self, run_locally):
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

    def write_yaml(self, backup=True):  # TODO: rename to save_status
        'Save the full status of the String to yaml file in directory $STRING_SIM_ROOT/#iteration'
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
        config['strings'] = [string]
        mkdir(os.path.expandvars('$STRING_SIM_ROOT/strings/%s_%03d/' % (self.branch, self.iteration)))
        fname_base = os.path.expandvars('$STRING_SIM_ROOT/strings/%s_%03d/plan' % (self.branch, self.iteration))
        if backup and os.path.exists(fname_base + '.yaml'):
            attempt = 0
            fname_backup = fname_base + '.bak'
            while os.path.exists(fname_backup):
                fname_backup = fname_base + '.bak%d' % attempt
                attempt += 1
            shutil.move(fname_base + '.yaml', fname_backup)
        with open(fname_base + '.yaml', 'w') as f:
            yaml.dump(config, f, width=1000)  # default_flow_style=False,

    def reparametrize(self, freeze=None):
        'Created a copy of the String where the images are reparametrized. The resulting string in an unpropagated String. Call .propagate to launch the simulations.'
        # this routine is inspired by the design of wepy
        assert self.propagated

        mus = reorder_nodes(nodes=[image.mean for image in self.images_ordered])

        # 'nodes' are hypothetical points in conformation space that do not need to be realized
        nodes = compute_equidistant_nodes_2(old_nodes=mus, d=self.image_distance)  # in the future we could let the arclength vary with node sigma
        #print('newly computed nodes', nodes)

        new_string = self.empty_copy(iteration=self.iteration + 1, previous=self)

        start = self.images[0]
        new_string.images[0] = start.copy(branch=self.branch, iteration=self.iteration + 1, node=start.pcoords[0, :], major_id=0, minor_id=0)  # add endpoint

        for i_node, x in enumerate(nodes):
            # we have to convert the unrealized nodes to realized frames
            old_image, old_replica_id, _ = self.find(x)
            new_image = Image(branch=self.branch, iteration=self.iteration + 1, id_major=i_node + 1, id_minor=0, # new_string already contains one image
                              previous_image_id=old_image.image_id, previous_frame_number=old_replica_id,
                              node=x, endpoint=False)
            new_string.images[i_node + 1] = new_image

        end = self.images[max(self.images)]
        n_end = len(new_string.images)
        new_string.images[n_end] = \
            end.copy(branch=self.branch, iteration=self.iteration + 1, node=end.pcoords[0, :], major_id=n_end, minor_id=0)  # add endpoint

        assert new_string.images[0].endpoint and new_string.images[max(new_string.images)].endpoint

        #print('B base', new_string.images[-1].base)

        #print('reparametrized string')  # debug

        new_string.write_yaml()

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
        folder = os.path.expandvars('$STRING_SIM_ROOT/strings/%s_%03d' % (branch, iteration))
        with open(folder + '/plan.yaml') as f:
            config = yaml.load(f)['strings'][0]
        branch = config['branch']
        iteration = config['iteration']
        image_distance = config['image_distance']
        images_arr = [Image.load(config=img_cfg) for img_cfg in config['images']]
        images = {image.seq:image for image in images_arr}
        return String(branch=branch, iteration=iteration, images=images, image_distance=image_distance, previous=None)

    @property
    def previous_string(self):
        if self.previous is None:
            if self.iteration > 1:
                print('loading -1')
                self.previous = String.load(branch=self.branch, iteration=self.iteration - 1)
            else:
                print('loading *')
                self.previous = String.from_scratch(branch=self.branch, iteration_id=0)  # TODO: find better solution
        return self.previous

    def overlap(self, subdir='', indicator='max', matrix=False, return_ids=False):
        if not matrix:
            o = np.zeros(len(self.images_ordered) - 1) + np.nan
            for i, (a, b) in enumerate(zip(self.images_ordered[0:-1], self.images_ordered[1:])):
                try:
                    o[i] = a.overlap_plane(b, subdir=subdir, indicator=indicator)
                except Exception as e:
                    warnings.warn(str(e))
            if return_ids:
                return o, [(p.seq, q.seq) for p, q in zip(self.images_ordered[0:-1], self.images_ordered[1:])]
            else:
                return o
        else:
            o = np.zeros((len(self.images_ordered), len(self.images_ordered))) + np.nan
            for i, a in enumerate(self.images_ordered[0:-1]):
                o[i, i] = 1.
                for j, b in enumerate(self.images_ordered[i+1:]):
                    try:
                       o[i, i + j + 1] = a.overlap_plane(b, subdir=subdir, indicator=indicator)
                       o[i + j + 1, i] = o[i, i + j + 1]
                    except Exception as e:
                       warnings.warn(str(e))
            o[-1, -1] = 1.
            return o

    def fel(self, T=303.15):
        f = np.zeros(len(self.images) - 1) + np.nan
        for i, (a, b) in enumerate(zip(self.images_ordered[0:-1], self.images_ordered[1:])):
            #if a.propagated and b.propagated:
            try:
                f[i] = a.fel(b, T=T)
            except Exception as e:
                warnings.warn(str(e))
        return f

        # TODO: implement simple overlap ... add selection of order parameters!

    #def arclength_projections(self, subdir=''):
    #    io = self.images_ordered
    #    results = []
    #    for a, b, c, i in zip(io[0:-2], io[1:-1], io[2:], np.arange(len(io) - 2)):
    #        # TODO: could also use center? But this would be confusing here
    #        plus = a.pcoords(subdir=subdir).mean
    #        minus = b.pcoords(subdir=subdir).mean
    #        x = i + b.pcoords(subdir=subdir).projection(plus, minus)
    #        results.append(x)
    #    return results

    def arclength_projections(self, subdir='', order=0):
        x0s = [image.x0(subdir=subdir) for image in self.images_ordered]
        results = []
        for image in self.images_ordered:
            try:
                x = image.pcoords(subdir=subdir)
                results.append(x.arclength_projection(x0s, order=order))
            except Exception as e:
                warnings.warn(str(e))
        return results

    def mbar(self, T=303.15, k_half=1., subdir='rmsd'):
        # TODO: also implement the simpler case with a simple order parameter
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
            bias_identifier = (im.previous_image_id, im.previous_frame_number, tuple(im.atoms_1))  # convert back from numpy?
            parameters_to_names[bias_identifier].append(im.image_id)
        # TODO: what if the defintions span multiple string iterations or multiple branches?
        K = len(parameters_to_names)  # number of (unique) ensembles
        # generate running indices for all the different biases; generate map from bias IDs to running indices
        names_to_indices = {}
        for i, names in enumerate(parameters_to_names.values()):
            for name in names:
                names_to_indices[name] = i

        btrajs = []
        ttrajs = []
        for im in self.images.values():
            # print('loading', im.image_id)
            x = im.pcoords(subdir=subdir, memoize=False)
            # print('done loading')
            btraj = np.zeros((len(x), K)) + np.nan  # shape??
            biases_defined = set()

            for name in x._pcoords.dtype.names:
                if name in names_to_indices:
                    btraj[:, names_to_indices[name]] = k_half * x[name] ** 2 / RT
                    biases_defined.add(names_to_indices[name])
                else:
                    warnings.warn('field %s in pcoord does not correspond to any ensemble' % name)
            if len(biases_defined) != K:
                raise ValueError('Image %s is missing some biased / has too many biases' % im.image_id)
            assert not np.any(np.isnan(btraj))
            btrajs.append(btraj)
            ttrajs.append(np.zeros(len(x), dtype=int) + names_to_indices[im.image_id])

        print('running MBAR')  # TODO: overlap option????
        mbar = pyemma.thermo.MBAR(direct_space=True, maxiter=100000, maxerr=1e-13)
        mbar.estimate((ttrajs, ttrajs, btrajs))

        # prepare "xaxis" to use for plots
        xaxis = [-1] * len(names_to_indices)
        for id_, number in names_to_indices.items():
            _, _, major, minor = id_.split('_')
            xaxis[number] = float(major + '.' + minor)

        return mbar, xaxis


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
    folder = os.path.expandvars('$STRING_SIM_ROOT/strings/')
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

