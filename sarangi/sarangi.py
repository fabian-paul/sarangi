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
            dtype_def.append((name, np.float64, len(v)))
        elif isinstance(value, float):
            dtype_def.append((name, np.float64))
        else:
            raise RuntimeError('unrecognized type')
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


class Image(object):
    def __init__(self, image_id, previous_image_id, previous_frame_number,
                 node, spring, endpoint):
        self.image_id = image_id
        self.previous_image_id = previous_image_id
        self.previous_frame_number = previous_frame_number
        self.node = node
        self.spring = spring
        self.endpoint = endpoint
        self._pcoords = None
        self._mean = None
        self._cov = None
        self._x0 = None

    def copy(self, branch=None, iteration=None, major_id=None, minor_id=None, node=None,
             spring=None, previous_image_id=None, previous_frame_number=None):
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
        image_id = '{branch}_{iteration:03d}_{major_id:02d}_{minor_id:02d}'.format(
                        branch = branch, iteration=iteration, major_id=major_id, minor_id=minor_id
                    )
        if previous_image_id is None:
            previous_image_id=self.previous_image_id
        if previous_frame_number is None:
            previous_frame_number=self.previous_frame_number,
        return Image(image_id=image_id, previous_image_id=previous_image_id, 
                     previous_frame_number=previous_frame_number,
                     node=node, spring=spring, endpoint=self.endpoint)

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
        if 'endpoint' in config:
            endpoint = config['endpoint']
        else:
            endpoint = False
        return Image(image_id=image_id, previous_image_id=previous_image_id,
                     previous_frame_number=previous_frame_number,
                     node=node, spring=spring, endpoint=endpoint)

    def dump(self):
        'Dump state of object to dictionary. Called by String.dump'
        if self.node is not None:
            node = dump_structured(self.node)
        else:
            node = None  # TODO: omit?
        if self.spring is not None:
            spring = dump_structured(self.spring)
        else:
            spring = None  # TODO: omit?
        config = {'image_id': self.image_id, 'prev_image_id': self.previous_image_id, 
                  'prev_frame_number': self.previous_frame_number,
                  'node': node, 'spring': spring, 'endpoint': self.endpoint}
        return config

    @property
    def propagated(self):
        return os.path.exists(self.base + '.dcd') and os.path.exists(self.base + '.colvars.traj')

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
    #    return '%3s_%03d_%02d_%02d' % (self.branch, self.iteration, self.id_major, self.id_minor)

    @property
    def job_name(self):
        return 'im_' + self.image_id

    @property
    def base(self):
        'Base path of the image. base+".dcd" are the replicas, base+".dat" are the order parameters'
        root = os.environ['STRING_SIM_ROOT']
        return '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{id_major:02d}_{id_minor:02d}'.format(
               root=root, branch=self.branch, iteration=self.iteration, id_minor=self.id_minor, id_major=self.id_major)

    @property
    def previous_base(self):
        root = os.environ['STRING_SIM_ROOT']
        branch, iteration, id_major, id_minor = self.previous_image_id.split('_')
        return '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{id_major:02d}_{id_minor:02d}'.format(
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
        return env

    def propagate(self, random_number, wait, queued_jobs, run_locally=False):
        'Generic propagation command. Submits jobs for the intermediate points. Copies the end points.'
        if self.propagated:
            #print(self.job_name, 'already completed')
            return self

        #  if the job is already queued, return or wait and return then
        if self.job_name in queued_jobs:
            print('skipping submission of', self.job_name, 'because alrealy queued')
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
            if os.path.exists(source + '.colvar.traj'):  # if order were already computed
                shutil.copy(source + '.colvar.traj', dest + '.colvar.traj')  # copy them
            #else:  # compute the order parameters
            #    default_env = dict(os.environ)
            #    default_env.update(env)
            #    subprocess.run('python $STRING_SIM_ROOT/string_scripts/pcoords.py $STRING_ARCHIVE.nc > $STRING_ARCHIVE.dat',
            #                   shell=True, env=default_env)  # TODO: abstract this as a method

        else:  # normal (intermediate string point)
            if run_locally:
                job_file = self._make_job_file(env)
                print('would run', job_file, '(', self.job_name, ')')
                subprocess.run('bash ' + job_file, shell=True)  # directly execute the job file
                exit(0)
            else:
                job_file = self._make_job_file(env)
                if wait:
                    command = 'qsub --wait ' + job_file
                    print('would run', command, '(', self.job_name, ')')
                    #subprocess.run(command, shell=True)  # debug
                    # TODO: delete the job file
                else:
                    command = 'qsub ' + job_file
                    print('would run', command, '(', self.job_name, ')')
                    #subprocess.run(command, shell=True)  # debug

        return self


    def closest_replica(self, x):  # TODO: rewrite!!!
        # TODO: offer option to search for a replica that is closest to a given plane
        'Find the replica which is closest to x in order parameters space.'
        assert self.propagated
        dist = np.linalg.norm(self.pcoords - x[np.newaxis, :], axis=1)
        i = np.argmin(dist)
        return int(i), dist[i]

    def get_pcoords(self, selection=None):
        return load_colvar(self.base + '.colvars.traj', selection=selection)

    @property
    def pcoords(self):
        assert self.propagated
        if self._pcoords is None:
            var_names, self._pcoords = load_colvar(self.base + '.colvars.traj')
        return self._pcoords

    @property
    def mean(self):
        self._compute_moments()
        return self._mean

    @property
    def cov(self):
        self._compute_moments()
        return self._cov

    def _compute_moments(self):
        if self._mean is None or self._cov is None:
            pcoords = self.pcoords
            mean = np.mean(pcoords, axis=0)
            mean_free = pcoords - mean[np.newaxis, :]
            cov = np.dot(mean_free.T, mean_free) / pcoords.shape[0]
            self._mean = mean
            self._cov = cov

    def overlap_Bhattacharyya(self, other):
        # https://en.wikipedia.org/wiki/Bhattacharyya_distance
        if self.endpoint or other.endpoint:
            return 0
        half_log_det_s1 = np.sum(np.log(np.diag(np.linalg.cholesky(self.cov))))
        half_log_det_s2 = np.sum(np.log(np.diag(np.linalg.cholesky(other.cov))))
        s = 0.5*(self.cov + other.cov)
        half_log_det_s = np.sum(np.log(np.diag(np.linalg.cholesky(s))))
        delta = self.mean - other.mean
        return 0.125*np.dot(delta, np.dot(np.linalg.inv(s), delta)) + half_log_det_s - 0.5*half_log_det_s1 - 0.5*half_log_det_s2
 
    def overlap_plane(self, other):
        import sklearn.svm
        clf = sklearn.svm.LinearSVC()
        X = np.vstack((self.pcoords, other.pcoords))
        n_self = self.pcoords.shape[0]
        n_other = other.pcoords.shape[0]
        labels = np.zeros(n_self + n_other, dtype=int)
        labels[n_self:] = 1
        clf.fit(X, labels)
        c = np.zeros((2, 2), dtype=int)
        p_self = clf.predict(self.pcoords)
        p_other = clf.predict(other.pcoords)
        c[0, 0] = np.count_nonzero(p_self == 0)
        c[0, 1] = np.count_nonzero(p_self == 1)
        c[1, 0] = np.count_nonzero(p_other == 0)
        c[1, 1] = np.count_nonzero(p_other == 1)
        c_sum = c.sum(axis=1)
        c_norm = c / c_sum[:, np.newaxis]
        return max(c_norm[0, 1], c_norm[1, 0])

    @property
    def x0(self):
        if self._x0 is None:
            root = os.environ['STRING_SIM_ROOT']
            path = self.previous_base + '.colvars.traj'
            self._x0 = load_colvar(np.loadtxt(path))[self.previous_frame_number, :]
        return self._x0

    @staticmethod
    def u(x, c, k, T=303.15):

        pot = np.zeros(x.shape[0])
        for n in k.dtype.names:
            pot += 0.5*k[n]*(x[n] - c[n])**2  # TODO: handle the case when x is a vector # TODO: correct units!!!
        return pot

    @staticmethod
    def bar(x_a, c_a, k_a, x_b, c_b, k_b, T=303.15):
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

    def fel(self, other, method='bar', T=303.15):
        assert method == 'bar'
        #if not self.propagated:
        #    raise RuntimeError('String not completely propagated. Can\'t compute free energies')
        # TODO: concatenate all repeats
        fields = self.spring.dtype.names
        # assert all(fields == im_2.spring.dtype.names)  # FIXME
        return Image.bar(x_a=self.get_pcoords(selection=fields), c_a=self.node, k_a=self.spring,
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

    def _launch_simulation(self, image, random_number, wait, queued_jobs, run_locally):
        return image.propagate(random_number=random_number, wait=wait,
                               queued_jobs=queued_jobs, run_locally=run_locally)

    def propagate(self, wait=False, run_locally=False):
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
            futures = [executor.submit(self._launch_simulation, image, np.random.randint(0, high=np.iinfo(np.int64).max, size=1, dtype=np.int64)[0], wait, queued_jobs, run_locally)
                       for image in self.images_ordered]
            for future in concurrent.futures.as_completed(futures):
                image = future.result()
                propagated_images[image.image_id] = image

        return self.empty_copy(images=propagated_images)

    def connected(self, threshold=0.1):
        if not self.propagated:
            raise RuntimeError('Trying to find connectedness of string that has not been (fully) propagated. Giving up.')
        return all(p.overlap_plane(q) >= threshold for p, q in zip(self.images_ordered[0:-1], self.images_ordered[1:]))

    def bisect(self):
        if not self.propagated:
            raise RuntimeError('Trying to bisect string that has not been (fully) propagated. Giving up.')
        # TODO
        new_string = self.empty_copy()  # we must make a new string which can later be propagated

        for p, q in zip(self.images_ordered[0:-1], self.images_ordered[1:]):
            new_string.images[p.image_id] = p  # copy p

            overlap = p.overlap_plane(q)
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
                rib += 'c'  # completed
            elif queued_jobs is None:
                rib += '?'
            elif image.submitted(queued_jobs):
                rib += 's'  # submitted
            else:
                rib += '.'  # not submitted, only defined
        return rib

    def write_yaml(self, backup=False):  # TODO: rename to save_status
        'Save the full status of the String to yaml file in directory $STRING_SIM_ROOT/#iteration'
        import shutil
        config = {}
        for key, image in self.images.items():
            assert key==image.image_id
        config['images'] = [image.dump() for image in self.images_ordered]
        config['branch'] = self.branch
        config['iteration'] = self.iteration
        config['image_distance'] = self.image_distance
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
            yaml.dump([config], f, default_flow_style=False)

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

    def overlap(self):
        return [p.overlap_plane(q) for p,q in zip(self.images_ordered[0:-1], self.images_ordered[1:])]


    def fel(self, T=303.15):
        f = np.zeros(len(self.images) - 1) + np.nan
        for i, (a, b) in enumerate(zip(self.images_ordered[0:-1], self.images_ordered[1:])):
            #if a.propagated and b.propagated:
            try:
                f[i] = a.fel(b, T=T)
            except Exception as e:
                pass
        return f

        # TODO: implement simple overlap ... add selection of order parameters!


def parse_commandline(argv=None):
    import argparse
    import sys
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', help='wait for job completion', default=False, action='store_true')
    parser.add_argument('--local', help='run locally (in this machine)', default=False, action='store_true')
    #parser.add_argument('--distance', help='distance between images', default=1.0)
    #parser.add_argument('--boot', help='bootstrap computation', default=False, action='store_true')
    args = parser.parse_args(argv)

    return {'wait':args.wait, 'run_locally':args.local}

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
        string = string.propagate(wait=options['wait'], run_locally=options['run_locally'])  # finish propagating
    else:  # reparametrize and propagate at least once
        string = string.bisect_and_propagate_util_connected(run_locally=options['run_locally']).reparametrize().propagate(wait=options['wait'], run_locally=options['run_locally'])

    if options['wait']:  # keep looping
        while True:
            print(string.iteration, ':', string.ribbon(run_locally=options['run_locally']))
            string = string.bisect_and_propagate_util_connected(run_locally=options['run_locally']).reparametrize().propagate(wait=options['wait'], run_locally=options['run_locally'])


if __name__ == '__main__':
    import sys
    sys.exit(main())

