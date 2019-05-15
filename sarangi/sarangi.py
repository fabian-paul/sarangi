import numpy as np
import os
import warnings
import collections
import concurrent.futures
import subprocess
import yaml
import tempfile
from .util import *
from .reparametrization import reorder_nodes, compute_equidistant_nodes_2
from .colvars import Colvars
from .image import Image, load_image, interpolate_id
from .queuing import *


# TODO: better handling of "all" fields: this is currently in really bad shape
# TODO: find a better way to handle the "step" column in NAMD covlar files
# TODO: offer some rsyc script to copy data
# TODO: write function that returns the angles between displacements


__all__ = ['String', 'root', 'load', 'main']
__author__ = 'Fabian Paul <fab@physik.tu-berlin.de>'


_Bias = collections.namedtuple('_Bias', ['ri', 'spring', 'rmsd_simids', 'bias_simids'], verbose=False)


class Group(object):
    'Group of images, that typically belong to the same replica exchange simulation'

    def __init__(self, group_id, string, images=None):
        import weakref
        self.group_id = group_id
        if images is None:
            self.images = dict()
        else:
            self.images = images
            for im in images.values():
                if im.group_id != group_id:
                    raise ValueError('Trying to add image from group %s to group %s.' % (im.group_id, self.group_id))
        self.string = weakref.proxy(string)  # avoid cyclic dependency between string and group

    def _make_env(self, random_number):
        env = dict()
        root_ = root()
        env['STRING_SIM_ROOT'] = root_
        env['STRING_GROUP_ID'] = self.group_id
        env['STRING_BRANCH'] = self.string.branch
        env['STRING_ITERATION'] = int(self.string.iteration)
        env['STRING_IMAGE_IDS'] = '"' + ' '.join([self.images[k].image_id for k in sorted(self.images.keys())]) + '"'
        env['STRING_PLAN'] = '{root}/strings/{branch}_{iteration:03d}/plan.yaml'.format(root=root_,
                                                                                        branch=self.string.branch,
                                                                                        iteration=self.string.iteration)
        env['STRING_RANDOM'] = str(random_number)
        env['STRING_ARCHIVIST'] = os.path.dirname(__file__) + '/string_archive.py'
        env['STRING_SARANGI_SCRIPTS'] = os.path.dirname(__file__) + '/../scripts'
        env['STRING_BASE'] = '{root}/strings/{branch}_{iteration:03d}'.format(root=root_,
                                                                              branch=self.string.branch,
                                                                              iteration=self.string.iteration)
        env['STRING_OBSERVABLES_BASE'] = '{root}/observables/{branch}_{iteration:03d}'.format(root=root_,
                                                                                              branch=self.string.branch,
                                                                                              iteration=self.string.iteration)
        return env

    def __getitem__(self, key):
        if isinstance(key, float) or isinstance(key, int):
            return self.images[float(key)]
        elif isinstance(key, str):
            return next(im for im in self.images.values() if im.image_id == key)
        else:
            raise ValueError('key is neither a number nor a string, don\'t know what to do with it.')

    @property
    def job_name(self):
        'Job name under which the simulation is known to the queuing system'
        return 're_%s_%03d_%s' % (self.string.branch, self.string.iteration, self.group_id)

    def _make_job_file(self, env, cpus_per_replica=32):
        'Created a submission script for the job on the local file system.'
        with open('%s/setup/re_jobfile.template' % root()) as f:
            template = ''.join(f.readlines())
            environment = '\n'.join(['export %s=%s' % (k, v) for k, v in env.items()])
        num_replicas = len(self.images)
        with tempfile.NamedTemporaryFile(suffix='.sh', delete=False) as f:
            f.write(template.format(job_name=self.job_name, environment=environment, num_replicas=num_replicas,
                                    num_cpus=num_replicas*cpus_per_replica).encode(encoding='UTF-8'))
            job_file_name = f.name
        return job_file_name

    @property
    def propagated(self):
        'MD simulation was completed?'
        return any(im.propagated for im in self.images.values())

    def propagate(self, random_number, queued_jobs, dry=False, cpus_per_replica=32):
        'Run or submit to queuing system the propagation script'
        if self.propagated:
            return self

        #  if the job is already queued, return or wait and return then
        if self.job_name in queued_jobs:
            print('skipping submission of', self.job_name, 'because already queued')
            return

        env = self._make_env(random_number=random_number)

        job_file = self._make_job_file(env, cpus_per_replica=cpus_per_replica)
        command = 'qsub ' + job_file  # TODO: slurm (sbatch)
        print('run', command, '(', self.job_name, ')')
        if not dry:
            subprocess.run(command, shell=True)

    def add_image(self, image):
        if image.seq in self.images:
            warnings.warn('Image with seq %f is already in group %s. Overwriting.' % (image.seq, self.group_id))
        self.images[image.seq] = image

    @property
    def images_ordered(self):
        'Images ordered by ID, where id_major.id_minor is interpreted as a floating point number'
        return [self.images[key] for key in sorted(self.images.keys())]

    def _load_permutations(self):
        'Load permutations and return as 2-D numpy array. replica[t, i_bias]'
        base = '{root}/observables/{branch}_{iteration:03d}/replica/'.format(root=root(), branch=self.string.branch,
                                                                             iteration=self.string.iteration)
        replica = [np.loadtxt(base + im.image_id + '.sort.history')[:, 1].astype(int) for im in self.images.values()]
        replica = np.vstack(replica).T
        return replica

    @property
    def hamilonian(self):
        'Permutations as 2-D numpy array, hamiltonian[t, i_replica] = index of bias. Ideal for plotting (together with hamiltonian_labels).'
        replica = self._load_permutations()
        return np.argsort(replica, axis=1)

    @property
    def hamiltonian_labels(self):
        'Y-axes labels to use togehter with self.hamiltonian'
        # The MD script must follow the same convention of mapping image ids to integers as sarangi.
        # We assume that replicas are by numerically increasing images id and are numbered without any gaps.
        # E.g. Assume the the images in the group have IDs 001_000, 001_123 and 001_312, the assignment will be
        # as follows: 001_000 -> 0, 001_123 -> 1, 001_312 -> 2.
        # This is automatically guaranteed, if sarangi's archivist is used to set up and post-process the simulations.
        images_ordered = [self.images[key] for key in sorted(self.images.keys())]
        return [im.image_id for im in images_ordered]

    @property
    def exchange_frequency(self):
        'Compute the count matrix of successful exchanges between Hamiltonian.'
        replica = self._load_permutations()
        n = len(self.images)
        counts = np.zeros((n, n), dtype=int)
        for a, b in zip(replica[0:-1, :], replica[1:, :]):
            for i, j in zip(a, b):
                counts[i, j] += 1
        return counts


class String(object):
    def __init__(self, branch, iteration, images, image_distance, previous, colvars_def, opaque):
        self.branch = branch
        self.iteration = iteration
        self.images = images
        self.image_distance = image_distance
        self.previous = previous
        self.colvars_def = colvars_def
        self.opaque = opaque
        self.groups = dict()
        # create and populate groups, if group_ids were found in the config
        for im in images.values():
            if im.group_id is not None:
                if im.group_id not in self.groups:
                    self.groups[im.group_id] = Group(group_id=im.group_id, string=self)
                self.groups[im.group_id].add_image(im)

    def __str__(self):
        str_images = '{' + ', '.join(['%g:%s'%(seq, im) for seq, im in self.images.items()]) + '}'
        return 'String(branch=\'%s\', iteration=%d, images=%s, image_distance=%f, previous=%s, colvars_def=%s, opaque=%s)' % (
            self.branch, self.iteration, str_images, self.image_distance, self.previous, self.colvars_def, self.opaque)

    def __getitem__(self, key):
        if isinstance(key, float) or isinstance(key, int):
            return self.images[float(key)]
        elif isinstance(key, str):
            return next(im for im in self.images.values() if im.image_id == key)
        else:
            raise ValueError('key is neither a number nor a string, don\'t know what to do with it.')

    def discretize(self, points, states_per_arc=100):
        # TODO first check compatibility with path (fields)
        from .util import pairing
        arcs = [self]  # currently we only support one arc, TODO: change this
        sz = np.array([b.project(points, return_z=True) for b in arcs])
        best = np.argmin(sz[:, 1])  # find the closest arc for each input point
        i = np.fromiter((arcs[best_t].i for best_t in best), dtype=int)
        j = np.fromiter((arcs[best_t].j for best_t in best), dtype=int)
        return ((pairing(i, j, ordered=False) + sz[best, 0]) * states_per_arc).astype(int)

    def empty_copy(self, iteration=None, images=None, previous=None):
        'Creates a copy of the current String object. Image array is left empty.'
        if iteration is None:
            iteration = self.iteration
        if previous is None:
            previous = self.previous
        if images is None:
            images = dict()
        return String(branch=self.branch, iteration=iteration, images=images, image_distance=self.image_distance,
                      previous=previous, colvars_def=self.colvars_def, opaque=self.opaque)

    def add_image(self, image):
        'Add new image to string'
        if image.seq in self.images:
            raise ValueError('String already contains an image with sequence id %f, aborting operation.' % image.seq)
        else:
            self.images[image.seq] = image

    def __len__(self):
        'Number of images'
        return len(self.images)

    @property
    def images_ordered(self):
        'Images ordered by ID, where id_major.id_minor is interpreted as a floating point number'
        return [self.images[key] for key in sorted(self.images.keys())]

    def _launch_simulation(self, image, random_number, wait, queued_jobs, run_locally, dry):
        return image.propagate(random_number=random_number, wait=wait,
                               queued_jobs=queued_jobs, run_locally=run_locally, dry=dry)

    def propagate(self, wait=False, run_locally=False, dry=False, max_workers=1, cpus_per_replica=32):
        'Propagated the String (one iteration). Returns a modified copy.'
        if self.propagated:
            return self

        if run_locally:
            queued_jobs = []
        else:
            queued_jobs = get_queued_jobs()
            if queued_jobs is None:
                raise RuntimeError('queued jobs is undefined, I don\'t know which jobs are currently running. Giving up propagation.')

        propagated_images = dict()

        mkdir('%s/strings/%s_%03d/' % (root(), self.branch, self.iteration))

        # max_workers = len(self.images)

        # propagate individual images
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for image in self.images_ordered:
                # TODO: think about different ways of organizing single replica and multi-replica execution ...
                if image.group_id is None:  # do not submit multi-replica simulations here
                    random_number = np.random.randint(0, high=np.iinfo(np.int64).max, size=1, dtype=np.int64)[0]
                    futures.append(executor.submit(self._launch_simulation, image, random_number, wait, queued_jobs, run_locally, dry))
            for future in concurrent.futures.as_completed(futures):
                image = future.result()
                propagated_images[image.seq] = image

        # propagate replica exchange groups
        # parallel execution is not needed here ... or is it?
        for group in self.groups.values():
            random_number = np.random.randint(0, high=np.iinfo(np.int64).max, size=1, dtype=np.int64)[0]
            group.propagate(random_number=random_number, queued_jobs=queued_jobs, dry=dry, cpus_per_replica=cpus_per_replica)
            propagated_images.update(group.images)

        return self.empty_copy(images=propagated_images)  # FIXME: why is this needed?

    def connected(self, threshold=0.1):
        'Test if all images are overlapping'
        # TODO: replace by implementation based on the overlap matrix and its eigenvalues / eigenvectors / PCCA states
        if not self.propagated:
            raise RuntimeError('Trying to find connectedness of string that has not been (fully) propagated. Giving up.')
        return all(p.overlap_plane(q) >= threshold for p, q in zip(self.images_ordered[0:-1], self.images_ordered[1:]))

    def bisect_at(self, i, j=None, subdir='colvars', where='node', search='string', fields=All):
        r'''Create new image half-way between images i and j.

        example
        -------
            ov, ids = string.overlap(return_ids=True)
            for pair, o in zip(ids, ov):
                if o < 0.05:
                    new_image = string.bisect_at(i=pair[0], j=pair[1])
                    string.add_image(new_image)
            string.write_yaml(message='bisected at positions of low overlap')

        returns
        -------
        A new image. Image is not inserted into the string.

        notes
        -----
        The group id of the new images is set to None.
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
        elif where == 'node' or where == 'center':
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
            # TODO: move code to .find() ?
            responses = []
            for im in self.images.values():
                try:
                    responses.append((im, im.colvars(subdir=subdir, fields=fields).closest_point(x)))
                except FileNotFoundError as e:
                    warnings.warn(str(e))
            best_idx = np.argmin([r[1]['d'] for r in responses])
            best_image = responses[best_idx][0]
            best_step = responses[best_idx][1]['i']
            print('best distance is', responses[best_idx][1]['d'], '@', responses[best_idx][0].image_id)
        else:
            raise ValueError('Unrecognized value "%s" of parameter "search"' % search)

        new_image_id = interpolate_id(p.image_id, q.image_id, excluded=[im.image_id for im in self.images.values()])
        if any(new_image_id == im.image_id for im in self.images.values()):
            raise RuntimeError('Bisection produced new image id which is not unique. This should not happen.')

        new_image = best_image.__class__(image_id=new_image_id, previous_image_id=best_image.image_id,
                                         previous_frame_number=best_step, node=x, spring=p.spring.copy(), group_id=None)
        # TODO: for linear bias, where to put the terminal (just the next node along the string)
        # TODO: or should we use some interpolation to control the distance between node and termimal?

        #   self.images[new_image.seq] = new_image
        return new_image

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
            elif image.group_id is None and (image.job_name in queued_jobs):
                rib += 's'  # submitted
            elif image.group_id is not None and (self.groups[image.group_id].job_name in queued_jobs):
                rib += 'S'  # submitted
            else:
                rib += '.'  # not submitted, only defined
        return rib

    @property
    def base(self):
        return '{root}/strings/{branch}_{iteration:03d}'.format(root=root(), branch=self.branch, iteration=self.iteration)

    def check_against_string_on_disk(self):
        'Compare currently loaded string to the version on disk. Check that read-only elements were not modified'
        try:
            on_disk = String.load(branch=self.branch, iteration=self.iteration)
        except FileNotFoundError:
            # plan file does not exist, we therefore assume that this is a new string (or new iteration)
            return True
        # image values written to disk are not allowed to be changed in memory
        for key, image in on_disk.images.items():
            if self.images[key] != image:
                warnings.warn('Images that have already been written to disk have changed in RAM. '
                              'Unless you know exactly what you are doing, the current string cannot be saved.')
                return False
        return True

    def write_yaml(self, backup=True, message=None, _override=False):  # TODO: rename to save_status
        'Save the full status of the String to yaml file in directory $STRING_SIM_ROOT/strings/<branch>_<iteration>'
        import shutil
        if not self.check_against_string_on_disk() and not _override:
            raise RuntimeError('Not saved.')
        string = {}
        for key, image in self.images.items():
            assert key==image.seq
        string['images'] = [image.dump() for image in self.images_ordered]
        string['branch'] = self.branch
        string['iteration'] = self.iteration
        string['image_distance'] = self.image_distance
        string['colvars'] = self.colvars_def
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

    def reparametrize(self, subdir='colvars', fields=All, rmsd=True, linear_bias=False):
        'Created a copy of the String where the images are reparametrized. The resulting string in an unpropagated String.'

        # collect all means, in the same time check that all the coordinate dimensions and
        # coordinate names are the same across the string
        colvars_0 = self.images_ordered[0].colvars(subdir=subdir, fields=fields)
        real_fields = colvars_0.fields
        dims = colvars_0.dims
        current_means = []
        for image in self.images_ordered:
            colvars = image.colvars(subdir=subdir, fields=fields)
            if colvars.fields != real_fields or colvars.dims != dims:
                raise RuntimeError('colvars fields / dimensions are inconsistent across the string')
            # The geometry functions in the reparametrization module work with 2-D numpy arrays, while the colvar
            # class used recarrays and (1, n) shaped ndarrays. We therefore convert to plain numpy and strip extra dimensions.
            current_means.append(structured_to_flat(colvars.mean, fields=real_fields)[0, :])

        if rmsd:
            n_atoms = len(real_fields)
        else:
            n_atoms = 1

        # do the string reparametrization
        ordered_means = reorder_nodes(nodes=current_means)  # in case the string "coiled up", we reorder its nodes
        nodes = compute_equidistant_nodes_2(old_nodes=ordered_means, d=self.image_distance * n_atoms**0.5,
                                            d_skip=self.image_distance * n_atoms**0.5 / 2)

        # do some self-consistency tests
        eps = 1E-6
        # check distances
        for i, (a, b) in enumerate(zip(nodes[0:-1], nodes[1:])):
            delta = np.linalg.norm(a - b)
            if delta > self.image_distance * n_atoms ** 0.5 + eps \
                    or (delta < self.image_distance * n_atoms ** 0.5 - eps and i != len(nodes) - 2):
                warnings.warn('Reparametrization failed at new nodes %d and %d (distance %f)' % (i, i + 1, delta))
        # check order
        for i in range(len(nodes) - 1):
            if np.argmin(np.linalg.norm(nodes[i, np.newaxis, :] - nodes[i + 1:, :], axis=1)) != 0:
                warnings.warn('Reparametrization did not yield an ordered string.')
        # end of self-consistency test

        iteration = self.iteration + 1
        new_string = self.empty_copy(iteration=iteration, previous=self)
        image_class = self.images_ordered[0].__class__

        for i_node, x in enumerate(nodes):
            # we have to convert the unrealized nodes (predicted point of conformational space) to realized frames
            node = flat_to_structured(x[np.newaxis, :], fields=real_fields, dims=dims)
            responses = []
            for im in self.images.values():
                try:
                    responses.append((im, im.colvars(subdir=subdir, fields=fields).closest_point(node)))
                except FileNotFoundError as e:
                    warnings.warn(str(e))
            best_idx = np.argmin([r[1]['d'] for r in responses])
            best_image = responses[best_idx][0]
            best_step = responses[best_idx][1]['i']

            new_image = image_class(image_id='%s_%03d_%03d_%03d' % (self.branch, iteration, i_node, 0),
                                    previous_image_id=best_image.image_id, previous_frame_number=best_step,
                                    node=node, terminal=None, spring=best_image.spring.copy(),
                                    group_id=best_image.group_id)

            new_string.images[new_image.seq] = new_image

            if linear_bias:
                for a, b in zip(new_string.images_ordered[0:-1], new_string.images_ordered[1:]):
                    a.set_terminal_point(b.node)

        return new_string

    #@deprecated
    #def find(self, x):
    #    'Find the Image and a replica that is closest to point x in order parameter space.'
    #    # return image, replica_id (in that image)
    #    images =  [image for image in self.images.values() if not image.endpoint]
    #    results = [image.closest_replica(x) for image in images]
    #    best = int(np.argmin([r[1] for r in results]))
    #    return images[best], results[best][0], results[best][1]

    @classmethod
    def load(cls, branch='AZ', iteration=0):
        'Create a String object by recovering from the yaml file corresponding to the given branch and interation number (in the current project directory tree).'
        fname = '%s/strings/%s_%03d/plan.yaml' % (root(), branch, iteration)
        string = cls.load_form_fname(fname)
        if int(string.iteration) != iteration:
            raise RuntimeError(
                'Plan file is inconsitent: iteration recorded in the file is %s but the iteration encoded in the folder name is %d.' % (
                    string.iteration, iteration))
        return string

    @classmethod
    def load_form_fname(cls, fname):
        'Create a String object by recovering the information form the yaml file whose path is given as the argument.'
        with open(fname) as f:
            config = yaml.load(f)
        string = config['strings'][0]
        colvars_def = string['colvars'] if 'colvars' in string else None
        branch = string['branch']
        iteration = string['iteration']
        image_distance = string['image_distance']
        images_arr = [load_image(config=img_cfg, colvars_def=colvars_def) for img_cfg in string['images']]
        images = {image.seq: image for image in images_arr}
        opaque = {key: config[key] for key in config.keys() if key not in ['strings']}  # TODO: currently opaque refers to things outside of the string, also handle information inside
        return String(branch=branch, iteration=iteration, images=images, image_distance=image_distance, previous=None,
                      opaque=opaque, colvars_def=colvars_def)

    @property
    def previous_string(self):
        'Attempt to load previous iteration of the string'
        if self.previous is None:
            if self.iteration > 1:
                print('loading -1')
                self.previous = String.load(branch=self.branch, iteration=self.iteration - 1)
            else:
                print('loading *')  # TODO: this is currently broken and will not work like this
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

    def overlap_by_atom(self, subdir='colvars'):
        fields =list(self.images_ordered[0].node.dtype.names)
        overlap = np.zeros((len(self) - 1, len(fields))) + np.nan
        images_ordered = self.images_ordered
        for i_im, (a, b) in enumerate(zip(images_ordered[0:-1], images_ordered[1:])):
            try:
                for i_field, field in enumerate(fields):
                    overlap[i_im, i_field] = a.overlap_plane(b, subdir=subdir, fields=field)
            except FileNotFoundError as e:
                warnings.warn(str(e))
        return overlap

    def fel(self, subdir='colvars', T=303.15):
        'Compute an estimate of the free energy along the string, by running BAR for adjacent images'
        f = np.zeros(len(self.images) - 1) + np.nan
        mbar = [None] * (len(self.images) - 1)
        deltaf = [None] * (len(self.images) - 1)
        for i, (a, b) in enumerate(zip(self.images_ordered[0:-1], self.images_ordered[1:])):
            try:
                f[i], deltaf[i], mbar[i] = a.bar(b, subdir=subdir, T=T)
            except FileNotFoundError as e:
                warnings.warn(str(e))
        return -np.cumsum(f), deltaf, mbar

    def arclength_projections(self, subdir='colvars', order=1, x0=False):
        r'''For all frames in all simulations, compute the arc length order parameter.

        Parameters
        ----------
        order: int
            Interpolation order for the computation of arc length. Can take the values
            0, 1, or 2.
        subdir: str
            If working with non-default string setup, set this to the folder name for
            the collective variable trajectories.

        Notes
        -----
        This function is ideal for a simplified 1-D visualization of ensemble overlap.

        arc = string.arclength_projections()
        plt.figure(figsize=(15, 5))
        for group in arc:
            plt.hist(group)

        For order=2, the arc length is computed as detailed in the following publication

        :   Grisell Díaz Leines and Bernd Ensing. Path finding on high-dimensional free energy landscapes.
            Phys. Rev. Lett., 109:020601, 2012
        '''
        fields = self.images_ordered[0].fields
        support_points = [structured_to_flat(image.node, fields=fields)[0, :] for image in self.images_ordered]
        # remove duplicate points https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        x = []
        for r in support_points:
            if tuple(r) not in x:
                x.append(tuple(r))
        support_points = np.array(x)

        results = []
        for image in self.images_ordered:  # TODO: return as dictionary instead?
            try:
                if x0:
                    x = image.x0(subdir=subdir, fields=fields)
                    results.append(Colvars.arclength_projection([structured_to_flat(x, fields=fields)], support_points, order=order))
                else:
                    x = image.colvars(subdir=subdir, fields=fields)
                    results.append(Colvars.arclength_projection(x.as2D(fields=fields), support_points, order=order))
            except FileNotFoundError as e:
                warnings.warn(str(e))
        return results

    def mean_forces(self, subdir='colvars', integrate=False):
        r'''Returns the (P)MF in units of [unit of the spring constant] / [unit of length]^2

        Parameters
        ----------
        integrate: boolean
            If set to true, multiply mean forces with the image distance and compute cumulative sums.
            This is first order integration of the force along the string.

        Note
        ----
        Forces are always projected on the string.
        '''
        # RT = 1.985877534E-3 * T  # kcal/mol
        forces = []
        fields = list(self.images_ordered[0].node.dtype.names)
        support_points = [structured_to_flat(image.node, fields=fields)[0, :] for image in self.images_ordered]
        for image in self.images_ordered:
            for f in fields:
                if image.spring[f] != image.spring[fields[0]]:
                    raise NotImplementedError('PMF computation currently only implemented for isotropic forces')
            try:
                mean = image.colvars(subdir=subdir, fields=fields)
                node_proj = Colvars.arclength_projection(structured_to_flat(image.node, fields=fields), support_points, order=2)[0]
                #print(node_proj)
                mean_proj = Colvars.arclength_projection(mean.as2D(fields=fields), support_points, order=2)[0]
                #print(type(mean_proj), type(node_proj), image.spring[fields[0]][0])
                forces.append((node_proj - mean_proj)*image.spring[fields[0]][0])  # TODO: implement the general anisotropic case (TODO: move part to the Image classes)
            except FileNotFoundError as e:
                forces.append(0.)
                warnings.warn(str(e))
        if integrate:
            return np.cumsum(np.array(forces) * self.image_distance)
        else:
            return forces

    def mbar(self, subdir='colvars', T=303.15, disc_subdir='colvars', disc_fields=All, disc_centers=None):
        'Estimate all free energies using MBAR (when running with conventional order parameters, not RMSD)'
        import pyemma
        import pyemma.thermo

        RT = 1.985877534E-3 * T  # kcal/mol

        fields = list(self.images_ordered[0].node.dtype.names)
        for image in self.images.values():
            if list(image.node.dtype.names) != fields:
                raise RuntimeError('Images have varying node dimensions, cannot use this MBAR wrapper.')
            if list(image.spring.dtype.names) != fields:
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
                    y = image.colvars(subdir=disc_subdir, fields=disc_fields, memoize=False).as2D(fields=disc_fields)
                    dtrajs.append(pyemma.coordinates.assign_to_centers(y, centers=disc_centers)[0])
                else:
                    dtrajs.append(np.zeros(len(x), dtype=int))
            except FileNotFoundError as e:
                warnings.warn(str(e))

        print('running MBAR')
        mbar = pyemma.thermo.MBAR(direct_space=True, maxiter=100000, maxerr=1e-13)
        mbar.estimate((ttrajs, dtrajs, btrajs))

        return mbar

    def mbar_RMSD(self, T=303.15, subdir='rmsd'):  # TODO: move some of this into the Image classes
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
                found, simid = find(keys=bias.rmsd_simids, items=list(x._pcoords.dtype.names))
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

    # @deprecated
    # def mbar_RMSD_old(self, T=303.15, subdir='rmsd'):
    #     'For RMSD-type bias: Estimate all free energies using MBAR'
    #     # TODO: also implement the simpler case with a "simple" (possibly multidimensional) order parameter
    #     # TODO: implement some way to provide mbar with am order parameter that is binned (discretized) -> dtrajs
    #     # e.g. # discretize='com_distance' (how to select the number of bins?)
    #     import collections
    #     import pyemma.thermo
    #
    #     RT = 1.985877534E-3 * T  # kcal/mol
    #
    #     # Convention for ensemble IDs: sarangi does not use any explict ensemble IDs.
    #     # Ensembles are simply defined by the bias parameters and are not given any canonical label.
    #     # This is the same procedure as in Pyemma (Ch. Wehmeyer's harmonic US API).
    #     # For data exchange, biases are simply labeled with the ID of a simulations that uses the bias.
    #     # This is e.g. used in precomputed bias energies / precomputed spring extensions.
    #     # Data exchange labels are not unique (there can't be any unique label that's not arbitrary).
    #     # So we have to go back to the actual bias definitions and map them to their possible IDs.
    #     parameters_to_names = collections.defaultdict(list)  # which bias IDs point to the same bias?
    #     for im in self.images.values():  # TODO: have a full universal convention for defining biases (e.g. type + params)
    #         bias_def = (im.previous_image_id, im.previous_frame_number, tuple(im.atoms_1), im.spring[0][0])  # convert back from numpy?
    #         parameters_to_names[bias_def].append(im.image_id)
    #     # TODO: what if the defintions span multiple string iterations or multiple branches?
    #     K = len(parameters_to_names)  # number of (unique) ensembles
    #     print('number of unique biases is', K)
    #     # generate running indices for all the different biases; generate map from bias IDs to running indices
    #     names_to_indices = {}  # index = running index
    #     names_to_spring_constants = {}  # index = running index
    #     for i, (bias_def, names) in enumerate(parameters_to_names.items()):
    #         for name in names:
    #             names_to_indices[name] = i
    #             names_to_spring_constants[name] = bias_def[-1]
    #
    #     btrajs = []
    #     ttrajs = []
    #     for im in self.images.values():
    #         # print('loading', im.image_id)
    #         x = im.colvars(subdir=subdir, memoize=False)
    #         # print('done loading')
    #         btraj = np.zeros((len(x), K)) + np.nan  # shape??
    #         biases_defined = set()
    #
    #         for name in x._pcoords.dtype.names:
    #             if name in names_to_indices:
    #                 # loop over all indices
    #                 btraj[:, names_to_indices[name]] = 0.5 * names_to_spring_constants[name] * x[name] ** 2 / RT
    #                 biases_defined.add(names_to_indices[name])
    #             else:
    #                 warnings.warn('Trajectory %s contains unused observable %s.' % (im.image_id, name))
    #         if len(biases_defined) > K:
    #             raise ValueError('Image %s has too many biases' % im.image_id)
    #         if len(biases_defined) < K:
    #             raise ValueError('Image %s is missing some biases' % im.image_id)
    #         assert not np.any(np.isnan(btraj))
    #         btrajs.append(btraj)
    #         ttrajs.append(np.zeros(len(x), dtype=int) + names_to_indices[im.image_id])
    #
    #     print('running MBAR')
    #     mbar = pyemma.thermo.MBAR(direct_space=True, maxiter=100000, maxerr=1e-13)
    #     mbar.estimate((ttrajs, ttrajs, btrajs))
    #
    #     # prepare "xaxis" to use for plots
    #     xaxis = [-1] * len(names_to_indices)
    #     for id_, number in names_to_indices.items():
    #         _, _, major, minor = id_.split('_')
    #         xaxis[number] = float(major + '.' + minor)
    #
    #     return mbar, xaxis

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

    def filter(self, regex):
        'Return a new string that only contains the subset of images whose ID matched regex.'
        import re
        pattern = re.compile(regex)
        filtered_string = self.empty_copy()
        for im in self.images.values():
            if pattern.search(im.image_id):
                filtered_string.add_image(im)
        return filtered_string

    def group_images(self, new_group_id, images):
        'Assign set of images a new group id.'
        # first find unique a new group id (that has not been used before)
        current_group_ids = []
        for im in self.images.values():
            if im.group_id is not None and im.group_id not in current_group_ids:
                current_group_ids.append(im.group_id)
        if new_group_id in current_group_ids:
            raise ValueError('Group id is already used.')
        group = Group(new_group_id, self)
        for im in images:
            if im.propagated:
                warnings.warn('Image %s was already propagated. Skipping this image.' % im)
            if im.group_id is not None:
                warnings.warn('Image %s already has a group id (%s). Skipping this image.' % (im, im.group_id))
            else:
                im.group_id = new_group_id
                group.add_image(im)
        self.groups[new_group_id] = group


def parse_commandline(argv=None):
    import argparse
    import sys
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wait', help='wait for job completion', default=False, action='store_true')
    parser.add_argument('--dry', help='dry run', default=False, action='store_true')
    parser.add_argument('--local', help='run locally (in this machine)', default=False, action='store_true')
    parser.add_argument('--re', help='run replica exchange simulations', default=False, action='store_true')
    parser.add_argument('--cpus_per_replica', help='number of CPU per replica (only for multi-replica jobs with NAMD)', default=32)
    parser.add_argument('--iteration', help='do not propagate current string but a past iteration', default=None)
    parser.add_argument('--branch', help='select branch', default='AZ')
    #parser.add_argument('--distance', help='distance between images', default=1.0)
    #parser.add_argument('--boot', help='bootstrap computation', default=False, action='store_true')
    args = parser.parse_args(argv)

    options=  {'wait': args.wait, 'run_locally': args.local, 'dry': args.dry, 're': args.re,
               'cpus_per_replica': args.cpus_per_replica, 'branch': args.branch}
    if args.iteration is not None:
        options['iteration'] = int(args.iteration)
    else:
        options['iteration'] = None
    return options


#def init(image_distance=1.0, argv=None):
#    String.from_scratch(image_distance=image_distance).write_yaml()


def load(branch='AZ', offset=0):
    'Find the latest iteration of the string in $STRING_SIM_ROOT/strings/ and recover it from the yaml file.'
    folder = root() + '/strings/'
    iteration = -1
    for entry in os.listdir(folder):
        splinters = entry.split('_')
        if len(splinters) == 2:
            folder_branch, folder_iteration = splinters 
            if folder_branch == branch and folder_iteration.isdigit():
                iteration = max([iteration, int(folder_iteration)])
    print('Highest current iteration is %d. Loading iteration %d' % (iteration, iteration + offset))
    return String.load(branch=branch, iteration=iteration + offset)


def main(argv=None):
    options = parse_commandline(argv)

    if options['iteration'] is None:
        string = load(branch=options['branch'])
    else:
        string = String.load(branch=options['branch'], iteration=options['iteration'])
    print(string.branch, string.iteration, ':', string.ribbon(run_locally=options['run_locally']))

    if not string.propagated:
        string = string.propagate(wait=options['wait'], run_locally=options['run_locally'], dry=options['dry'],
                                  cpus_per_replica=options['cpus_per_replica'])
    #else:  # reparametrize and propagate at least once
    #    string = string.bisect_and_propagate_util_connected(run_locally=options['run_locally']).reparametrize().propagate(wait=options['wait'], run_locally=options['run_locally'])

    #if options['wait']:  # keep looping
    #    while True:
    #        print(string.iteration, ':', string.ribbon(run_locally=options['run_locally']))
    #        string = string.bisect_and_propagate_util_connected(run_locally=options['run_locally']).reparametrize().propagate(wait=options['wait'], run_locally=options['run_locally'])


if __name__ == '__main__':
    import sys
    sys.exit(main())

