import numpy as np
import os
import warnings
import collections
import concurrent.futures
import subprocess
import yaml
import tempfile
from tqdm import tqdm
from .util import *
from .reparametrization import reorder_nodes, compute_equidistant_nodes_2
from .colvars import Colvars
from .image import Image, load_image, interpolate_id
from .queuing import *


# TODO: better handling of "all" fields: this is currently in bad shape
# TODO: find a better way to handle the "step" column in NAMD covlar files
# TODO: offer some rsyc script to copy data


__all__ = ['String', 'root', 'load', 'main']
__author__ = 'Fabian Paul <fapa@uchicago.edu>'


_Bias = collections.namedtuple('_Bias', ['ri', 'spring', 'rmsd_simids', 'bias_simids'])


def find_realization_in_string(strings, node, subdir='colvars', ignore_missing=True):
    r'''In a string or in multiple strings, find actual frame that is close to the given point.

    Parameters
    ----------
    strings: String or iterable of Strings
        strings to search
    node: numpy structured array
        The point, given in structured numpy array format.
    subdir: str
        Name of the colvars subdirectory. Must be compatible with fields of given node.
    ignore_missing: bool
        Ignore missing colvar files.

    Returns
    -------
    image: Image
        the optimal Image object
    step: int
        the optimal frame index (discrete time step) in the MD data of the optimal index
    dist: float
    '''
    if isinstance(strings, String):
        strings = [strings]

    responses = []
    for string in strings:
        for im in string.images.values():
            try:
                responses.append((im, im.colvars(subdir=subdir, fields=list(node.dtype.names)).closest_point(node)))
            except FileNotFoundError as e:
                if ignore_missing:
                    warnings.warn(str(e))
                else:
                    raise
    best_idx = np.argmin([r[1]['d'] for r in responses])
    best_image = responses[best_idx][0]
    best_step = responses[best_idx][1]['i']
    best_dist = responses[best_idx][1]['d']
    print('best image is %s at distance %f' % (best_image.image_id, best_dist))
    return best_image, best_step, best_dist


# TODO: currently not clear if replica exchange groups will ever be needed, better remove this code
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
        self._previous = previous
        self.colvars_def = colvars_def
        self.opaque = opaque
        self.groups = dict()
        # create and populate groups, if group_ids were found in the config
        for im in images.values():
            if im.group_id is not None:
                if im.group_id not in self.groups:
                    self.groups[im.group_id] = Group(group_id=im.group_id, string=self)
                self.groups[im.group_id].add_image(im)

    def _default_fields(self, selection):
        if isinstance(selection, AllType):
            return self.images_ordered[0].fields
        else:
            return selection

    def __str__(self):
        str_images = '{' + ', '.join(['%g:%s'%(seq, im) for seq, im in self.images.items()]) + '}'
        return 'String(branch=\'%s\', iteration=%d, images=%s, image_distance=%f, previous=%s, colvars_def=%s, opaque=%s)' % (
            self.branch, self.iteration, str_images, self.image_distance, self._previous, self.colvars_def, self.opaque)

    def __getitem__(self, key):
        if isinstance(key, float) or isinstance(key, int):
            return self.images[float(key)]
        elif isinstance(key, str):
            return next(im for im in self.images.values() if im.image_id == key)
        else:
            raise ValueError('key is neither a number nor a string, don\'t know what to do with it.')

    @property
    def is_homogeneous(self) -> bool:
        'True, if all the nodes live in the same collective variable space (uniform dimension).'
        fields = self.images_ordered[0].fields
        for im in self.images_ordered[1:]:
            if im.fields != fields:
                return False
        return True

    def discretize(self, points, states_per_arc=100):
        # TODO first check compatibility with path (fields)
        from .util import pairing
        arcs = [self]  # currently we only support one arc, TODO: change this
        sz = np.array([b.project(points, return_z=True) for b in arcs])
        best = np.argmin(sz[:, 1])  # find the closest arc for each input point
        i = np.fromiter((arcs[best_t].i for best_t in best), dtype=int)
        j = np.fromiter((arcs[best_t].j for best_t in best), dtype=int)
        return ((pairing(i, j, ordered=False) + sz[best, 0]) * states_per_arc).astype(int)

    def empty_copy(self, iteration=None, images=None, previous=None, branch=None):
        'Creates a copy of the current String object. Image array is left empty.'
        if iteration is None:
            iteration = self.iteration
        if previous is None:
            previous = self._previous
        if images is None:
            images = dict()
        if branch is None:
            branch = self.branch
        return String(branch=branch, iteration=iteration, images=images, image_distance=self.image_distance,
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

    def bisect_at(self, i, j=None, subdir='colvars', where='node', search='string'):
        r'''Create new image half-way between images i and j.

        Parameters
        ----------
        i : int or float
            Identifies the first image in the pair. If int, pick self.images_ordered[i] as the first,
            if float, pick self[i] as the first image (interpret the float as major.minor)
        j : int or float or None
            Like j. If None, assume j =  i + 1
        subdirs : str
            Subfolder where to search for colvars. Used to find images that are close in CV space.
        where : str
            One of 'mean', 'x0', or 'node'. Select which property of images to interpolate to find the midpoint.
        search : str
            One of 'points' or 'string'. Where to search for realized frames that are close to the midpoint.
            'string' searches in the whole string, 'points' searches only among the swarm data of images i and j.
        fields : iterable or All
            What do consider as the collective variable space. Default All,

        example
        -------
            ov, ids = string.overlap(return_ids=True)
            new_images = []
            for pair, o in zip(ids, ov):
                if o < 0.10:
                    new_images.append( string.bisect_at(i=pair[0], j=pair[1]) )
            for new_image in new_images:
                string.add_image(new_image)
            string.write_yaml(message='bisected at positions of low overlap')  # adds the images to the plan.yaml file

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

        real_fields = p.fields  # use same cvs as node of p

        if where == 'mean':
            x = recarray_average(p.colvars(subdir=subdir, fields=real_fields).mean, q.colvars(subdir=subdir, fields=real_fields).mean)
        elif where == 'x0':
            x = recarray_average(p.x0(subdir=subdir, fields=real_fields), q.x0(subdir=subdir, fields=real_fields))
        elif where == 'node' or where == 'center':
            x = recarray_average(p.node, q.node)
        elif where == 'plane':
            raise NotImplementedError('bisection at SVM plane not implemented yet')
        else:
            raise ValueError('Unrecognized value "%s" for option "where"' % where)

        if search == 'points':
            query_p = p.colvars(subdir=subdir, fields=real_fields).closest_point(x)
            query_q = q.colvars(subdir=subdir, fields=real_fields).closest_point(x)
            if query_p['d'] < query_q['d']:
                print('best distance is', query_p['d'])
                best_image = p
                best_step = query_p['i']
            else:
                print('best distance is', query_q['d'])
                best_image = q
                best_step = query_q['i']
        elif search == 'string':
            best_image, best_step, best_dist = find_realization_in_string([self], node=x, subdir=subdir)
            print('best distance is', best_dist, '@', best_image.image_id, ':', best_step)
        else:
            raise ValueError('Unrecognized value "%s" of parameter "search"' % search)

        new_image_id = interpolate_id(p.image_id, q.image_id, excluded=[im.image_id for im in self.images.values()])
        if any(new_image_id == im.image_id for im in self.images.values()):
            raise RuntimeError('Bisection produced new image id which is not unique. This should not happen.')

        new_image = best_image.__class__(image_id=new_image_id, previous_image_id=best_image.image_id,
                                         previous_frame_number=best_step, node=x, spring=p.spring.copy(), group_id=None,
                                         swarm=best_image.swarm)
        # TODO: for linear bias, where to put the terminal (just the next node along the string)
        # TODO: or should we use some interpolation to control the distance between node and terminal?

        #   self.images[new_image.seq] = new_image
        return new_image

    def bisect(self, threshold=0.1, write=False):
        'Add intermediate images at locations of low overlap between images.'
        ov, ids = self.overlap(return_ids=True)
        new_images = []
        for pair, o in zip(ids, ov):
            if o < threshold:
                new_images.append(self.bisect_at(i=pair[0], j=pair[1]))
        for new_image in new_images:
            self.add_image(new_image)
        if write:
            self.write_yaml(message='bisected at positions of low overlap')

    interpolate = bisect

    def bisect_and_propagate_util_connected(self, run_locally=False):
        r'''Blocking function that automatically bisects and submits jobs until the string is connected.

        Parameters
        ----------
        run_locally: bool
            Run MD on this machine instead of submitting to the queuing system.
        '''
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
        # TODO: this property name is confusing, pick a better one
        return '{root}/strings/{branch}_{iteration:03d}'.format(root=root(), branch=self.branch, iteration=self.iteration)

    def check_against_string_on_disk(self):
        'Compare currently loaded string to the version on disk. Check that read-only elements were not modified.'
        try:
            on_disk = String.load(branch=self.branch, offset=self.iteration)
        except FileNotFoundError:
            # plan file does not exist, we therefore assume that this is a new string (or new iteration)
            return True
        # image values written to disk are not allowed to be changed in memory (once a sim_id is assigned, never change record)
        for on_disk_key, on_disk_image in on_disk.images.items():
            if on_disk_key not in self.images:
                warnings.warn('Some images that have already been written to disk to the same plan have been deleted '
                              'in RAM. Unless you know exactly what you are doing, the current string cannot be saved.')
                return False
            if not self.images[on_disk_key] == on_disk_image:
                warnings.warn(('Image %f that has already been written to disk have changed in RAM. ' +
                              'Unless you know exactly what you are doing, the current string cannot be saved.') % on_disk_key)
                return False
        return True

    def write_yaml(self, backup=True, message=None, _override=False):
        'Save the full status of the String to yaml file in directory $STRING_SIM_ROOT/strings/<branch>_<iteration>'
        import shutil
        if not _override:
            if not self.check_against_string_on_disk():
                raise RuntimeError(
                    'There is already a plan with the same name on disk that has different images from the one you are '
                    'trying to save. Stopping the operation, new string was not saved! You can override, if you are sure '
                    'what you are doing.')
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
            yaml.dump(config, f, width=1000, default_flow_style=None)  # default_flow_style=False,


    def evolve_kinetically(self, subdir='colvars', threshold=0.1, overlap='units', augment_colvars=True, adjust_springs=True, swarm=True, write=False):
        '''Create a copy of the String where images are evolved and the string is reduced to images on the kinetically widest sub-path.

            Parametes
            ---------
            overlap: str or ndarray((n, n))
                One of 'plane', 'units' or an overlap matrix in ndarray format.

            Notes
            -----
            TODO: Future generation of this software should not only find the single widest path
            to propagate but some collection of dominant paths.
        '''
        from .util import widest_path, structured_to_dict, dict_to_structured, IDEAL_GAS_CONSTANT, DEFAULT_TEMPERATUE
        from collections import defaultdict

        max_spring_constant = 1000.

        if not swarm:
            raise NotImplementedError('evolve_kinetically can be only used with swarm=True')

        if isinstance(overlap, str) and overlap in ['plane', 'units']:
            overlap_matrix = self.overlap(subdir=subdir, algorithm=overlap, matrix=True)
        else:
            overlap_matrix = overlap
        epsilon = 1E-6  # to make sure that the matrix stays formally connected; TODO: find better solution (distance-based!)
        indices_of_images_to_evolve = widest_path(overlap_matrix + epsilon)
        print('length of widest path is', len(indices_of_images_to_evolve))
        print('overlap at bottleneck is', min(overlap_matrix[i, j] for i, j in
                                              zip(indices_of_images_to_evolve[0:-1], indices_of_images_to_evolve[1:])))
        images_to_evolve = [self.images_ordered[i] for i in indices_of_images_to_evolve]

        # Augment colvars searches through all the colvars atom by atom (or unit by unit) and
        # adds them to the node, if there is no overlap. Colvars that are already part of the node,
        # stay there.
        # We already selected the widest path, but that does not mean that the path is wide enough,
        # i.e. the bottleneck can still be small. So we need to get a bigger colvar space under control.
        if augment_colvars:
            fields = defaultdict(set)
            for i, (a, b) in enumerate(zip(images_to_evolve[0:-1], images_to_evolve[1:])):
                fields[a] |= set(a.fields)  # copy ...
                fields[b] |= set(b.fields)  # ... old fields
                f = set(Image.get_fields_non_overlapping(a, b, subdir=subdir, threshold=threshold))
                fields[a] |= f  # for each image B, we might add fields to improve both overlaps each of A and C
                fields[b] |= f  # A .. B .. C
                if len(f) > 0:
                    print('Added new field(s)', f, 'to images', i, 'and', i + 1)
            new_nodes = [im.colvars(fields=list(fields[im])).mean for im in images_to_evolve]  # set new node to the old means
            del fields
        else:
            new_nodes = [im.colvars(fields=im.fields).mean for im in images_to_evolve]

        # Compute the distance between neighbors and set spring constant accordingly, sigma=delta.
        # If adjust springs is False, only fill in missing spring constants.
        new_springs = [structured_to_dict(im.spring) for im in images_to_evolve]
        RT = IDEAL_GAS_CONSTANT * DEFAULT_TEMPERATUE  # kcal/mol
        for i, (a, b) in enumerate(zip(new_nodes[0:-1], new_nodes[1:])):
            for f in set(a.dtype.names) & set(b.dtype.names):  # only adjust springs for which we can compute the distance between (augmented) nodes
                dist = np.linalg.norm(a[f] - b[f])  # not sure if this is a good idea; node distances are too random
                k = RT / dist**2  # kcal/mol/A^2
                if f not in new_springs[i] or adjust_springs:
                    #current_spring = new_springs[i][f] if f in new_springs[i] else 0.
                    new_springs[i][f] = min(k, max_spring_constant)
                if f not in new_springs[i + 1] or adjust_springs:
                    #current_spring = new_springs[i + 1][f] if f in new_springs[i + 1] else 0.
                    new_springs[i + 1][f] = min(k, max_spring_constant)


        # create the actual string object
        iteration = self.iteration + 1
        new_string = self.empty_copy(iteration=iteration, previous=self)
        image_class = images_to_evolve[0].__class__

        for i_running, (node, spring) in enumerate(zip(new_nodes, new_springs)):
            best_image, best_step, best_dist = find_realization_in_string([self], node=node, subdir=subdir)

            new_image = image_class(image_id='%s_%03d_%03d_%03d' % (self.branch, iteration, i_running, 0),
                                    previous_image_id=best_image.image_id, previous_frame_number=best_step,
                                    node=node, terminal=None, spring=dict_to_structured(spring),
                                    group_id=best_image.group_id, swarm=swarm)

            new_string.add_image(new_image)

        if write:
            new_string.write_yaml(message='kinetic evolution')
        return new_string

    # TODO: write down the simplest version that includes a colvar into all iamges of the new iteration, once it is found to be important
    # for any pair of images!
    def evolve(self, subdir='colvars', reparametrize=True, update_fields=False, n_nodes=None, rmsd=False, linear_bias=0,
               threshold=0.1, adjust_springs=True, swarm=None, do_smooth=False):
        '''Created a copy of the String where the images are evolved and the string is reparametrized to have geometically equidistant images.

           Parameters
           ----------
           reparametrize : bool
                Reparamtrized the new string such that images are equidistant in collective
                variable space.

            update_fields : bool
                Include colvars that exhibit low overlap into the (controlled) nodes.
                If a colvar in any image show low overlap, the colvar is added to the
                whole string.

            n_nodes: int
                Reparametrize the string such that the new string contains exactly n_nodes.

           rmsd: bool
               during reparametrization, image distance is measured with the RMSD metric that
               takes into account the number of atoms as apposed to the Euclidean matric
               (rmsd=False) that does not normalize by the number of atoms. Affects how
               self.image_distance is interpreted. With rmsd=True you will get a string with
               less images, everything else being euqal.
               You can check len() of the resulting string, to be sure.

           linear_bias: int
               If linear_bias > 0, construct a 1-D bias that acts solely in a direction
               tangential to the string. linear_bias = 2 uses a second order interpolation
               to define the tangents. binear_bias = 1 uses first order.

           swarm: boolean or None
               Run a swarm of trajectories after equilibrating with an umbrella potential.
               Refer to setup files in the setup folder for details. Images wil be run with
               the environment variable STRING_SWARM=1.

           Returns
           -------
           A new String object with an iteration number = 1 + iteration number of parent string (self).
           You can propagate it using .propagate or save it to disk with .write_yaml (recommended).
        '''

        # collect all means, in the same time check that all the coordinate dimensions and
        # coordinate names are the same across the string
        from .util import bisect_decreasing, dict_to_structured, structured_to_dict, IDEAL_GAS_CONSTANT
        from collections import defaultdict
        if not self.is_homogeneous and reparametrize:
            raise RuntimeError('Not all nodes live in exactly the same colvars space. This is not supported by this function.')

        max_spring_constant = 1000.

        if update_fields:
            fields = set()
            images_ordered = self.images_ordered
            for a, b in zip(images_ordered[0:-1], images_ordered[1:]):
                fields |= set(a.fields)  # copy ...
                fields |= set(b.fields)  # ... old fields
                fields |= set(Image.get_fields_non_overlapping(a, b, subdir=subdir, threshold=threshold))
            fields = list(fields)
        else:
            fields = self.images_ordered[0].fields

        colvars_0 = self.images_ordered[0].colvars(subdir=subdir, fields=fields)
        real_fields = colvars_0.fields  # replaces All by concrete list of names
        dims = colvars_0.dims
        current_means = []
        for image in self.images_ordered:
            colvars = image.colvars(subdir=subdir, fields=real_fields)
            #if reparametrize and (set(colvars.fields) != set(real_fields) or colvars.dims != dims):
            #    raise RuntimeError('colvars fields / dimensions are inconsistent across the string. First inconsistency in image %s.' % image.image_id)
            # The geometry functions in the reparametrization module work with 2-D numpy arrays, while the colvar
            # class used recarrays and (1, n) shaped ndarrays. We therefore convert to plain numpy and strip extra dimensions.
            current_means.append(structured_to_flat(colvars.mean, fields=real_fields)[0, :])

        if rmsd:
            n_atoms = len(real_fields)
        else:
            n_atoms = 1

        if do_smooth is not False:
            from .reparametrization import smooth
            current_means = smooth(np.array(current_means), do_smooth)

        # do the string reparametrization
        if reparametrize:
            ordered_means = reorder_nodes(nodes=current_means)  # in case the string "coiled up", we reorder its nodes
            if n_nodes is not None:
                print('running bisecion to find the correct parameter for reparametrization ')
                i_d = bisect_decreasing(lambda x: len(compute_equidistant_nodes_2(old_nodes=ordered_means, d=x)),
                            a=self.image_distance*n_atoms**0.5, b=self.image_distance*n_atoms**0.5, level=n_nodes)
                print('result is', i_d)
                self.image_distance = i_d / n_atoms**0.5
            nodes = compute_equidistant_nodes_2(old_nodes=ordered_means, d=self.image_distance * n_atoms**0.5,
                                                d_skip=self.image_distance * n_atoms**0.5 / 2)
            # do some self-consistency tests of the reparamtrization step
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
        else:
            nodes = current_means

        # compute new spring constants
        springs = defaultdict(dict)
        T = 303.15
        RT = IDEAL_GAS_CONSTANT * T
        for i, (x, y) in enumerate(zip(nodes[0:-1], nodes[1:])):
            for f in real_fields:
                k = RT / np.linalg.norm(x[f] - y[f])**2
                #current_spring = springs[i][f] if f in springs[i] else 0.
                springs[i][f] = min(k, max_spring_constant)
                #current_spring = springs[i + 1][f] if f in springs[i + 1] else 0.
                springs[i + 1][f] = min(k, max_spring_constant)

        iteration = self.iteration + 1
        new_string = self.empty_copy(iteration=iteration, previous=self)
        image_class = self.images_ordered[0].__class__

        for i_node, x in enumerate(nodes):
            # we have to convert the unrealized nodes (predicted point of conformational space) to realized frames
            node = flat_to_structured(x[np.newaxis, :], fields=real_fields, dims=dims)
            best_image, best_step, best_dist = find_realization_in_string([self], node=node, subdir=subdir)

            if swarm is None:
                swarm = best_image.swarm

            if adjust_springs:
                spring = dict_to_structured(springs[i_node])
            else:
                spring = best_image.spring

            new_image = image_class(image_id='%s_%03d_%03d_%03d' % (self.branch, iteration, i_node, 0),
                                    previous_image_id=best_image.image_id, previous_frame_number=best_step,
                                    node=node, terminal=None, spring=spring,
                                    group_id=best_image.group_id, swarm=swarm)

            new_string.images[new_image.seq] = new_image

            # only for tangential biases (no isotropic or elliptic bias)
            if linear_bias > 0:
                if linear_bias == 1:
                    for a, b in zip(new_string.images_ordered[0:-1], new_string.images_ordered[1:]):
                        a.set_terminal_point(b.node)
                elif linear_bias == 2:  # order 2 interpolation
                    # a  ----- b ----- c ---- ....
                    # |\_______ _______/
                    # |        |
                    # node     +> terminal
                    imo = new_string.images_ordered
                    for a, c in zip(imo[0:-2], imo[2:]):
                        a.set_terminal_point(recarray_average(a.node, c.node))
                    imo[-2].set_terminal_point(imo[-1].node)
                else:
                    raise NotImplementedError('Higher-order interpolation schemes (beyond 2) are not implemented yet.')

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
    def load(cls, branch='AZ', offset=-1):
        r'''Find specific iteration of the string in $STRING_SIM_ROOT/strings/ and recover it from the yaml file.

        Parameters
        ----------
        branch: str
            name of the branch to load
        offset: int
            If this number is positive, load string with iteration number equal to offset.
            If this number is negative, count iteration numbers backwards starting
            from the maximal iteration (which can currently be found on disk).
            -1 (default) loads the highest iteration.
            -2 loads the second to last iteration and so on.
            This corresponds to Python array indexing.

        Returns
        -------
        A String object.
        '''
        if offset == 0:
            raise ValueError('Offset cannot be zero. Must be >0 or <0.')
        elif offset > 0:
            fname = '%s/strings/%s_%03d/plan.yaml' % (root(), branch, offset)
            string = cls.load_form_fname(fname)
        else:
            folder = root() + '/strings/'
            max_iteration = float('-inf')
            for entry in os.listdir(folder):
                splinters = entry.split('_')
                if len(splinters) == 2:
                    folder_branch, folder_iteration = splinters
                    if folder_branch == branch and folder_iteration.isdigit():
                        max_iteration = max([max_iteration, int(folder_iteration)])
            if max_iteration > float('-inf'):
                print('Highest current iteration is %d. Loading iteration %d' % (max_iteration, max_iteration + 1 + offset))
                fname = '%s/strings/%s_%03d/plan.yaml' % (root(), branch, max_iteration + 1 + offset)
                string = cls.load_form_fname(fname)
            else:
                raise RuntimeError('No string with branch identifier "%s" found' % branch)
        return string

    @classmethod
    def load_form_fname(cls, fname):
        'Create a String object by recovering the information form the yaml file whose path is given as the argument.'
        with open(fname) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        string = config['strings'][0]  # TODO: in the future, support multiple strings per file (possibly different iterations?)
        colvars_def = string['colvars'] if 'colvars' in string else None
        branch = string['branch']
        iteration = string['iteration']
        image_distance = string['image_distance']
        images_arr = [load_image(config=img_cfg, colvars_def=colvars_def) for img_cfg in string['images']]
        images = {image.seq: image for image in images_arr}
        opaque = {key: config[key] for key in config.keys() if key not in ['strings']}  # TODO: currently opaque refers to things outside of the string, also handle information inside
        # TODO: issue a warning, if we are potentially discarding opaque fields in the string...
        return String(branch=branch, iteration=iteration, images=images, image_distance=image_distance, previous=None,
                      opaque=opaque, colvars_def=colvars_def)

    @property
    def previous_string(self):
        'Attempt to load previous iteration of the string'
        if self._previous is None:
            if self.iteration > 1:
                print('loading iteration %d' % (self.iteration - 1))
                self._previous = String.load(branch=self.branch, offset=self.iteration - 1)
            else:
                raise RuntimeError('There is no string before iteration 1. Please use Image.x0 to access data from the "zero\'th" iteration.')
                # print('loading *')  # TODO: this is currently broken and will not work like this
                # self._previous = String.from_scratch(branch=self.branch, iteration_id=0)  # TODO: find better solution
        return self._previous


    def bisect_and_lift(self, subdirs: str='colvars', fields=All, threshold: float=0.1, order='sequential', overlap='units', write=False):
        r'''Bisect and lift to higher dimension (by increasing the number of collective variables), if needed.


        This method essentially calls bottlenecks and bisect_and_lift at.

        Parameters
        ----------
        See `String.bisect_and_lift_at` and `String.bottlenecks`.
        '''
        new_images = []
        bottlenecks = self.bottlenecks(subdirs=subdirs, overlap=overlap, order=order, threshold=threshold, fields=fields,
                                       ignore_missing=True)
        for a, b in bottlenecks:
            new_image = self.bisect_and_lift_at(a, b, insert=False, threshold=threshold)
            new_images.append(new_image)
        for im in new_images:
            print('Created new image', im.image_id)
            self.add_image(im)
        if write:
            self.write_yaml(message='filled gaps and lifted')
        return new_images


    def bisect_and_lift_at(self, a: Image, b: Image, subdir: str='colvars', insert: bool=True, threshold: float=0.1, T: float=303.15) -> Image:
        r'''Bisect and lift to higher dimension (by increasing the number of collective variables), if needed.

        Parameters
        ----------
        a: Image
            First image in pair.
        b: Image
            Second image in pair.
        subdir: str
            Name of the subdir with the "large" colvar space.
        default_subdir: str
            Name of the subdir wiht the "small" colvar space.
        insert: bool
            Insert new image into the current string?
        threshold: float
            Threshold for overlap under which, new atoms are added to colvar space.

        Notes
        -----
        Use this together with bottlenecks:
        >>> for a, b in s.bottlenecks(overlap='units_3D'):
        >>>     s.bisect_and_lift_at(a, b, insert=True)
        >>> s.write_yaml(message='filled gaps')

        Returns
        -------
        A newly generated (unpropagated) image
        '''
        from .colvars import overlap_svm
        from .util import recarray_dims, IDEAL_GAS_CONSTANT
        RT = IDEAL_GAS_CONSTANT * T  # kcal/mol

        x = a.colvars(subdir=subdir)  #  This is not necessarily the colvars subdir!
        y = b.colvars(subdir=subdir)
        # x.fields and y.fields will contain all fields and not only the ones that are currently under control

        # now add new fields, if required (the current node contains only the cvs that are already known, need to go back to x0 to look at all dimensions)
        x0 = a.x0(subdir=subdir)  # only needed for new fields
        y0 = b.x0(subdir=subdir)  # ditto

        # find field to be added (because of low overlap)
        low_fields = [field for field in x.fields if overlap_svm(x._colvars[field], y._colvars[field]) < threshold]
        all_fields = list(set(low_fields) | set(a.fields) | set(b.fields))
        #print('all_fields', all_fields)
        #print('low_fields', low_fields)
        new_fields = list(set(low_fields) - set(a.fields) - set(b.fields))
        if len(new_fields) > 0:
            print('added fields', new_fields, 'to nodes', a.image_id, '+', b.image_id)
        # now make new node object; first collect sizes of all new and old (node) fields
        dim = {name: d for name, d in zip(x.fields, x.dims) if name in new_fields}
        dim.update({name:d for name, d in zip(a.node.dtype.names, recarray_dims(a.node))})
        dim.update({name:d for name, d in zip(b.node.dtype.names, recarray_dims(b.node))})
        new_dtype = np.dtype([(f, np.float32, dim[f]) for f in all_fields])
        new_node = np.zeros(1, dtype=new_dtype)
        new_spring = np.zeros(1, dtype=new_dtype)
        for field in new_fields:
            a_node_f = x0[field]
            b_node_f = y0[field]
            new_node[field] = 0.5*(a_node_f + b_node_f)
            new_spring[field] = RT * np.linalg.norm(a_node_f - b_node_f)**-2.  # RMSD or not???
        # now add the known fields
        for field in set(a.fields) & set(b.fields):
            new_node[field] = 0.5*(a.node[field] + b.node[field])
            new_spring[field] = 0.5*(a.spring[field] + b.spring[field])
        # half known
        for field in set(a.fields) - set(b.fields):
            new_node[field] = a.node[field]
            new_spring[field] = a.spring[field]
        for field in set(b.fields) - set(a.fields):
            new_node[field] = b.node[field]
            new_spring[field] = b.spring[field]

        # find realization, first prepare list of all previous strings
        s = self
        strings_to_search = [s]
        n_iterations_crawled = 1
        while s.iteration >= 2 and n_iterations_crawled < 10:  # lowest string to search is string with index 1
            s = s.previous_string
            strings_to_search.append(s)
            n_iterations_crawled += 1
        best_image, best_frame, _ = find_realization_in_string(strings=strings_to_search, node=new_node, subdir=subdir)
        new_image_id = interpolate_id(a.image_id, b.image_id, excluded=[im.image_id for im in self.images_ordered])
        new_image = a.__class__(image_id=new_image_id, previous_image_id=best_image.image_id,
                                previous_frame_number=best_frame, group_id=None, node=new_node, spring=new_spring,
                                swarm=a.swarm)

        if insert:
            self.add_image(new_image)

        return new_image


    def bottlenecks(self, subdirs='colvars', overlap='units', order='sequential', threshold=0.1, fields=All, ignore_missing=True):
        r'''Advanced identification of gaps in the sampling. Determines pairs of images to sample in-between.

        Parameters
        ----------
        subdirs: str or list of str
            List of subdirectory names in the observables directory.
        overlap: str  # TODO: allow to pass in a matrix and just use that?
            One of 'units' or 'plane'. Determines type of overlap computation. 'units' finds dividing planes
            for each atom (or more precisely: for each 3D unit) separately. 'plane' finds a plane in the full
            product space of all observables.
        order: str
            One of 'sequential' or 'shortest'. How to search for bottlenecks. 'sequential' follows the canonical
            order of the string, i.e. only examines nodes that are direct neightbors according to the canconical
            (arc length) indexing of the string. 'shortest' determines the min-bottleneck path from all possible
            paths through the images.
        threshold:
            Pairs of images with overlap lower than this value are returned.
        ignore_missing: bool
            Ignore missing observable files; simply skip pairs that have missing files.
        alpha: float
            Parameter for definition of image distance for the computation of the optimal
            image pair for interpolation.

        Returns
        -------
        List of pairs of images. See notes below.

        Notes
        -----
        >>> import sarangi
        >>> s = sarangi.load(branch='AZ')
        >>> for a, b in s.bottlenecks(overlap='units'):
        >>>    s.bisect_and_lift_at(a, b, insert=True)
        >>> s.write_yaml()
        >>> s.propagate()
        '''
        from .util import recarray_difference, recarray_norm, widest_path
        if overlap not in ['units', 'plane']:
            raise NotImplementedError('Only overlap="units_3D" is currently implemented.')
        pairs = []
        if order == 'sequential':
            for a, b in zip(self.images_ordered[0:-1], self.images_ordered[1:]):
                try:
                    if a.overlap_3D_units(b, subdir=subdirs, fields=fields) < threshold:
                        pairs.append((a, b))
                except FileNotFoundError as e:
                    if ignore_missing:
                        warnings.warn(str(e))
                    else:
                        raise
            return pairs
        elif order == 'shortest':
            n = len(self)
            print('computing the overlap matrix')
            overlap_matrix = np.zeros((n, n)) + np.nan
            #distance_matrix = np.array((n, n))
            # find the overlaps for all pairs of nodes
            for i, a in enumerate(tqdm(self.images_ordered)):
                overlap_matrix[i, i] = 0.
                for j_excess, b in enumerate(self.images_ordered[i + 1:]):
                    j = i + 1 + j_excess
                    if overlap == 'units':
                        overlap_matrix[i, j] = a.overlap_3D_units(b, subdir=subdirs, fields=fields)
                    else:
                        overlap_matrix[i, j] = a.overlap_plane(b, subdir=subdirs, fields=fields)
                    overlap_matrix[j, i] = overlap_matrix[i, j]
                    #distance_matrix[i, j] = recarray_norm(recarray_difference(a.node, b.node), rmsd=True)
                    #distance_matrix[j, i] = distance_matrix[i, j]
            overlap_matrix[-1, -1] = 0.
            # now find the min-bottleneck path
            # TODO: play around with different score: e.g. 1/overlap_matrix as distance score
            #weight_matrix = np.where(overlap_matrix > 0, 1. - overlap_matrix, 1. + alpha*distance_matrix)  # this might require a bit of experimentation
            print('finding the widest path')
            epsilon = 1.E-6  # TODO: do this distance-based instead of
            path = widest_path(overlap_matrix + epsilon)
            pairs = [(a,b) for a, b in zip(path[0:-1], path[1:])]
            overlap = [overlap_matrix[a, b] for a, b in pairs]
            print('bottleneck has overlap:', min(overlap))
            im_o = self.images_ordered
            return [(im_o[i], im_o[j]) for ov, (i, j) in zip(overlap, pairs) if ov < threshold]
        else:
            raise ValueError('Unknown value of `order`. Must be one of "sequential" or "shortest".')

    def overlap(self, subdir='colvars', fields=All, algorithm='plane', indicator='max', matrix=False, return_ids=False):
        r'''Compute the overlap (SVM) between images of the string.

        Notes
        -----
        Order of images in the retuned array/ matrix is the same as in self.images_ordered.
        '''
        from tqdm import tqdm
        real_fields = self.images_ordered[0].colvars(subdir=subdir, fields=fields).fields
        ids = []
        if not matrix:
            o = np.zeros(len(self.images_ordered) - 1) + np.nan
            for i, (a, b) in enumerate(zip(tqdm(self.images_ordered[0:-1]), self.images_ordered[1:])):
                try:
                    if algorithm=='plane':
                        o[i] = a.overlap_plane(b, subdir=subdir, fields=real_fields, indicator=indicator)
                    else:
                        o[i] = a.overlap_3D_units(b, subdir=subdir, fields=real_fields, indicator=indicator)
                    ids.append((a.seq, b.seq))
                except FileNotFoundError as e:
                    warnings.warn(str(e))
            if return_ids:
                return o, ids
            else:
                return o
        else:
            o = np.zeros((len(self.images_ordered), len(self.images_ordered))) + np.nan
            for i, a in enumerate(tqdm(self.images_ordered[0:-1])):
                o[i, i] = 0.
                for j, b in enumerate(self.images_ordered[i+1:]):
                    try:
                        if algorithm == 'plane':
                            o[i, i + j + 1] = a.overlap_plane(b, subdir=subdir, fields=real_fields, indicator=indicator)
                        else:
                            o[i, i + j + 1] = a.overlap_3D_units(b, subdir=subdir, fields=real_fields, indicator=indicator)
                        o[i + j + 1, i] = o[i, i + j + 1]
                    except FileNotFoundError as e:
                        warnings.warn(str(e))
            o[-1, -1] = 0.
            return o

    def overlap_by_atom(self, subdir='colvars'):
        'Good for finding non-overlapping atoms.'
        fields = list(self.images_ordered[0].node.dtype.names)
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
        self.check_order(subdir=subdir)
        f = np.zeros(len(self.images) - 1) + np.nan
        mbar = [None] * (len(self.images) - 1)
        deltaf = [None] * (len(self.images) - 1)
        for i, (a, b) in enumerate(zip(self.images_ordered[0:-1], self.images_ordered[1:])):
            try:
                f[i], deltaf[i], mbar[i] = a.bar(b, subdir=subdir, T=T)
            except FileNotFoundError as e:
                warnings.warn(str(e))
        return -np.cumsum(f), deltaf, mbar

    def arclength_projections(self, subdir='colvars', order=1, x0=False, curve_defining_string=None, where='nodes', defining_subdir='colvars'):
        r'''For all frames in all simulations, compute the arc length order parameter.

        Parameters
        ----------
        order: int
            Interpolation order for the computation of arc length. Can take the values
            0, 1, or 2.
        subdir: str
            If working with non-default string setup, set this to the folder name for
            the collective variable trajectories.
        x0: boolean, default=False
            By default project all the time series data to the curve defining string.
            If x0 is True, only project the initial points (image.x0) to the string.
        curve_defining_string: String object, default=self
            Images of this string define the curve through CV space onto which the images
            in self are projected. This is good to e.g. compare several projections of
            the same data or to have the exact same project for multiple data sets.
        where: str
            One of 'nodes' or 'means'. Whether to use the means or the nodes from
            curve_defining_string to define the curve through collective variable space.
        defining_subdir: str
            One used if `where=="nodes"`. Where to find the collective variables to
            compute the means. (Why do we need this?)

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
        from tqdm import tqdm
        if curve_defining_string is None:
            curve_defining_string = self
        fields = curve_defining_string.images_ordered[0].fields
        if where == 'nodes':
            support_points = [structured_to_flat(image.node, fields=fields)[0, :] for image in
                              curve_defining_string.images_ordered]
        elif where == 'means':
            support_points = [structured_to_flat(image.colvars(subdir=defining_subdir, fields=fields).mean, fields=fields)[0, :]
                              for image in curve_defining_string.images_ordered]
        else:
            raise ValueError('where must be one of "nodes" or "means"')
        # remove duplicate points https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        # TODO: allow for some epsilon parameter!
        x = []
        for r in support_points:
            if tuple(r) not in x:
                x.append(tuple(r))
        support_points = np.array(x)

        results = []
        for image in tqdm(self.images_ordered):  # TODO: return as dictionary instead?
            try:
                if x0:
                    x = image.x0(subdir=subdir, fields=fields)
                    results.append(Colvars.arclength_projection([structured_to_flat(x, fields=fields)],
                                                                nodes=support_points, order=order))
                else:
                    x = image.colvars(subdir=subdir, fields=fields)
                    results.append(Colvars.arclength_projection(x.as2D(fields=fields), nodes=support_points, order=order))
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
        # RT = * T  # kcal/mol
        forces = []
        fields = self.images_ordered[0].fields
        support_points = [structured_to_flat(image.node, fields=fields)[0, :] for image in self.images_ordered]
        for image in self.images_ordered:
            for f in fields:
                if isinstance(image.spring, np.ndarray) and image.spring[f] != image.spring[fields[0]]:
                    raise NotImplementedError('PMF computation currently only implemented for non-elliptic forces')
            try:
                if image.bias_is_isotropic:
                    mean = image.colvars(subdir=subdir, fields=fields).mean
                    node_proj = Colvars.arclength_projection(structured_to_flat(image.node, fields=fields), support_points, order=1)[0]
                    #print(node_proj)
                    mean_proj = Colvars.arclength_projection(structured_to_flat(mean, fields=fields), support_points, order=1)[0]
                    #print(mean_proj)
                    #print(type(mean_proj), type(node_proj), image.spring[fields[0]][0])
                    forces.append((node_proj - mean_proj)*image.spring[fields[0]][0])  # TODO: implement the general anisotropic case (TODO: move part to the Image classes)
                else:
                    forces.append(np.mean(image.tangential_displacement())*image.spring)
            except FileNotFoundError as e:
                forces.append(0.)
                warnings.warn(str(e) + ' Replacing the missing mean force value by a value of zero.')
        if integrate:
            distances = np.zeros(len(self.images_ordered) - 2)
            for i, (a, b) in enumerate(zip(self.images_ordered[0:-2], self.images_ordered[2:])):
                delta = recarray_difference(a.node, b.node)
                distances[i] = 0.5 * recarray_vdot(delta, delta)**0.5
            return np.cumsum(np.array(forces)[1:-1] * np.array(distances))
        else:
            return forces

    def mbar(self, subdir='colvars', T=303.15, disc_subdir='colvars', disc_fields=All, disc_centers=None, f=1.0):
        'Estimate all free energies using MBAR (when running with conventional order parameters, not RMSD)'
        import pyemma
        import pyemma.thermo
        from .util import IDEAL_GAS_CONSTANT

        RT = IDEAL_GAS_CONSTANT * T  # kcal/mol

        fields = self.images_ordered[0].fields
        for image in self.images.values():
            if image.fields != fields:
                raise RuntimeError('Images have varying node dimensions, cannot use this MBAR wrapper.')
            if not isinstance(image.spring.dtype.names, float) and list(image.spring.dtype.names) != fields:
                raise RuntimeError('Images have varying spring dimensions, cannot use this MBAR wrapper.')

        # collect all possible biases
        # FIXME: this will not recognize duplicate biases
        unique_biases = []
        for im in self.images_ordered:
            bias = (im.node, im.spring, im)
            if bias not in unique_biases:
                unique_biases.append(bias)
        print('found', len(unique_biases), 'unique biases')

        btrajs = []
        ttrajs = []
        K = len(unique_biases)

        dtrajs = [np.round(np.maximum(x, 0.)*f).astype(int) for x in self.arclength_projections(x0=False)]

        for i_im, image in enumerate(self.images_ordered):
            try:
                x = image.colvars(subdir=subdir, fields=fields, memoize=False)
                btraj = np.zeros((len(x), K))
                ttraj = np.zeros(len(x), dtype=int) + i_im
                for k, bias in enumerate(unique_biases):
                    bias[2].potential(colvars=x, factor=1./RT, out=btraj[:, k])
                btrajs.append(btraj)
                ttrajs.append(ttraj)
                #if disc_centers is not None:
                #    y = image.colvars(subdir=disc_subdir, fields=disc_fields, memoize=False).as2D(fields=disc_fields)
                #    dtrajs.append(pyemma.coordinates.assign_to_centers(y, centers=disc_centers)[0])
                #else:
                #    dtrajs.append(np.zeros(len(x), dtype=int))
            except FileNotFoundError as e:
                warnings.warn(str(e))

        print('running MBAR')
        mbar = pyemma.thermo.MBAR(direct_space=True, maxiter=100000, maxerr=1e-13)
        mbar.estimate((ttrajs, dtrajs, btrajs))

        return mbar

    def mbar_RMSD(self, T=303.15, subdir='rmsd'):  # TODO: move some of this into the Image classes
        'For RMSD-type bias: Estimate all free energies using MBAR'
        import pyemma.thermo
        from .util import IDEAL_GAS_CONSTANT

        RT = IDEAL_GAS_CONSTANT * T  # kcal/mol

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
        'Compute displacement vector in collective variable space for each image.'
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

    @property
    def propagated_substring(self):
        'Return a new string that only contains the subset of propagated images.'
        filtered_string = self.empty_copy()
        for im in self.images.values():
            if im.propagated:
                filtered_string.add_image(im)
        return filtered_string

    def kinkiness(self):
        'Return scalar products between successive tangents to the string. If terminal points of images are set, use them.'
        from .util import recarray_difference, recarray_vdot
        scalar_products = []
        for a, b, c in zip(self.images_ordered[0:-2], self.images_ordered[1:-1], self.images_ordered[2:]):
            if a.terminal is not None:
                d1 = recarray_difference(a.node, a.terminal)
            else:
                d1 = recarray_difference(a.node, b.node)
            norm1 = recarray_vdot(d1, d1)
            if b.terminal is not None:
                d2 = recarray_difference(b.node, b.terminal)
            else:
                d2 = recarray_difference(b.node, c.node)
            norm2 = recarray_vdot(d2, d2)
            scalar_products.append(recarray_vdot(d1, d2) / (norm1 ** 0.5 * norm2 ** 0.5))
        return scalar_products

    def check_order(self, subdir='colvars'):
        'Check whether the ordering by ID (as in .images_ordered) is consistent with the node distance.'
        fields = self.images_ordered[0].fields
        nodes = [structured_to_flat(im.colvars(subdir=subdir, fields=fields).mean, fields=fields) for im in
                 self.images_ordered]
        nodes = np.concatenate(nodes)
        for i, n in enumerate(nodes):
            dist = np.linalg.norm(nodes - n[np.newaxis, :], axis=1)
            me, top1, top2 = np.argsort(dist)[0:3]
            # print(i, me, top1, top2)
            if top1 != i + 1 and top1 != i - 1:
                print('Node %d is not closest to its neighbors according to ordering by ID. Clostest node is %d.' % (
                i, top1))


    def microstate_assignments(self, subdir='colvars', fields=All, centers='means'):
        r'''Compute microstate assignment for all data in string. Ordering of result is consistent with self.images_ordered.

            Parameters
            ----------
            subdir : str
                subfolder name for the colvars of the initial point / initial ensemble starting conformations

            TODO

            Returns
            -------
            state_seq: list of np.ndarray
                state_seq[i] is the collection of assigned swarm points from image i
        '''
        if isinstance(centers, np.ndarray):
            cluster_centers = centers
        elif centers == 'means':
            cluster_centers = []
            for im in self.images_ordered:
                center = structured_to_flat(im.colvars(subdir=subdir, fields=fields).mean, fields=fields)[:, :]
                cluster_centers.append(center)
            cluster_centers = np.concatenate(cluster_centers)
        elif centers == 'nodes':
            cluster_centers = []
            for im in self.images_ordered:
                center = structured_to_flat(im.node, fields=fields)[:, :]
                cluster_centers.append(center)
            cluster_centers = np.concatenate(cluster_centers)
        else:
            raise ValueError('centers must be one of "means", "nodes" or an array')

        s_images = []
        # TODO: test me!!!
        for im in self.images_ordered:
            frames = im.colvars(subdir=subdir, fields=fields).as2D(fields=fields)
            #print('frames.shape', frames.shape)
            #print('cluster_centers.shape', cluster_centers.shape)
            dmat = np.linalg.norm(frames[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2)
            idx = np.argmin(dmat, axis=1)
            s_images.append(idx)
        #s_images = self.arclength_projections(subdir=subdir, x0=False, order=order,
        #                                      curve_defining_string=curve_defining_string, where=clusters,
        #                                      defining_subdir=defining_subdir)
        #s_images = np.maximum(s_images, 0.)
        return s_images

    def count_matrix(self, f=1.0, subdir='colvars', subdir_init='colvars_init', fields=All, centers='means', return_t=False):
        r'''Estimate MSM from swarm of trajectories data

            Parameters
            ----------
            return_t : bool
                return reversible transition matrix as well

            f : float
                stretching factor the state indices, arc length is multiplied by f before casting to integer

            subdir_init : str
                subfolder name for the colvars of the initial point / initial ensemble starting conformations


            centers : str
                One of 'nodes' or 'means'. Where to place the cluster centers of the microstates.
                # TODO: add multistate SVM as an extra option


            Returns
            -------
            count_matrix: ndarray
                count matrix estimated at the maximum possible lag time
                state indices correspond to the indices of the images in self.images_ordered,
                if f = 1
                Can be used to estimate free energies or kinetics (such as mean-first passage times)
                with MSM packages such as PyEmma or MSMTools.
            transition_matrix: ndarray (only if return_t is True)
                Reversible transition matrix estimated from the count matrix.
        '''
        import msmtools
        #from tqdm import tqdm_notebook as tqdm
        dtrajs = []
        #if subdir_init is None:
        #    s_starts_images = np.maximum(self.arclength_projections(x0=True), 0.)
        #else:
        s_starts_images = self.microstate_assignments(subdir=subdir_init, fields=fields, centers=centers)
        #s_starts_images = self.arclength_projections(subdir=subdir_init, x0=False, order=order,
        #                                             curve_defining_string=curve_defining_string, where=clusters,
        #                                             defining_subdir=defining_subdir)
        #s_starts_images = np.maximum(s_starts_images, 0.)
        s_ends_images = self.microstate_assignments(subdir=subdir, fields=fields, centers=centers)
        #s_ends_images = self.arclength_projections(x0=False, order=order, curve_defining_string=curve_defining_string,
        #                                           where=clusters, defining_subdir=defining_subdir)
        #s_ends_images = np.maximum(s_ends_images, 0.)
        if len(s_starts_images) != len(s_ends_images):
            raise RuntimeError(
                'Internal inconsistency of swarm data. Number of images in initial points and final points is different.')
        for s_starts, s_ends in zip(s_starts_images, s_ends_images):
            if len(s_starts) != len(s_ends):
                raise RuntimeError(
                    'Internal inconsistency of swarm data. Swarm size is different in initial points and final points.')
            for s_start, s_end in zip(s_starts, s_ends):
                dtrajs.append([int(round(s_start*f)), int(round(s_end*f))])
        c = msmtools.estimation.cmatrix(dtrajs, lag=1)
        if return_t:
            t = msmtools.estimation.tmatrix(c)
            return c.toarray(), t.toarray()
        else:
            return c.toarray()

    def fel_from_msm(self, T=303.15, subdir='colvars', subdir_init='colvars_init', fields=All, centers='means', return_pi=False):
        r'''Compute the PMF along the string from the stationary distribution of an MSM estimated form the swam data.

            Parameters
            ----------
            T : float
                temperature in Kelvin

            f : float
                see String.count_matrix

            subdir_init : str
                see String.count_matrix

            clusters : str
                One of 'nodes' or 'means'. Where to place the cluster centers of the microstates.
                see String.count_matrix

            order : int
                see String.count_matrix

            return_pi: bool
                If True, return the stationary vector (stationary distribution),
                else return the emprirical free energy estiamte -RT log(pi).

            Returns
            -------
            PMF in kcal/mol computed from -RT log(pi)

            Note
            ----
            It is assumed that the canonical ordering of the images in the string is meaningful.
            See String.check_order to test the assumption. Also consider to use MSM reweighting.

        '''
        import msmtools
        from .util import IDEAL_GAS_CONSTANT
        RT = IDEAL_GAS_CONSTANT * T  # kcal/mol
        c = self.count_matrix(subdir=subdir, subdir_init=subdir_init, fields=fields, centers=centers)
        cset = msmtools.estimation.largest_connected_set(c, directed=True)
        t_cset = msmtools.estimation.tmatrix(c[cset, :][:, cset])
        pi_cset = msmtools.analysis.stationary_distribution(t_cset)
        pi = np.zeros(c.shape[0])
        pi[cset] = pi_cset
        if return_pi:
            return pi
        else:
            return -RT * np.log(pi)
        # TODO: implement reweighting ... ; pass in some field
        # for every frame get (cluster id, observable); just for final or for final and initial data?

    def all_colvars(self, subdir='colvars', fields=All):
        r'''For all final swarm points from all swarms, return colvars.'''
        all_cv = []
        for im in self.images_ordered:
            all_cv.append(im.colvars(subdir=subdir, fields=fields).as2D(fields))
        return all_cv

    def msm_reweighting_factors(self, subdir='colvars', subdir_init='colvars_init', fields=All, centers='means'):
        r'''Compute reweighting factors that reweight simulation frames towards the equilibrium distribution.

        Note
        ----
        w = s.msm_reweighting_factors(fields=['X', 'Y'])  # estimate MSM in X,Y space
        o = s.all_colvars(fields=['X'])
        plt.hist(o, weights=w)  # only show the equilibrium histogram of X
        '''
        import msmtools
        assignments = self.microstate_assignments(subdir=subdir, fields=fields, centers=centers)
        #assignments_0 = self.microstate_assignments(subdir=subdir_init, fields=fields, centers=centers)
        c = self.count_matrix(subdir=subdir, subdir_init=subdir_init, fields=fields, centers=centers)
        cset = msmtools.estimation.largest_connected_set(c, directed=True)
        t_cset = msmtools.estimation.tmatrix(c[cset, :][:, cset])
        pi_cset = msmtools.analysis.stationary_distribution(t_cset)
        cnt_cset = c.sum(axis=0)[cset]
        mu = np.zeros(c.shape[0])
        for j, i in enumerate(cset):
            mu[i] = pi_cset[j] / cnt_cset[j]
        return [mu[assignments_image] for assignments_image in assignments]

    def _curvature_errors_analytical(self):
        fields = self.images_ordered[0].fields
        n = len(self.images_ordered) - 2
        chi = np.zeros(n) + np.nan
        delta_chi = np.zeros(n) + np.nan
        for i in range(n):
            im_m = self.images_ordered[i]
            im_i = self.images_ordered[i + 1]
            im_p = self.images_ordered[i + 2]
            if im_m.fields != fields or im_i.fields != fields or im_p.fields != fields:
                warnings.warn('Images in triple (%s,%s,%s) have inconsistent fields, cannot compute curvature.' % (
                    im_m.image_id, im_i.image_id, im_p.image_id))
            cv_m = im_m.colvars()
            cv_p = im_p.colvars()
            cv_i = im_i.colvars()
            y_m = structured_to_flat(cv_m.mean, fields=fields)[0, :]
            y_i = structured_to_flat(cv_i.mean, fields=fields)[0, :]
            y_p = structured_to_flat(cv_p.mean, fields=fields)[0, :]
            t_p = y_p - y_i
            t_i = y_i - y_m
            l_p = np.linalg.norm(t_p)
            l_i = np.linalg.norm(t_i)
            chi_i = np.vdot(t_p, t_i) / (l_p*l_i)
            chi[i] = np.abs(chi_i)
            grad_m = t_p / (l_i*l_p) - chi_i * y_m / (t_i*t_i)  # TODO: check equations!
            grad_i = (t_i - t_p) / (t_i*t_p) - chi_i * y_i / (t_i*t_i) - chi_i * y_i / (t_p*t_p)
            grad_p = t_i / (l_i*l_p) - chi_i * y_p / (t_p*t_p)
            delta_chi[i] = (np.vdot(grad_m, np.dot(cv_m.error_matrix, grad_m)) +
                            np.vdot(grad_i, np.dot(cv_i.error_matrix, grad_i)) +
                            np.vdot(grad_p, np.dot(cv_p.error_matrix, grad_p)))**0.5
        return chi, delta_chi
        #for field, k in zip(cv.fields, cv.dims):
        #    gradient[] =
        # TODO: also return a regularized string as a list of nodes (can be used for US or string evolution)

    def _curvature_errors_sampling(self, n_samples=100):
        from tqdm import tqdm
        fields = self.images_ordered[0].fields
        n = len(self.images_ordered) - 2
        chi = np.zeros(n) + np.nan
        delta_chi = np.zeros(n) + np.nan
        for i in tqdm(range(n)):
            im_m = self.images_ordered[i]
            im_i = self.images_ordered[i + 1]
            im_p = self.images_ordered[i + 2]
            if im_m.fields != fields or im_i.fields != fields or im_p.fields != fields:
                warnings.warn('Images in triple (%s,%s,%s) have inconsistent fields, cannot compute curvature.' % (
                    im_m.image_id, im_i.image_id, im_p.image_id))
            cv_m = im_m.colvars()
            cv_p = im_p.colvars()
            cv_i = im_i.colvars()
            y_m = structured_to_flat(cv_m.mean, fields=fields)[0, :]
            y_i = structured_to_flat(cv_i.mean, fields=fields)[0, :]
            y_p = structured_to_flat(cv_p.mean, fields=fields)[0, :]
            t_p = y_p - y_i
            t_i = y_i - y_m
            l_p = np.linalg.norm(t_p)
            l_i = np.linalg.norm(t_i)
            chi[i] = np.abs(np.vdot(t_p, t_i) / (l_p * l_i))
            samples = []
            for _ in range(n_samples):
                y_m_ = structured_to_flat(cv_m.bootstrap_mean(), fields=fields)[0, :]
                y_i_ = structured_to_flat(cv_i.bootstrap_mean(), fields=fields)[0, :]
                y_p_ = structured_to_flat(cv_p.bootstrap_mean(), fields=fields)[0, :]
                t_p_ = y_p_ - y_i_
                t_i_ = y_i_ - y_m_
                l_p_ = np.linalg.norm(t_p_)
                l_i_ = np.linalg.norm(t_i_)
                chi_i_ = np.abs(np.vdot(t_p_, t_i_) / (l_p_ * l_i_))
                samples.append(chi_i_)
            delta_chi[i] = np.std(samples)
        return chi, delta_chi

    def curvature_errors(self, analytical=False, n_samples=100):
        if analytical:
            return self._curvature_errors_analytical()
        else:
            return self._curvature_errors_sampling(n_samples=n_samples)

    def smooth(self, filter_width=3, points='mean'):
        'Smooth the string by applying an moving average filter to the nodes'
        fields = self.images_ordered[0].fields
        nodes = []
        for im in self.images_ordered:
            if points == 'mean':
                nodes.append(structured_to_flat(im.colvars().mean, fields=fields)[0, :])
            elif points == 'bootstrap':
                nodes.append(structured_to_flat(im.colvars().bootstrap_mean(), fields=fields)[0, :])
            elif points == 'node':
                nodes.append(structured_to_flat(im.node, fields=fields)[0, :])
            else:
                raise ValueError('Unrecognized value of parameter points. Should by one of "mean", "node" or "bootstrap".')
        nodes = np.array(nodes)
        smoothed = np.zeros_like(nodes)
        for i in range(nodes.shape[1]):
            padded = np.pad(nodes[:, i], (filter_width // 2, filter_width - 1 - filter_width // 2), mode='edge')
            smoothed[:, i] = np.convolve(padded, np.ones((filter_width,)) / filter_width, mode='valid')
        # preserve initial and final nodes
        smoothed[0, :] = nodes[0, :]
        smoothed[-1, :] = nodes[-1, :]
        # TODO: convert to string?

        # compute the hypothetical length of the string after equidistant reparametrization
        ordered_means = reorder_nodes(nodes=smoothed)
        new_len = len(compute_equidistant_nodes_2(old_nodes=ordered_means, d=self.image_distance, d_skip=self.image_distance))

        # TODO: compute some curvatures ...

        return smoothed, new_len

        #new_nodes = (nodes[0], nodes[-1])
        # TODO: return what kind of object? # TODO: have some simpler container class that does not have pointer to previous simulation...

    def smoothing_error(self, filter_width=3, n_samples=100):
        from tqdm import tqdm
        from .reparametrization import curvatures
        chi_samples = []
        for _ in tqdm(range(n_samples)):
            nodes, length = self.smooth(filter_width=filter_width, points='bootstrap')
            chi_samples.append(curvatures(nodes))
        return np.std(chi_samples, axis=0)


def parse_commandline(argv=None):
    import argparse
    import sys
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wait', help='wait for job completion', default=False, action='store_true')
    parser.add_argument('--dry', help='dry run', default=False, action='store_true')
    parser.add_argument('--local', help='run locally (on this machine)', default=False, action='store_true')
    parser.add_argument('--re', help='run replica exchange simulations', default=False, action='store_true')
    parser.add_argument('--cpus_per_replica', help='number of CPU per replica (only for multi-replica jobs with NAMD)', default=32)
    parser.add_argument('--iteration', help='do not propagate current string but a past iteration', default=None)
    parser.add_argument('--branch', help='select branch', default='AZ')
    parser.add_argument('--max_workers', help='number of concurrent workers, important if running locally', default=1)
    #parser.add_argument('--distance', help='distance between images', default=1.0)
    #parser.add_argument('--boot', help='bootstrap computation', default=False, action='store_true')
    args = parser.parse_args(argv)

    options=  {'wait': args.wait, 'run_locally': args.local, 'dry': args.dry, 're': args.re,
               'cpus_per_replica': args.cpus_per_replica, 'branch': args.branch, 'max_workers': int(args.max_workers)}
    if args.iteration is not None:
        options['iteration'] = int(args.iteration)
    else:
        options['iteration'] = None
    return options


def rounds(s, n):
    for i in range(n):
        # bisect to cure bad overlap
        print('closing gaps in string', s.iteration)
        for j in range(3):  # TODO: convert this into a while loop with bottleneck criterion
            s.bisect_and_lift(order='shortest', overlap='units', write=True)
            s.propagate(wait=True, run_locally=True)
        # propagation
        print('evolving the string', s.iteration)
        next_s = s.evolve_kinetically(write=True)
        next_s.propagate(wait=True, run_locally=True)
        s = next_s
    return s


def rounds_same_load(s, n):
    for _ in range(n):
        s_new = s.evolve(n_nodes=len(s) + 1)
        s_new.write_yaml()
        s_new.propagate(wait=True, run_locally=True)
        s_new.bisect_and_lift(overlap='units', write=True)
        s_new.propagate(wait=True, run_locally=True)
        s = s_new
    return s

def load(branch='AZ', offset=-1):
    return String.load(branch=branch, offset=offset)


def main(argv=None):
    options = parse_commandline(argv)

    if options['iteration'] is None:
        string = load(branch=options['branch'])  # TODO: refactor loading functions such that this becomes one function
    elif options['iteration'] < 0:
        string = load(branch=options['branch'], offset=options['iteration'])
    else:
        string = String.load(branch=options['branch'], offset=options['iteration'])
    print(string.branch, string.iteration, ':', string.ribbon(run_locally=options['run_locally']))

    if not string.propagated:
        string = string.propagate(wait=options['wait'], run_locally=options['run_locally'], dry=options['dry'],
                                  cpus_per_replica=options['cpus_per_replica'], max_workers=options['max_workers'])
    #else:  # reparametrize and propagate at least once
    #    string = string.bisect_and_propagate_util_connected(run_locally=options['run_locally']).reparametrize().propagate(wait=options['wait'], run_locally=options['run_locally'])

    #if options['wait']:  # keep looping
    #    while True:
    #        print(string.iteration, ':', string.ribbon(run_locally=options['run_locally']))
    #        string = string.bisect_and_propagate_util_connected(run_locally=options['run_locally']).reparametrize().propagate(wait=options['wait'], run_locally=options['run_locally'])

    # while not converged:
    #    iterate and reparametrize string
    #    while not connected
    #        fill gaps in string


if __name__ == '__main__':
    import sys
    sys.exit(main())

