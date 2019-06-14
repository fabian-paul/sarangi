import numpy as np
import warnings
import os
from .colvars import Colvars
from .util import load_structured, dump_structured, recarray_difference, recarray_vdot, structured_to_flat, \
    All, AllType, root


__all__ = ['Image', 'load_image', 'interpolate_id']


# name generation routines
def interpolate_id(s, z, excluded):
    'Create new simulation ID between s and z, preferentially leaving low digits zero.'
    excluded_int = []
    for id_ in excluded:
        _, _, a, b = id_.split('_')
        excluded_int.append(int(a + b))

    branch, iter_, a, b = s.split('_')
    _, _, c, d = z.split('_')
    x = a + b
    y = c + d
    upper = max(int(x), int(y))
    lower = min(int(x), int(y))
    # print('searching', upper, lower)
    result = between(0, 6, upper=upper, lower=lower, excluded=excluded_int)
    # print('searching', upper, lower)
    if not result:
        raise RuntimeError('Could not generate an unique id.')
    return '%s_%03s_%03d_%03d' % (branch, iter_, result // 1000, result % 1000)


def interval_overlap(a, b, c, d):
    assert a <= b and c <= d
    if b < c or a > d:
        return 0
    else:
        return min(b, d) - max(a, c)


def between(a, e, upper, lower, excluded):
    integers = np.array([0, 5, 4, 6, 3, 7, 2, 8, 9, 1])

    if e < 0:
        return False
    for i in integers:
        x = a + 10 ** e * i
        if lower < x < upper and x not in excluded:
            # print('success', x)
            return x

    # compute overlap with possible trial intervals, then try the ones with large overlap first
    sizes = [interval_overlap(lower, upper, a + 10 ** e * i, a + 10 ** e * (i + 1)) for i in integers]
    for i in integers[np.argsort(sizes)[::-1]]:
        l = a + 10 ** e * i
        r = a + 10 ** e * (i + 1)
        if lower < r and upper > l:
            # print('recursing', l, r, e, i)
            result = between(l, e - 1, upper, lower, excluded)
            if result:
                return result
    # print('not found')
    return False


class Image(object):
    _known_keys = ['id', 'prev_image_id', 'prev_frame_number', 'group', 'swarm']

    def __init__(self, image_id, previous_image_id, previous_frame_number, group_id, swarm, opaque):
        self.image_id = image_id
        self.previous_image_id = previous_image_id
        self.previous_frame_number = previous_frame_number
        self.group_id = group_id
        self.swarm = swarm
        self.opaque = opaque
        self._colvars = {}
        self._x0 = {}

    @classmethod
    def _load_params(cls, config):
        image_id = config['id']
        previous_image_id = config['prev_image_id']
        previous_frame_number = config['prev_frame_number']
        if 'swarm' in config:
            swarm = config['swarm']
        else:
            swarm = False
        if 'group' in config:
            group_id = config['group']
        else:
            group_id = None
        return {'image_id': image_id, 'previous_image_id': previous_image_id,
                'previous_frame_number': previous_frame_number, 'group_id': group_id, 'swarm': swarm}

    def dump(self):
        'Dump state of object to dictionary. Called by String.dump'
        config = {'id': self.image_id, 'prev_image_id': self.previous_image_id,
                  'prev_frame_number': self.previous_frame_number}
        if self.group_id is not None:
            config['group'] = self.group_id
        if self.swarm:
            config['swarm'] = True
        return config

    @property
    def bias_is_isotropic(self):
        return self.terminal is None

    def namd_conf(self, cwd='./'):
        if self.spring is None:
            return ''
        elif self.bias_is_isotropic:
            return self._isotropic_namd_conf(cwd)
        else:
            return self._linear_namd_conf(cwd)

    def openmm_conf(self, cwd):
        raise NotImplementedError('Maybe next year.')

    @property
    def fields(self):
        'Names of colvars used to define the umbrella center (node)'
        return list(self.node.dtype.fields)

    def __str__(self):
        return self.image_id

    def __eq__(self, other):
        for key in ['image_id', 'previous_image_id', 'previous_frame_number', 'group_id']:
            if self.__dict__[key] != other.__dict__[key]:
                return False
        return True

    @property
    def propagated(self):
        'MD simulation was completed.'
        if os.path.exists(self.base + '.dcd'):
            return True
        for folder in os.listdir(self.colvar_root):
            if os.path.exists('%s/%s/%s.colvars.traj' % (self.colvar_root, folder, self.image_id)):
                return True
            if os.path.exists('%s/%s/%s.npy' % (self.colvar_root, folder, self.image_id)):
                return True
        return False

    def _make_job_file(self, env):  # TODO: move into gobal function
        'Created a submission script for the job on the local file system.'
        import tempfile
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
        return float('%03d.%03d' % (self.id_major, self.id_minor))

    @property
    def job_name(self):
        return 'im_' + self.image_id

    @property
    def base(self):
        'Base path of the image. base+".dcd" is the path to the MD data.'
        return '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
            root=root(), branch=self.branch, iteration=self.iteration, id_minor=self.id_minor, id_major=self.id_major)

    @property
    def previous_base(self):
        branch, iteration, id_major, id_minor = self.previous_image_id.split('_')
        return '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
            root=root(), branch=branch, iteration=int(iteration), id_minor=int(id_minor), id_major=int(id_major))

    def _make_env(self, random_number):
        # this will typically be run on the headnode or some other dedicated node on the compute cluster
        env = dict()
        root_ = root()
        # TODO: clean up this list a little, since the job script has access to the plan file/string object anyway
        env['STRING_SIM_ROOT'] = root_
        env['STRING_ITERATION'] = str(self.iteration)
        env['STRING_BRANCH'] = self.branch
        env['STRING_IMAGE_ID'] = self.image_id
        env['STRING_PLAN'] = '{root}/strings/{branch}_{iteration:03d}/plan.yaml'.format(root=root_, branch=self.branch,
                                                                                        iteration=self.iteration)
        env['STRING_PREV_IMAGE_ID'] = self.previous_image_id
        env['STRING_PREV_FRAME_NUMBER'] = str(self.previous_frame_number)
        env['STRING_RANDOM'] = str(random_number)
        env['STRING_PREV_ARCHIVE'] = self.previous_base
        env['STRING_ARCHIVE'] = self.base
        env['STRING_ARCHIVIST'] = os.path.dirname(__file__) + '/../scripts/string_archive.py'
        env['STRING_SARANGI_SCRIPTS'] = os.path.dirname(__file__) + '/../scripts'
        if self.swarm:
            env['STRING_SWARM'] = 1
        else:
            env['STRING_SWARM'] = 0
        env['STRING_BASE'] = '{root}/strings/{branch}_{iteration:03d}'.format(root=root_,
                                                                              branch=self.branch,
                                                                              iteration=self.iteration)
        env['STRING_OBSERVABLES_BASE'] = '{root}/observables/{branch}_{iteration:03d}'.format(root=root_,
                                                                                              branch=self.branch,
                                                                                              iteration=self.iteration)
        return env

    def propagate(self, random_number, wait, queued_jobs, run_locally=False, dry=False):
        'Generic propagation command. Submits jobs for the intermediate points. Copies the end points.'
        import subprocess
        if self.propagated:
            return self

        #  if the job is already queued, return or wait and return then
        if self.job_name in queued_jobs:
            print('skipping submission of', self.job_name, 'because already queued')
            if wait:  # poll the results file
                import time
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

    @property
    def colvar_root(self):
        return '{root}/observables/{branch}_{iteration:03d}/'.format(root=root(), branch=self.branch,
                                                                     iteration=self.iteration)

    def colvars(self, subdir='colvars', fields=All, memoize=True):
        'Return Colvars object for the set of collective variables saved in a given subdir and limited to given fields'
        if not isinstance(fields, AllType) and (subdir, tuple(fields)) in self._colvars:
            return self._colvars[(subdir, tuple(fields))]
        else:
            folder = '{root}/observables/{branch}_{iteration:03d}/'.format(
                root=root(), branch=self.branch, iteration=self.iteration)
            base = '{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
                branch=self.branch, iteration=self.iteration, id_minor=self.id_minor, id_major=self.id_major)
            pcoords = Colvars(folder=folder + subdir, base=base, fields=fields)
            if memoize and not isinstance(fields, AllType):  # All is to vague to be memoized
                self._colvars[(subdir, tuple(fields))] = pcoords
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

    def potential(self, colvars, factor=1.0, out=None):
        'Compute the bias potential parametrized by self.node and self.spring evaluated for all time steps of the colvar trajectory.'
        if out is None:
            u = np.zeros(len(colvars))
        else:
            u = out
        if self.bias_is_isotropic:
            node = self.node
            spring = self.spring
            u[:] = 0.  # zero out to be sure
            for name in self.node.dtype.names:
                if isinstance(spring, float):
                    k = spring
                else:
                    k = float(spring[name])
                u[:] += 0.5 * factor * k * np.sum((colvars._colvars[name] - node[name]) ** 2, axis=1)
            return u
        else:
            delta = recarray_difference(self.node, self.terminal)
            norm = recarray_vdot(delta, delta) ** 0.5
            disp = -recarray_vdot(recarray_difference(colvars._colvars, self.node), delta, allow_broadcasting=True) / norm
            u[:] += 0.5 * factor * self.spring * disp**2.
            return u

    def bar(self, other, subdir='colvars', T=303.15):
        'Compute thermodynamic free energy difference between this window (self) and other using BAR.'
        import pyemma
        from pyemma.util.contexts import settings
        RT = 1.985877534E-3 * T  # kcal/mol
        fields = list(self.node.dtype.names)
        self_x = self.colvars(subdir=subdir, fields=fields)
        other_x = other.colvars(subdir=subdir, fields=fields)
        btrajs = [np.zeros((len(self_x), 2)), np.zeros((len(other_x), 2))]
        self.potential(colvars=self_x, factor=1./RT, out=btrajs[0][:, 0])
        other.potential(colvars=self_x, factor=1./RT, out=btrajs[0][:, 1])
        self.potential(colvars=other_x, factor=1./RT, out=btrajs[1][:, 0])
        other.potential(colvars=other_x, factor=1./RT, out=btrajs[1][:, 1])
        ttrajs = [np.zeros(len(self_x), dtype=int), np.ones(len(other_x), dtype=int)]

        # TODO: use some discretization (TODO; implement discretization)
        mbar = pyemma.thermo.MBAR(maxiter=100000)
        with settings(show_progress_bars=False):
            mbar.estimate((ttrajs, ttrajs, btrajs))
            df = mbar.f_therm[0] - mbar.f_therm[1]
            # error computation
            N_1 = btrajs[0].shape[0]
            N_2 = btrajs[1].shape[0]
            u = np.concatenate((btrajs[0], btrajs[1]), axis=0)
            du = u[:, 1] - u[:, 0]
            b = (1.0 / (2.0 + 2.0 * np.cosh(
                df - du - np.log(1.0 * N_1 / N_2)))).sum()  # TODO: check if we are using df with the correct sign!
            delta_df = 1 / b - (N_1 + N_2) / (N_1 * N_2)

        return df, delta_df, mbar

    # TODO: have some version that provides the relative signs (of the scalar product) of neighbors
    def displacement(self, subdir='colvars', fields=All, norm='rmsd', origin='node'):
        'Compute the difference between the windows\'s mean and its initial position in order parameter space'
        if origin == 'x0':
            o = self.x0(subdir=subdir, fields=fields)
        elif origin == 'node' or origin == 'center':
            if isinstance(fields, AllType):
                o = self.node
            else:
                o = self.node[fields]
        else:
            raise ValueError('origin must be either "node" or "x0"')
        mean = self.colvars(subdir=subdir, fields=fields).mean
        if norm == 'rmsd':
            n_atoms = len(o.dtype.names)
        else:
            n_atoms = 1
        if norm != 'rmsd':
            ord = norm
        else:
            ord = None
        return np.linalg.norm(
            structured_to_flat(mean, fields=list(mean.dtype.names)).reshape(-1) - structured_to_flat(o, fields=list(mean.dtype.names)).reshape(-1),
            ord=ord) * (n_atoms ** -0.5)

    def tangential_displacement(self, subdir='colvars'):
        'Return trajectory of tangential displacements (trajectory of signed scalar values, typically in units of Angstroms)'
        x = self.colvars(subdir=subdir, fields=self.fields)._colvars
        delta = recarray_difference(self.node, self.terminal)
        norm = recarray_vdot(delta, delta)**0.5
        return -recarray_vdot(recarray_difference(x, self.node), delta, allow_broadcasting=True) / norm


    def set_terminal_point(self, point):
        if sorted(point.dtype.names) != sorted(self.node.dtype.names):
            raise ValueError('point does not match field signature')
        self.terminal = point.copy()
        if self.spring is not None and not isinstance(self.spring, float):
            warnings.warn('Setting terminal point but spring is still parametrizing an "elliptic" bias. '
                          'If spring is not changed this won\'t run.')

    def discretize(self):
        raise NotImplementedError('Not yet implemented.')


class CompoundImage(Image):
    _known_keys = Image._known_keys + ['bias', 'node', 'terminal', 'spring', 'orthogonal_spring']

    # collection of other CVs; does not support and never will support PDB generation
    # NAMD conf will expand to customfunction and multiple harmonic forces
    # spring can eb a scarlar or a vector
    def __init__(self, image_id, previous_image_id, previous_frame_number, group_id,
                 node, spring, terminal=None, orthogonal_spring=None, colvars_def=None, swarm=None, opaque=None):
        super(CompoundImage, self).__init__(image_id=image_id, previous_image_id=previous_image_id,
                                            previous_frame_number=previous_frame_number, group_id=group_id,
                                            swarm=swarm, opaque=opaque)

        self.node = node
        self.terminal = terminal
        self.spring = spring
        self.orthogonal_spring = orthogonal_spring
        self.colvars_def = colvars_def
        self.bias = 'compound'

    def copy(self):
        node = self.node.copy() if self.node is not None else None
        if self.spring is None or isinstance(self.spring, float):
            spring = self.spring
        else:
            spring = self.spring.copy()
        terminal = self.terminal.copy() if self.node is not None else None
        return CompoundImage(image_id=self.image_id, previous_image_id=self.previous_image_id,
                             previous_frame_number=self.previous_frame_number, group_id=self.group_id,
                             node=node, spring=spring, terminal=terminal,
                             orthogonal_spring=self.orthogonal_spring, colvars_def=self.colvars_def, opaque=self.opaque)

    @classmethod
    def load(cls, config, colvars_def):
        known_keys = super(CompoundImage, cls)._known_keys + ['bias', 'node', 'terminal', 'spring', 'orthogonal_spring']
        opaque = {key: config[key] for key in config.keys() if key not in known_keys}
        node = load_structured(config['node'])
        if 'terminal' in config:
            terminal = load_structured(config['terminal'])
        else:
            terminal = None
        if 'spring' in config:
            if isinstance(config['spring'], dict):
                spring = load_structured(config['spring'])  # named arrays only for "elliptic" bias
            else:
                spring = config['spring']
        else:
            spring = None
        if 'orthogonal_spring' in config:
            orthogonal_spring = config['spring']
        else:
            orthogonal_spring = None

        image = cls(node=node, terminal=terminal, spring=spring, orthogonal_spring=orthogonal_spring,
                    colvars_def=colvars_def, opaque=opaque, **super(CompoundImage, cls)._load_params(config))
        return image

    def dump(self):
        config = super(CompoundImage, self).dump()
        config['bias'] = 'compound'
        if self.node is not None:
            config['node'] = dump_structured(self.node)
        if self.terminal is not None:
            config['terminal'] = dump_structured(self.terminal)
        if self.spring is not None:
            if isinstance(self.spring, float):
                config['spring'] = self.spring
            else:
                config['spring'] = dump_structured(self.spring)
        if self.orthogonal_spring is not None:
            config['orthogonal_spring'] = self.orthogonal_spring
        if self.opaque is not None:
            config.update(self.opaque)
        return config

    def __eq__(self, other):
        if not super(CompoundImage, self).__eq__(other):
            return False
        for key in ['node', 'spring', 'terminal', 'orthogonal_spring']:
            if not np.array_equal(self.__dict__[key], other.__dict__[key]):
                return False
        return True

    def _isotropic_namd_conf(self, cwd):
        import textwrap
        # for fun we could redefine all the colvars form the information in the plan file
        config = ''
        for restraint_name in self.fields:
            spring_value = self.spring[restraint_name]
            if self.node is not None and restraint_name in self.node.dtype.names:
                center_value = self.node[restraint_name]
                center_value_namd = '(' + ' , '.join([str(x) for x in center_value[0]]) + ')'
            else:
                warnings.warn('Spring constant was defined but no umbrella center. Using the default 0.0.')
                center_value_namd = '0.0'
            config += textwrap.dedent('''
                          harmonic {{
                            name {restraint_name}_restraint
                            colvars {restraint_name}
                            forceconstant {spring}
                            centers {center}
                          }}
                         '''.format(restraint_name=restraint_name,
                                    center=center_value_namd,
                                    spring=float(spring_value)))
        return config

    def _linear_namd_conf(self, cwd):
        import textwrap
        if self.colvars_def is None or 'compound' not in self.colvars_def:
            raise RuntimeError('MD setup with linear order parameter (linearly directed bias) was requested but, plan file '
                               'does not contain definitions of the collective variables. Can\'t create MD setup file.')
        config = ''
        a = self.node
        b = self.terminal
        if self.orthogonal_spring is not None:
            raise NotImplementedError('Linear (tangent) order parameter with constraint along the orthogonal direction is '
                                      'not yet supported. Only constraint along tangent direction is implemented.')
        # implements the equation (a-x(t))*(a-b) / ||a-b||  where * is the dot product and, x is the current point in CV space
        # this expression is 0, if x==a==node and is equal to ||a-b|| if x==b==terminal point
        a_minus_b = recarray_difference(a, b)
        norm = np.linalg.norm(structured_to_flat(a_minus_b))
        const = recarray_vdot(a, a_minus_b) / norm
        expr = ['%f' % const]  # a*(a-b) / ||a-b||
        for field in a_minus_b.dtype.names:
            dim = a_minus_b[field].shape[1]
            assert a_minus_b[field].shape[0] == 1
            for i in range(dim):
                expr.append(  # - x_i * (a-b)_i / ||a-b||
                    '{field}{dim}*{factor}'.format(field=field, dim=i + 1, factor=-a_minus_b[field][0][i] / norm))
        config += textwrap.dedent('''
                    colvar {{
                      name tangent
                      customFunction {{{expression}}}
                    '''.format(expression='+'.join(expr)))
        # now repeat all colvar definitions but with the name moved inside (see colvar documentation)
        for colvar in self.colvars_def['compound']:
            # name is inside in contrast to the "normal" colvar definition
            if 'type' in colvar and colvar['type'] != 'com':
                warnings.warn('Found colvars def of type %s, don\'t know how to handle.')
            config += textwrap.dedent('''
                        distanceVec {{
                          name {field}
                          group2 {{ atomnumbers {{ {atoms} }} }}
                          group1 {{ dummyAtom ( 0.0 , 0.0 , 0.0 ) }}
                        }}
                        '''.format(field=colvar['name'], atoms=' '.join([str(i) for i in colvar['atomnumbers']])))
        config += '}\n'
        # add harmonic force
        if not isinstance(self.spring, float):
            raise ValueError('Field spring is of type %s, expecting float. Don\'t know how to handle this for applying'
                             'a unidirectional force.' % (type(self.spring)))
        config += textwrap.dedent('''
                    harmonic {{
                      name tanget_restraint
                      colvars tangent
                      forceconstant {spring}
                      centers 0.0
                    }}
                    '''.format(spring=self.spring))
        return config

    # TODO: add method for writing replica groups


class CartesianImage(Image):
    _known_keys = Image._known_keys + ['bias', 'node', 'terminal', 'spring', 'orthogonal_spring']
    # collection of atoms; supports PDB generation (at leat in the future)
    # NAMD conf will expand to eigenvector and a RMSD object with a single harmonic force (that does not contain a center)
    # spring is a scalar

    #@classmethod
    #def from_pdb(cls, fname_pdb, atom_indices, system, frame_number=0, top=None):
    #    import mdtraj
    #    frame = mdtraj.load_frame(filename=fname_pdb, index=frame_number, top=top)
    #    conformation = frame.atom_slice(atom_indices=atom_indices)
    #
    #    return CartesianImage(colvar_def=colvar_def)

    def __init__(self, image_id, previous_image_id, previous_frame_number, node, spring, terminal=None,
                 orthogonal_spring=None, group_id=None, colvars_def=None, swarm=None, opaque=None):
        super(CartesianImage, self).__init__(image_id=image_id, previous_image_id=previous_image_id,
                                             previous_frame_number=previous_frame_number, group_id=group_id,
                                             swarm=swarm, opaque=opaque)
        self.node = node
        self.terminal = terminal
        self.spring = spring
        self.orthogonal_spring = orthogonal_spring
        self.colvars_def = colvars_def
        self.bias = 'Cartesian'

    @classmethod
    def load(cls, config, colvars_def):
        if 'bias' not in config or config['bias'] != 'Cartesian':
            raise ValueError('Attempted to construct CartesianImage from configuration entry, '
                             'but field "bias: Cartesian" is missing.')
        known_keys = super(CartesianImage, cls)._known_keys + ['bias', 'node', 'terminal', 'spring', 'orthogonal_spring']
        opaque = {key: config[key] for key in config.keys() if key not in known_keys}
        node = load_structured(config['node'])
        if 'terminal' in config:
            terminal = load_structured(config['terminal'])
        else:
            terminal = None
        if 'spring' in config:
            if isinstance(config['spring'], dict):
                spring = load_structured(config['spring'])  # named arrays only for "elliptic" bias
            else:
                spring = config['spring']
        else:
            spring = None
        if 'orthogonal_spring' in config:
            orthogonal_spring = config['spring']
        else:
            orthogonal_spring = None

        image = cls(node=node, terminal=terminal, spring=spring, orthogonal_spring=orthogonal_spring,
                    colvars_def=colvars_def, opaque=opaque, **super(CartesianImage, cls)._load_params(config))
        return image

    def dump(self):
        config = super(CartesianImage, self).dump()
        config['bias'] = 'Cartesian'
        if self.node is not None:
            config['node'] = dump_structured(self.node)
        if self.terminal is not None:
            config['terminal'] = dump_structured(self.terminal)
        if self.spring is not None:
            config['spring'] = self.spring
        if self.orthogonal_spring is not None:
            config['orthogonal_spring'] = self.orthogonal_spring
        if self.opaque is not None:
            config.update(self.opaque)
        return config

    def __eq__(self, other):
        if not super(CartesianImage, self).__eq__(other):
            return False
        for key in ['node', 'spring', 'terminal', 'orthogonal_spring']:
            if not np.array_equal(self.__dict__[key], other.__dict__[key]):
                return False
        return True

    @property
    def topology_file(self):
        if self.colvars_def is not None and 'topology_file' in self.colvars_def:
            return self.colvars_def['topology_file']
        else:  # return default
            return root() + '/setup/system.pdb'

    def triples(self, terminal=False):
        'Generates Cartesian triples for atom restraints to be used with NAMD colvars. Triples are ordered by PDB atom ID.'
        atom_id_by_field = self._read_atom_id_by_field(fname_pdb=self.topology_file)
        fields_ordered = sorted(atom_id_by_field.keys(), key=lambda field: atom_id_by_field[field])

        if terminal:
            x = self.terminal
        else:
            x = self.node
        res = []
        for field in fields_ordered:
            res.append('(' + ' , '.join([str(x) for x in x[field][0]]) + ')')
        return ' '.join(res)

    def _read_atom_id_by_field(self, fname_pdb, translate_histidines=True, use_mdtraj=True):
        'Convert atoms codes in self.fields (=self.node.dtype.names) to PDB indices as they are used the main PDB file.'
        atom_id_by_field = {}
        if use_mdtraj:
            import mdtraj
            top = mdtraj.load(fname_pdb).top
            for a in top.atoms:
                res_name = a.residue.name[0:3]  # only use three places for resname, of this is done consitently -> less confusion
                if translate_histidines and res_name in ['HSD', 'HSE', 'HSP']:
                    res_name = 'HIS'
                atom_code = '%s_%s%d' % (a.name, res_name, a.residue.resSeq)
                if atom_code in self.fields:
                    atom_id_by_field[atom_code] = a.serial
        else:
            with open(fname_pdb) as f:  # or use rather use mdtraj reader?
                lines = f.readlines()
            for line in lines:
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    atom_id = int(line[6:11])
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    if translate_histidines and res_name in ['HSD', 'HSE', 'HSP']:
                        res_name = 'HIS'
                    res_id = int(line[22:26])
                    atom_code = '%s_%s%d' % (atom_name, res_name, res_id)  # TODO: add chain ID?
                    if atom_code in self.fields:
                        atom_id_by_field[atom_code] = atom_id
        for field in self.fields:
            if field not in atom_id_by_field:
                raise RuntimeError('Atom/residue combination "%s" was not found in master pdb file "%s".'%(field,
                                                                                                           fname_pdb))
        return atom_id_by_field

    def atom_numbers(self):
        'Return PBD atom IDs for all atoms to be restained. Result is ordered increasingly, following the same convention as .triples()'
        atom_id_by_field = self._read_atom_id_by_field(fname_pdb=self.topology_file)
        return sorted(atom_id_by_field.values())

    def _isotropic_namd_conf(self, cwd, as_pdb=False):
        import textwrap
        c = textwrap.dedent('''
        colvar {{
            name RMSD
            rmsd {{
                atoms {{
                    atomnumbers {{ {atoms} }}
                    centerReference off
                    rotateReference off
                }}
                refPositions {{ {triples_node} }}
            }}
        }}

        harmonic {{
            name          RMSD
            colvars       RMSD
            forceconstant {spring}
            centers       0.0
        }}
        '''.format(atoms=' '.join(str(i) for i in self.atom_numbers()), triples_node=self.triples(terminal=False),
                   spring=self.spring))
        return c

    def _linear_namd_conf(self, cwd, as_pdb=False):
        import textwrap
        c = textwrap.dedent('''
        colvar {{
          name tangent
          eigenvector {{
            atoms {{
                atomnumbers {{ {atoms} }}
                centerReference off
                rotateReference off
            }}
            refPositions {{ {triples_node} }}
            vector {{ {triples_terminal} }}
            differenceVector on
          }}
        }}
        colvar {{
            name RMSD
            rmsd {{
                atoms {{ atomnumbers {{ {atoms} }} }}
            }}
        }}'''.format(atoms=' '.join(str(i) for i in self.atom_numbers()), triples_node=self.triples(terminal=False),
                     triples_terminal=self.triples(terminal=True)))
        return c

    def traj_to_observables(self, subdir='colvars', make=True):
        '''Extract Cartesian positions of colvars form MD trajectory and save as npy in colvar directory.

           Parameters
           ----------
           make: boolean, default=True
               Skip operation if npy file already exists and is newer (according to mtime) than the trajectory,
               much like the GNU "make" utility.
        '''
        import mdtraj
        npy_fname = '{root}/observables/{branch}_{iteration:03d}/{subdir}/{id}.npy'.format(
            root=root(), branch=self.branch, iteration=self.iteration, subdir=subdir, id=self.image_id)
        traj_fname = self.base + '.dcd'

        # the following assume that "observables" to not change
        if make and os.path.exists(npy_fname) and os.path.getmtime(npy_fname) > os.path.getmtime(traj_fname):
            warnings.warn('Skipping image %s because colvar file is already up to date.'%(self.image_id))
            return

        atom_id_by_field = self._read_atom_id_by_field(fname_pdb=self.topology_file)
        # now order by atom id
        fields_ordered = sorted(atom_id_by_field.keys(), key=lambda field: atom_id_by_field[field])
        atom_indices = sorted(atom_id_by_field.values())
        traj = mdtraj.load(traj_fname, top=self.topology_file, atom_indices=atom_indices)
        dtype = np.dtype([(name, np.float32, 3) for name in fields_ordered])
        to_save = np.core.records.fromarrays(np.transpose(traj.xyz*10., axes=(1, 0, 2)), dtype=dtype)  # nm -> Angstrom
        np.save(npy_fname,  to_save)

    #@deprecated
    #def bar_RMSD(self, other, T=303.15):
    #    import pyemma
    #    RT = 1.985877534E-3 * T  # kcal/mol
    #    id_self = '%03d_%03d' % (self.id_major, self.id_minor)
    #    id_other = '%03d_%03d' % (other.id_major, other.id_minor)
    #    p_self = self.colvars(subdir='RMSD')
    #    p_other = other.colvars(subdir='RMSD')
    #    btrajs = [np.zeros((len(p_self), 2)), np.zeros((len(p_other), 2))]
    #    btrajs[0][:, 0] = p_self[id_self][:] ** 2 * 5.0 / RT
    #    btrajs[0][:, 1] = p_self[id_other][:] ** 2 * 5.0 / RT
    #    btrajs[1][:, 0] = p_other[id_self][:] ** 2 * 5.0 / RT
    #    btrajs[1][:, 1] = p_other[id_other][:] ** 2 * 5.0 / RT
    #    ttrajs = [np.zeros(len(p_self), dtype=int), np.ones(len(p_other), dtype=int)]
    #    mbar = pyemma.thermo.MBAR()
    #    mbar.estimate((ttrajs, ttrajs, btrajs))
    #    return mbar.f_therm[0] - mbar.f_therm[1]

    #class CartesianPDBImage(Image):
        #def _isotropic_namd_conf(self, cwd, as_pdb=False):
        #     import textwrap
        #     c = textwrap.dedent('''
        #     colvar {{
        #         name RMSD
        #         rmsd {{
        #             atoms {{
        #                 atomsFile image.pdb
        #                 atomsCol B
        #                 atomsColValue 1.0
        #                 centerReference off
        #                 rotateReference off
        #             }}
        #             refPositionsFile image.pdb
        #             refPositionsCol B
        #             refPositionsColValue 1.0
        #         }}
        #     }}
        #
        #     harmonic {{
        #         name          RMSD
        #         colvars       RMSD
        #         forceconstant {spring}
        #         centers       0.0
        #     }}
        #     '''.format(spring=self.spring)
        #     self.as_pdb(cwd + 'image.pdb')
        #     raise NotImplementedError('not finished...')
        #     # return c
        # else:

def load_image(config, colvars_def):
    # if 'pdb' in node
    if 'bias' in config and config['bias'] == 'Cartesian':
        return CartesianImage.load(config=config, colvars_def=colvars_def)
    else:  # default to CompoundImage
        return CompoundImage.load(config=config, colvars_def=colvars_def)


def compound_string_to_Cartesian_string(string, atom_selection, new_iteration=None, spring=10.,
                                        colvars_template='$STRING_SIM_ROOT/setup/colvars_Cartesian.template'):
    'Convert string of CompoundImages to string of CartesianImages. Use atom MDTraj selection string to select atoms.'
    import mdtraj
    if new_iteration is None:
        new_iteration = string.iteration + 1
    s_new = string.empty_copy(iteration=new_iteration)
    for im in string.images_ordered:
        pdb = mdtraj.load(im.colvar_root + 'mean_pdb/' + im.image_id + '.pdb')
        frame = pdb.atom_slice(atom_indices=pdb.top.select(atom_selection))
        # we only use three characters for the resname; since we try to be consistent, this is more safe than
        # supporting 3 and 4 characters at the same time and working around the intricacies of different libraries
        fields = ['%s_%s%d' % (a.name, a.residue.name[0:3], a.residue.resSeq) for a in frame.top.atoms]
        atomnumbers_pdb = [a.serial for a in frame.top.atoms]
        dtype = np.dtype([(name, np.float64, 3) for name in fields])
        positions = frame.xyz[0, :, :] * 10.  # convert to Angstrom
        node = np.core.records.fromarrays(positions[:, np.newaxis, :], dtype=dtype)
        new_image = CartesianImage(
            image_id='%s_%03d_%03d_%03d' % (string.branch, new_iteration, im.id_major, im.id_minor),
            previous_image_id=im.previous_image_id,
            previous_frame_number=im.previous_frame_number, node=node, spring=spring)
        s_new.add_image(new_image)
    s_new.colvars_def = {'Cartesian': {'atomnumbers': atomnumbers_pdb, 'names': fields }}
    s_new.colvars_def['template_file'] = colvars_template
    return s_new

# colvars:
#
# system:
#   topology_file:


# node: 'IMAT-AH': [1.5, 15.0, 15.0]
# bias: compound,
# bias: Cartesian
# bias: PDB
# bias: None

# # TODO: add type that includes "atoms"
#
# class ExplicitPoint(Point):
#     def __init__(self, x, atoms_1):
#         self.x = x
#         self.atoms_1 = atoms_1
#
#     @classmethod
#     def load(cls, d):
#         # TODO: handle opaque
#         if 'atoms_1' in d:
#             atoms_1 = d['atoms_1']
#         else:
#             atoms_1 = None
#         return cls(x=load_structured(d['node']), atoms_1=atoms_1)
#
#     def write_colvar(self, prefix='refPositions'):
#         if self.atoms_1 is not None:
#             'atoms' # TODO
#         return prefix + ' ' + str(self.x).replace('[', '(').replace(']', ')')
#         # TODO: return what exactly?
#
#
# class FilePoint(Point):
#     def __init__(self, image_id):
#         self.image_id = image_id
#
#     @classmethod
#     def load(cls, d):
#         # TODO: handle opaque
#         pass
#
#     def write_colvar(self, prefix='refPositions'):
#         # TODO: cache the result ()
#         import mdtraj
#         # TODO: load dcd frame and
#
#         fname_traj =
#         frame = mdtraj.load(fname_traj, frame_idx=) # remove water and ions?
#         # TODO: save as pdb in temporary location  # TODO: set beta while writing the pdb
#         return '{prefix}File \n'.format(prefix=prefix)
#
#
#
#
#
