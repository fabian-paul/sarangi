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
    _known_keys = ['id', 'prev_image_id', 'prev_frame_number', 'group']

    def __init__(self, image_id, previous_image_id, previous_frame_number, group_id, opaque):
        self.image_id = image_id
        self.previous_image_id = previous_image_id
        self.previous_frame_number = previous_frame_number
        self.group_id = group_id
        self.opaque = opaque
        self._colvars = {}
        self._x0 = {}

    @classmethod
    def _load_params(cls, config):
        image_id = config['id']
        previous_image_id = config['prev_image_id']
        previous_frame_number = config['prev_frame_number']
        if 'group' in config:
            group_id = config['group']
        else:
            group_id = None
        return {'image_id': image_id, 'previous_image_id': previous_image_id,
                'previous_frame_number': previous_frame_number, 'group_id': group_id}

    def dump(self):
        'Dump state of object to dictionary. Called by String.dump'
        config = {'id': self.image_id, 'prev_image_id': self.previous_image_id,
                  'prev_frame_number': self.previous_frame_number}
        if self.group_id is not None:
            config['group'] = self.group_id
        return config

    @property
    def bias_is_isotropic(self):
        return self.terminal is not None

    def namd_conf(self, cwd):
        if self.bias_is_isotropic:
            return self._isotropic_namd_conf(cwd)
        else:
            return self._linear_namd_conf(cwd)

    @property
    def fields(self):
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
        env['STRING_ARCHIVIST'] = os.path.dirname(__file__) + '/string_archive.py'
        env['STRING_SARANGI_SCRIPTS'] = os.path.dirname(__file__) + '/../scripts'
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
        if (subdir, tuple(fields)) in self._colvars:
            return self._colvars[(subdir, tuple(fields))]
        else:
            folder = '{root}/observables/{branch}_{iteration:03d}/'.format(
                root=root(), branch=self.branch, iteration=self.iteration)
            base = '{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}'.format(
                branch=self.branch, iteration=self.iteration, id_minor=self.id_minor, id_major=self.id_major)
            pcoords = Colvars(folder=folder + subdir, base=base, fields=fields)
            if memoize:
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

    @staticmethod
    def potential(x, node, spring):
        'Compute the bias potential parametrized by node and spring evaluated along the possibly multidimensional order parameter x.'
        u = np.zeros(x._colvars.shape[0])
        for name in node.dtype.names:
            # print('x', x[name].shape)
            # print('spring', spring[name].shape)
            # print('node', node[name].shape)
            # u_part = 0.5 * spring[name] * np.linalg.norm(x[name] - node[name], axis=1)**2
            # print(x[name].shape, node[name].shape, (x[name] - node[name]).shape)
            u_part = 0.5 * float(spring[name]) * np.sum((x[name] - node[name]) ** 2, axis=1)
            # print(u_part.shape)
            assert u_part.ndim == 1
            # print('pot', u_part.shape, u_part.ndim)
            # if u is None:
            #    u = u_part
            # else:
            u += u_part
        return u

    def bar(self, other, subdir='colvars', T=303.15):
        'Compute thermodynamic free energy difference between this window (self) and other using BAR.'
        import pyemma
        from pyemma.util.contexts import settings
        RT = 1.985877534E-3 * T  # kcal/mol
        fields = list(self.node.dtype.names)
        my_x = self.colvars(subdir=subdir, fields=fields)
        other_x = other.colvars(subdir=subdir, fields=fields)
        btrajs = [np.zeros((len(my_x), 2)), np.zeros((len(other_x), 2))]
        btrajs[0][:, 0] = Image.potential(my_x, self.node, self.spring) / RT
        btrajs[0][:, 1] = Image.potential(my_x, other.node, other.spring) / RT
        btrajs[1][:, 0] = Image.potential(other_x, self.node, self.spring) / RT
        btrajs[1][:, 1] = Image.potential(other_x, other.node, other.spring) / RT
        ttrajs = [np.zeros(len(my_x), dtype=int), np.ones(len(other_x), dtype=int)]
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
            structured_to_flat(mean).reshape(-1) - structured_to_flat(o, fields=list(mean.dtype.names)).reshape(-1),
            ord=ord) * (n_atoms ** -0.5)

    def set_terminal_point(self, point):
        if sorted(point.dtype.names) != sorted(self.node.dtype.names):
            raise ValueError('point does not match field signature')
        self.terminal = point.copy()


class CompoundImage(Image):
    _known_keys = Image._known_keys + ['bias', 'node', 'terminal', 'spring', 'orthogonal_spring']

    # collection of other CVs; does not support and never will support PDB generation
    # NAMD conf will expand to customfunction and multiple harmonic forces
    # spring can eb a scarlar or a vector
    def __init__(self, image_id, previous_image_id, previous_frame_number, group_id,
                 node, spring, terminal=None, orthogonal_spring=None, colvars_def=None, opaque=None):
        super(CompoundImage, self).__init__(image_id=image_id, previous_image_id=previous_image_id,
                                            previous_frame_number=previous_frame_number, group_id=group_id,
                                            opaque=opaque)

        self.node = node
        self.terminal = terminal
        self.spring = spring
        self.orthogonal_spring = orthogonal_spring
        self.colvars_def = colvars_def
        self.bias = 'compound'

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
            if self.__dict__[key] != other.__dict__[key]:
                return False
        return True

    def _isotropic_namd_conf(self, cwd):
        # for fun we could redefine all the colvars form the information in the plan file
        config = ''
        for restraint_name in self.spring.dtype.names:
            spring_value = self.spring[restraint_name]
            if self.node is not None and restraint_name in self.node.dtype.names:
                center_value = self.node[restraint_name]
            else:
                warnings.warn('Spring constant was defined but no umbrella center. Using the default 0.0.')
                center_value = '0.0'
            spring_value_namd = str(spring_value).replace('[', '(').replace(']', ')')
            center_value_namd = str(center_value).replace('[', '(').replace(']', ')')
            config += '''harmonic {{
                           name {restraint_name}_restraint
                           colvars {restraint_name}
                           forceconstant {spring}
                           centers {center}
                         }}
                         '''.format(restraint_name=restraint_name,
                                    center=center_value_namd,
                                    spring=spring_value_namd)
        return config

    def _linear_namd_conf(self, cwd):
        if self.colvars_def is None or 'compound' not in self.colvars_def:
            raise RuntimeError('MD setup with linear order parameter (linearly directed bias) was requested but, plan file '
                               'does not contain definitions of the collective variables. Can\'t create MD setup file.')
        config = ''
        a = self.node
        b = self.terminal
        a_minus_b = recarray_difference(a, b)
        norm = np.linalg.norm(structured_to_flat(a_minus_b))
        const = recarray_vdot(a, a_minus_b) / norm
        expr = ['%f'%const]
        # see equation in appendix to the paper
        for field in a_minus_b.dtype.names:
            for i in range(3):  # TODO: find the correct dimension from numpy array
                expr.append('{field}{dim}*{factor}'.format(field=field, dim=i+1, factor=-a_minus_b[i]/norm))

        config += '''config {{
                       name tangent
                       customFunction {{{expression}}}
                  '''.format(expression='+'.join(expr))
        # now repeat all colvar definitions but with the name moved inside (see colvar documentation)
        for colvar in self.colvars_def['compound']:
            # name is inside in contrast to the "normal" colvar definition
            if 'type' in colvar and colvar['type'] != 'com':
                warnings.warn('Found colvars def of type %s, don\'t know how to handle.')
            config += '''distanceVec {{
                           name {field}
                           group2 {{ atomnumbers {{ {atoms} }} }}
                           group1 {{ dummyAtom ( 0.0 , 0.0 , 0.0 ) }}
                         }
                         '''.format(field=colvar['name'], atoms=' '.join(colvar['atomnumbers']))
        # add harmonic force
        if not isinstance(self.spring, float):
            raise ValueError('Field spring is of type %s, expecting float. Don\'t know how to handle this for applying'
                             'a unidirectional force.' % (type(self.spring)))
        config += '''harmonic {{
                       name tanget_restraint
                       colvars tangent
                       forceconstant {spring}
                       centers 0.0 }}
                       '''.format(spring=self.spring)
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
                 orthogonal_spring=None, group_id=None, colvars_def=None, opaque=None):
        super(CartesianImage, self).__init__(image_id=image_id, previous_image_id=previous_image_id,
                                             previous_frame_number=previous_frame_number, group_id=group_id,
                                             opaque=opaque)
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
                    colvars_def=colvars_def, opaque=opaque, **super(CompoundImage)._load_params(config))
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
            if self.__dict__[key] != other.__dict__[key]:
                return False
        return True

    @property
    def topology_file(self):
        return root() + '/setup/system.pdb'

    def triples(self, terminal=False):
        atom_id_by_field = self._read_atom_id_by_field(fname_pdb=self.topology_file)
        fields_ordered = sorted(atom_id_by_field.keys(), key=lambda field: atom_id_by_field[field])

        if terminal:
            x = self.terminal
        else:
            x = self.node
        res = []
        for field in fields_ordered:
            res.append('(' + ' , '.join([str(x) for x in x[field]]) + ')')
        return ' '.join(res)

    def _read_atom_id_by_field(self, fname_pdb):
        atom_id_by_field = {}
        with open(fname_pdb) as f:  # or use rather use mdtraj reader?
            lines = f.readlines()
        for line in lines:
            if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                atom_id = int(line[6:11])
                atom_name = line[12:16].strip()
                res_name = line[17:21].strip()
                res_id = int(line[22:26])
                field = '%s%d_%s' % (res_name, res_id, atom_name)  # TODO: add chain ID?
                atom_id_by_field[field] = atom_id
        for field in self.fields:
            if field not in atom_id_by_field:
                raise RuntimeError('Atom/residue combination "%s" was not found in master pdb file "%s".'%(field,
                                                                                                           fname_pdb))
        return atom_id_by_field

    def atom_numbers(self):
        atom_id_by_field = self._read_atom_id_by_field(fname_pdb=self.topology_file)
        return sorted(atom_id_by_field.values())

    def _isotropic_namd_conf(self, cwd, as_pdb=False):
        # if as_pdb:
        #     c = '''
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
        c = '''
        colvar {{
            name RMSD
            rmsd {{
                atoms {{
                    atomnumbers {{ {atoms} }}
                }}
                refPositions {triples_node}
            }}
        }}

        harmonic {{
            name          RMSD
            colvars       RMSD
            forceconstant {spring}
            centers       0.0
        }}
        '''.format(atoms=self.atom_numbers(), triples_node=self.triples(terminal=False), spring=self.spring)
        return c

    def _linear_namd_conf(self, cwd, as_pdb=False):
        c = '''
        colvar {{
          name tangent
          eigenvector {{
            atoms {{ atomnumbers {{ {atoms} }} }}
            refPositions {triples_node}
            vector {triples_terminal}
            differenceVector on
          }}
        }}
        colvar {{
            name RMSD
            rmsd {{
                atoms {{ atomnumbers {{ {atoms} }} }}
            }}
        }}'''.format(atoms=self.atom_numbers(), triples_node=self.triples(terminal=False),
                     triples_terminal=self.triples(terminal=True))
        return c

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


def load_image(config, colvars_def):
    # if 'pdb' in node
    if 'bias' in config and config['bias'] == 'cartesian':
        return CartesianImage.load(config=config, colvars_def=colvars_def)
    else:  # default to CompoundImage
        return CompoundImage.load(config=config, colvars_def=colvars_def)

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