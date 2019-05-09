import warnings
from .util import load_structured, dump_structured


def load_image(me, top_fname):
    prev_id = me['prev_image_id']
    branch, iteration, image_major, image_minor = prev_id.split('_')
    fname_dcd = '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{image_major:03d}_{image_minor:03d}.dcd'.format(
        root=root(), branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
    )
    frame_index = me['prev_frame_number']
    frame = mdtraj.load_frame(fname_dcd, index=frame_index, top=top_fname)
    return frame



def write_image(me, fname_dest_pdb, top_fname):
    if 'atoms_1' not in me:
        warnings.warn('String archivist was instructed to write an image file but the plan does not contain an atom '
                      'reference. Skipping this step.')
        return
    frame = load_image(me, top_fname)
    atoms = me['atoms_1']
    b_factors = [1 if (i + 1) in atoms else 0 for i in range(frame.top.n_atoms)]
    frame.save_pdb(fname_dest_pdb, bfactors=b_factors)

class Point(object):
    pass

# TODO: add type that includes "atoms"

class ExplicitPoint(Point):
    def __init__(self, x, atoms_1):
        self.x = x
        self.atoms_1 = atoms_1

    @classmethod
    def load(cls, d):
        # TODO: handle opaque
        if 'atoms_1' in d:
            atoms_1 = d['atoms_1']
        else:
            atoms_1 = None
        return cls(x=load_structured(d['node']), atoms_1=atoms_1)

    def write_colvar(self, prefix='refPositions'):
        if self.atoms_1 is not None:
            'atoms' # TODO
        return prefix + ' ' + str(self.x).replace('[', '(').replace(']', ')')
        # TODO: return what exactly?


class FilePoint(Point):
    def __init__(self, image_id):
        self.image_id = image_id

    @classmethod
    def load(cls, d):
        # TODO: handle opaque
        pass

    def write_colvar(self, prefix='refPositions'):
        # TODO: cache the result ()
        import mdtraj
        # TODO: load dcd frame and

        fname_traj =
        frame = mdtraj.load(fname_traj, frame_idx=) # remove water and ions?
        # TODO: save as pdb in temporary location  # TODO: set beta while writing the pdb
        return '{prefix}File \n'.format(prefix=prefix)


def load_point(d):
    #if 'type' not in d:  # default assume ExplicitPoint
    #    ExplicitPoint.load(d)
    if 'image_id' in d:
        return FilePoint.load(d)
    elif 'node' in d:
        return ExplicitPoint.load(d)
    else:
        raise Exception('Point type not recognized')


class Bias(object):
    pass

## TODO: correct handling of recarray format

class IsotropicBias(Bias):
    def __init__(self, node, spring, opaque=None):
        self.node = node
        self.spring = spring
        self.opaque = opaque
        # TODO: include optional atoms_1 here

    @classmethod
    def load(cls, d):
        opaque = {key: d[key] for key in d.keys() if key not in ['node', 'spring']}
        return cls(node=load_point(d['node']), spring=load_structured(d['spring']), opaque=opaque)

    def write_colvars(self):
        config = ''
        for restraint_name, spring_value in self.spring.items():
            if self.node is not None and restraint_name in self.node:
                center_value = self.node[restraint_name]
            else:
                warnings.warn('Spring constant was defined but no umbrella center. Using the default 0.0.')
                center_value = '0.0'
            center_value_namd = str(center_value).replace('[', '(').replace(']', ')')
            config += 'harmonic {{\n' \
                      'name {restraint_name}_restraint\n' \
                      'colvars {restraint_name}\n' \
                      'forceconstant {spring}\n' \
                      'centers {center}\n}}\n'.format(restraint_name=restraint_name,
                                                      center=center_value_namd,
                                                      spring=spring_value)
        return config

    def write_colvars_re(self, image_index):
        r'''image index refers to the internal numbering in a re group'''
        config = '  %d {\n    return {' % image_index
        for restraint_name, spring_value in self.spring.items():
            if self.node is not None and restraint_name in self.node:
                center_value = self.node[restraint_name]
            else:
                warnings.warn('Spring constant was defined but no umbrella center. Using the default 0.0.')
                center_value = '0.0'
            center_value_namd = str(center_value).replace('[', '(').replace(']', ')')
            config += '    {restraint_name}_restraint {{ centers {center} forceconstant {spring} }}\\\n'.format(
                restraint_name=restraint_name,
                center=center_value_namd,
                spring=spring_value)
        config += '    }}\n'
        return config

    def write_openmm(self):
        pass

    def write(self):
        d = {}
        if self.node is not None:
            d['node'] = dump_structured(self.node)
        if self.spring is not None:
            d['spring'] = dump_structured(self.spring)
        if self.opaque is not None:
            d.update(self.opaque)
        return d


class LinearBias(Bias):
    def __init__(self, node, spring, end, opaque=None):
        self.node = node
        self.spring = spring
        self.end = end
        self.opaque = opaque
        # TODO: include optional atoms_1 here

    @classmethod
    def load(cls, d):
        opaque = {key: d[key] for key in d.keys() if key not in ['node', 'end', 'spring']}
        return cls(node=load_point(d['node']), spring=load_structured(d['spring']), end=load_point(d['end']),
                   opaque=opaque)

    def write_colvar(self):
        # TODO: these are
        config = 'covar {{\n' \
                 'name string_tangent\n' \
                 'eigenvector {{\n' \
                 'refPositions {node}' \
                 'vector {end}\n' \
                 'differenceVector on\n' \
                 '}}\n}}\n'.format(node=self.node, end=self.end)

        config += 'harmonic {{\n' \
                  'name string_tangent_restraint\n' \
                  'colvars string_tangent\n' \
                  'forceconstant {spring}\n' \
                  'centers 0\n}}\n'.format(spring=self.spring)
        return config

    def write_openmm(self):
        pass

    def write(self):
        d = {}
        if self.node is not None:
            d['node'] = dump_structured(self.node)
        if self.spring is not None:
            d['spring'] = dump_structured(self.spring)
        if self.end is not None:
            d['spring'] = dump_structured(self.end)
        if self.opaque is not None:
            d.update(self.opaque)
        return d


def load_bias(d):
    if 'type' not in d:
        return IsotropicBias.load(d)
    elif d['type'] == 'linear':
        return IsotropicBias.load(d)
    elif d['type'] == 'isotropic':
        return LinearBias.load(d)
    else:
        raise Exception('Bias type not recognized.')
