import numpy as np
import weakref
import warnings
from .util import load_structured, dump_structured, recarray_difference, recarray_vdot, structured_to_flat, All
from .sarangi import root

__all__ = ['load_image']

# colvars:
#  -name: coformation
#   atomnumbers: [0, 1, 2]
#   type: Cartesian


class Image(object):
    _known_keys = ['id', 'prev_image_id', 'prev_image_id', 'group']

    def __init__(self, config, colvars):
        self.colvars = weakref.proxy(colvars)  # avoid cyclic dependency between string and image
        self.image_id = config['id']
        self.previous_image_id = config['prev_image_id']
        self.previous_frame_number = config['prev_image_id']
        if 'group' in config:
            self.group_id = config['group']
        else:
            self.group_id = None

    def dump(self):
        'Dump state of object to dictionary. Called by String.dump'
        config = {'id': self.image_id, 'prev_image_id': self.previous_image_id,
                  'prev_frame_number': self.previous_frame_number}
        if self.group_id is not None:
            config['group'] = self.group_id
        return config

    @property
    def bias_is_isotropic(self):
        return self.end is not None

    def namd_conf(self, cwd):
        if self.bias_is_isotropic:
            return self._isotropic_namd_conf(cwd)
        else:
            return self._linear_namd_conf(cwd)

    @property
    def fields(self):
        return self.node.dtype.fields


class CompoundImage(Image):
    # collection of other CVs; does not support and never will support PDB generation
    # NAMD conf will expand to customfunction and multiple harmonic forces
    # spring can eb a scarlar or a vector
    def __init__(self, config, colvars):
        self.opaque = None
        super(CompoundImage, self).__init__(config=config, colvars=colvars)
        known_keys = super(CompoundImage)._known_keys + ['node', 'end', 'spring']
        self.opaque = {key: config[key] for key in config.keys() if key not in known_keys}

        # These are kept in the subclass to allow future extension to Images that do not contain coordinated and
        # only refer to IDs of completed simulations
        self.node = load_structured(config['node'])
        if 'end' in config:
            self.end = load_structured(config['end'])
        else:
            self.end = None
        if 'spring' in config:
            self.spring = config['spring']
        else:
            self.spring = None

    def dump(self):
        config = super(CompoundImage).dump()
        if self.node is not None:
            config['node'] = dump_structured(self.node)
        if self.end is not None:
            config['node'] = dump_structured(self.node)
        if self.spring is not None:
            config['spring'] = dump_structured(self.spring)  # TODO: if the bias is linear, spring is just a single number
        config.update(self.opaque)
        return

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
        config = ''
        a = self.point
        b = self.end
        a_minus_b = recarray_difference(a, b)
        norm = np.linalg.norm(structured_to_flat(a_minus_b))
        const = recarray_vdot(a, a_minus_b) / norm
        expr = ['%f'%const]
        # see equation in appendix to the paper
        for field in a_minus_b.dtype.names:
            for i in range(3):  # TODO: find the correct dimenion
                expr.append('{field}{dim}*{factor}'.format(field=field, dim=i+1, factor=-a_minus_b[i]/norm))

        config += '''config {{
                       name tangent
                       customFunction {{{expression}}}
                  '''.format(expression='+'.join(expr))
        # now repeat all colvar definitions but with the name moved inside (see colvar documentation)
        for field in self.fields:
            # name is inside in contrast to the "normal" colvar definition
            config += '''distanceVec {{
                           name {field}
                           group2 {{ atomnumbers {{ {atoms} }} }}
                           group1 {{ dummyAtom ( 0.0 , 0.0 , 0.0 ) }}
                         }
                         '''.format(field=field, atoms=' '.join(self.colvars[field].atoms))
        # add harmonic force
        spring_value = self.spring[a_minus_b.dtype.names[0]]  # just pick the first spring value TODO: replace with average?
        config += '''harmonic {{
                       name tanget_restraint
                       colvars tangent
                       forceconstant {spring}
                       centers 0.0 }}
                       '''.format(spring=spring_value)
        return config

    # TODO: method for writing replica groups

    def set_end(self, point):
        if sorted(point.dtype.names) != sorted(self.node.dtype.names):
            raise ValueError('point does not match field singature')
        self.end = point.copy()


class CartesianImage(Image):  # TODO: add more subclasses (FileImage that refers to PDB, etc.)
    # collection of atoms; supports PDB generation (at leat in the future)
    # NAMD conf will expand to eigenvector and a RMSD object with a single harmonic force (that does not contain a center)
    # spring is a scalar

    @classmethod
    def from_pdb(cls, fname_pdb):
        pass

    def __init__(self, config, colvars):
        super(CartesianImage, self).__init__(config=config, colvars=colvars)
        known_keys = super(CompoundImage)._known_keys + ['node', 'end', 'spring']
        self.opaque = {key: config[key] for key in config.keys() if key not in known_keys}

        self.node = load_structured(config['node'])
        if 'end' in config:
            self.end = load_structured(config['end'])
        else:
            self.end = None
        if 'spring' in config:
            self.spring = config['spring']
        else:
            self.spring = None

    def triples(self, do_end=False):
        atom_id_by_field = self._read_atom_id_by_field(root() + '/setup/system.pdb')
        fields_ordered = sorted(atom_id_by_field.keys(), key=lambda field: atom_id_by_field[field])

        if do_end:
            x = self.node
        else:
            x = self.end
        res = []
        for field in fields_ordered:
            res.append('(' + ' '.join([str(x) for x in x[field]]) + ')')
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
                raise RuntimeError('Atom/residue combination "%s" was not found in master pdb file "%s".'%(field, fname_pdb))
        return atom_id_by_field

    def atom_numbers(self):
        atom_id_by_field = self._read_atom_id_by_field(root() + '/setup/system.pdb')
        return sorted(atom_id_by_field.values())


    def _isotropic_namd_conf(self, cwd, as_pdb=False):
        if as_pdb:
            c = '''
            colvar {{
                name RMSD
                rmsd {{
                    atoms {{
                        atomsFile image.pdb
                        atomsCol B
                        atomsColValue 1.0
                        centerReference off
                        rotateReference off
                    }}
                    refPositionsFile image.pdb
                    refPositionsCol B
                    refPositionsColValue 1.0
                }}
            }}
            
            harmonic {{
                name          RMSD
                colvars       RMSD
                forceconstant {spring}
                centers       0.0
            }}
            '''.format(spring=self.spring)
            self.as_pdb(cwd + 'image.pdb')
            raise NotImplementedError('not finished...')
            # return c
        else:
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
            '''.format(atoms=self.atom_numbers(), triples_node=self.triples(do_end=False), spring=self.spring)
            return c


    def _linear_namd_conf(self, cwd, as_pdb=False):
        c = '''
        colvar {{
          name tangent
          eigenvector {{
            atoms {{ atomnumbers {{ {atoms} }} }}
            refPositions {triples_node}
            vector {triples_end}
            differenceVector on
          }}
        }}
        colvar {{
            name RMSD
            rmsd {{
                atoms {{ atomnumbers {{ {atoms} }} }}
            }}
        }}'''.format(atoms=self.atom_numbers(), triples_node=self.triples(do_end=False), triples_end=self.triples(do_end=True))
        return c


def load_image(config, colvars):
    if 'type' in config and config['type'] == 'cartesian':  # TODO: find some solution!
        return CartesianImage(config=config, colvars=colvars)
    else:
        return CompoundImage(config=config, colvars=colvars)

# node: 'IMAT-AH': [1.5, 15.0, 15.0]

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
