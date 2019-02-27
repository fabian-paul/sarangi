from sarangi.observables import main_transform
from sarangi import root
import mdtraj


fname_pdb = root() + '/setup/system.pdb'


def pdb(fname_traj, fname_base_out, sim_id):
    pdb = mdtraj.load(fname_pdb)
    top = pdb.top
    atom_indices = top.select('not water and not element Na and not element K and not element Cl')
    frame = mdtraj.load(fname_traj, top=top, atom_indices=atom_indices)[-1]
    print(fname_traj, '->', fname_base_out + '.pdb')
    frame.save_pdb(fname_base_out + '.pdb')


if __name__ == '__main__':
    main_transform(transform_and_save=pdb, cvname='pdb')