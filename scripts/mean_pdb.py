from sarangi.observables import main_transform
from sarangi import root
import mdtraj
import numpy as np


fname_pdb = root() + '/setup/system.pdb'


def pdb(fname_traj, fname_base_out, sim_id):
    pdb = mdtraj.load(fname_pdb)
    top = pdb.top
    atom_indices = top.select('not water and not element Na and not element K and not element Cl')
    traj = mdtraj.load(fname_traj, top=top, atom_indices=atom_indices)
    frame = traj[0]
    frame.xyz[0, :, :] = np.mean(traj.xyz, axis=0)
    print(fname_traj, '->', fname_base_out + '.pdb')
    frame.save_pdb(fname_base_out + '.pdb')


if __name__ == '__main__':
    main_transform(transform_and_save=pdb, cvname='mean_pdb')