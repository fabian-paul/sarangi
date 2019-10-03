from sarangi.observables import main_transform
from toymodel import get_top, to_structured
import mdtraj
import numpy as np


def pdb(fname_traj, fname_base_out, sim_id):
    print(fname_traj, '->', fname_base_out + '.npy')
    traj = mdtraj.load(fname_traj, top=get_top())
    np.save(fname_base_out + '.npy', to_structured(traj.xyz[:, :, 0:2]))


if __name__ == '__main__':
    main_transform(transform_and_save=pdb, cvname='colvars')
