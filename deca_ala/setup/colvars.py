import numpy as np
import mdtraj
import os
from sarangi import root


top_fname = root() + '/setup/da.psf'
dihedrals = \
   {'psi1': [1, 4, 10, 12],  # atom indices are 1-based (NAMD compatible)
    'psi2': [12, 14, 20, 22],
    'psi3': [22, 24, 30, 32],
    'psi4': [32, 34, 40, 42],
    'psi5': [42, 44, 50, 52],
    'psi6': [52, 54, 60, 62],
    'psi7': [62, 64, 70, 72],
    'psi8': [72, 74, 80, 82],
    'psi9': [82, 84, 90, 97],
    'psi10': [97, 99, 92, 94],
    'phi2': [10, 12, 14, 20],
    'phi3': [20, 22, 24, 30],
    'phi4': [30, 32, 34, 40],
    'phi5': [40, 42, 44, 50],
    'phi6': [50, 52, 54, 60],
    'phi7': [60, 62, 64, 70],
    'phi8': [70, 72, 74, 80],
    'phi9': [80, 82, 84, 90],
    'phi10': [90, 97, 99, 92]}


def to_npy(fname_traj, fname_base_out, sim_id, **kwargs):
    traj = mdtraj.load(fname_traj, top=top_fname)
    names = sorted(dihedrals.keys())
    dtype = np.dtype([('endtoend', np.float32)] + [('cossin_'+n, np.float32, 2) for n in names])
    indices = np.array([dihedrals[n] for n in names], dtype=int) - 1
    dihedral_trajs = mdtraj.compute_dihedrals(traj, indices, periodic=True, opt=False)
    T = len(traj)
    result = np.zeros(T, dtype=dtype)
    for d,n in zip(dihedral_trajs.T, names):
        result['cossin_'+n][:, 0] = np.cos(d)
        result['cossin_'+n][:, 1] = np.sin(d)
    xyz = traj.xyz
    result['endtoend'][:] = np.linalg.norm(xyz[:, 0, :]-xyz[:, 93, :], axis=1)*10. # Angstrom
    fname_npy = fname_base_out + '.npy'
    print(fname_traj, '->', fname_npy)
    np.save(fname_npy, result)


if __name__=='__main__':
    from sarangi.observables import main_transform
    main_transform(transform_and_save=to_npy, cvname='colvars')

