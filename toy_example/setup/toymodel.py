import numpy as np
import os
import mdtraj
import tempfile
from sarangi.util import mkdir
from typing import Union, Tuple

# TODOs:
# Have multiple 2D "atoms"? Eg. heterodynmer in the same 2d potential? (has director) wants to align in the channel.
# make faster (extension module?)
# (low prio) handle restraint forces properly

__all__ = ['to_structured', 'get_top', 'DEFAULT_N_ATOMS']

DEFAULT_N_ATOMS = 5  # 1 metastable dynamics + 4 noise


def get_top(n_atoms: int=DEFAULT_N_ATOMS) -> mdtraj.Topology:
    f = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
    with f:
        for i in range(n_atoms):
            f.write(('ATOM    %03d  C   ALA     0       0.000   0.000   0.000  1.00  0.00\n' % i).encode())
    top = mdtraj.load(f.name).top
    os.unlink(f.name)
    return top


def to_structured(data: np.ndarray, separate: bool=True) -> np.ndarray:
    if not data.ndim == 3:
        raise ValueError('data.ndim must be 3')
    n_atoms = data.shape[1]
    if separate:
        dim = data.shape[2]
        alphabet = ['X', 'Y'] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        dtype = np.dtype([(alphabet[i], np.float32, 1) for i in range(n_atoms*dim)])
        structured = np.core.records.fromarrays([data[:, i, j] for i in range(n_atoms) for j in range(dim)] , dtype=dtype)
    else:
        alphabet = ['X'] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        dtype = np.dtype([(alphabet[i], np.float32, data.shape[2]) for i in range(n_atoms)])
        structured = np.core.records.fromarrays([data[:, i, :] for i in range(n_atoms)], dtype=dtype)
    return structured


# from file mullermsm/lib/toymodel.py
# https://github.com/rmcgibbo/mullermsm/tree/master/lib
def mueller_potential(x: Union[np.ndarray, float], y: Union[np.ndarray, float]) -> Union[np.ndarray, np.float]:
    """Mueller potential

    Parameters
    ----------
    x : {float, np.ndarray}
        X coordinate. Can be either a single number or an array. If you supply
        an array, x and y need to be the same shape.
    y : {float, np.ndarray}
        Y coordinate. Can be either a single number or an array. If you supply
        an array, x and y need to be the same shape.
    Returns
    -------
    potential : {float, np.ndarray}
        Potential energy. Will be the same shape as the inputs, x and y.

    Reference
    ---------
    Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
    """

    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    value = 0
    for j in range(0, 4):
        value += AA[j] * np.exp(aa[j] * (x - XX[j]) ** 2 + \
                                bb[j] * (x - XX[j]) * (y - YY[j]) + cc[j] * (y - YY[j]) ** 2)

    return value


# from http://www.acmm.nl/molsim/users/bolhuis/tps/content/exercise.pdf
def bistable_potential(x: Union[np.ndarray, float], y: Union[np.ndarray, float]) -> Union[np.ndarray, np.float]:
    r'''Bistable potential used in the publications about the string method by Pan and Roux.

    Notes
    -----
    Minima at (0.96, 0.06) and (-0.98, 0.06)
    Saddle point at (-1.96, ?)
    '''
    return -np.exp(-(x-1)**2 - y**2) - np.exp(-(x+1)**2 - y**2) \
           + 5*np.exp(-0.32*(x**2 + y**2 + 20*(x + y)**2)) \
           + 32/1875*(x**4 + y**4) + 2/15*np.exp(-2-4*y)


def plot_bistable_potential(ax=None):
    import matplotlib.pyplot as plt
    x_pot = np.linspace(-2.5, 2.5)
    y_pot = np.linspace(-1, 3)
    X, Y = np.meshgrid(x_pot, y_pot)
    u = bistable_potential(X, Y)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.contour(u, extent=[x_pot[0], x_pot[-1], y_pot[0], y_pot[-1]])
    ax.set_aspect('equal')
    return ax


def add_noise_atoms(data: np.ndarray, atom_indices: int=slice(1, DEFAULT_N_ATOMS)) -> np.ndarray:
    data[:, atom_indices, :] = np.random.randn(*data.shape)[:, atom_indices, :]
    return data


def load_frame(fname: str, n_atoms: int=DEFAULT_N_ATOMS, frame_no: int=0) -> np.ndarray:
    frame = mdtraj.load(fname, top=get_top(n_atoms=n_atoms)).xyz[frame_no, :, 0:2]
    return frame


def save(fname: str, frames: np.ndarray):
    T = frames.shape[0]
    n_atoms = frames.shape[1]
    assert frames.shape[2] == 2
    xyz = np.zeros((T, n_atoms, 3))
    xyz[:, :, 0:2] = frames
    time = np.zeros(T)
    unitcell_lengths = np.zeros((T, 3))
    unitcell_angles = np.zeros((T, 3))
    traj = mdtraj.Trajectory(xyz=xyz, topology=get_top(), time=time, unitcell_lengths=unitcell_lengths,
                             unitcell_angles=unitcell_angles)
    traj.save(fname)


def propagate(x0: float, y0: float, n_steps=0, potential=bistable_potential) -> Tuple[float, float]:
    x, y = x0, y0
    e = potential(x, y)
    for t in range(n_steps):
        x_prime = x + (np.random.rand() - .5)*0.05
        y_prime = y + (np.random.rand() - .5)*0.05
        e_prime = potential(x_prime, y_prime)
        # evaluate MCMC criterion
        if np.exp(e - e_prime) > np.random.rand():
            # accept
            x = x_prime
            y = y_prime
            e = e_prime
    return x, y


def interp(a: float, b: float, r: float) -> float:
    return a + r*(b - a)


def resolve(x: str, cast: object=str, default=None):
    x = os.path.expandvars(x)
    if '$' in x:
        return default
    else:
        return cast(x)


if __name__=='__main__':
    import argparse
    from sarangi.mcmc import propagate_bistable

    # TODO: handling of node and spring...

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', metavar='file', default='$STRING_PREV_ARCHIVE.dcd',
                        help='File name of input frame.')
    parser.add_argument('--output', metavar='file', default='$STRING_ARCHIVE.dcd',
                        help='File name for output points (final swarm points)')
    parser.add_argument('--output_init', metavar='file', default='${STRING_ARCHIVE}_init.dcd',
                        help='File name for output points (initial swarm points)')
    parser.add_argument('--frame', metavar='number', default='$STRING_PREV_FRAME_NUMBER',
                        help='frame number tp continue simulation from')
    parser.add_argument('--mueller', action='store_true', help='Use Mueller potential instead of bistable potential.')
    parser.add_argument('--steps', metavar='number', help='number of integration steps', default=100)
    parser.add_argument('--swarm', metavar='number', help='number of trajectories in swarm', default=100)
    parser.add_argument('--seed', metavar='integer', help='random number seed', default='$STRING_RANDOM')
    parser.add_argument('--init', action='store_true', help='start from initial point number n on the segment connecting (-0.5, 1.5) and (0.6, 0.0)')
    parser.add_argument('--colvars', default='$STRING_OBSERVABLES_BASE/colvars/$STRING_IMAGE_ID',
                        help='Base name for colvar file (without extention)')
    parser.add_argument('--colvars_init', default='$STRING_OBSERVABLES_BASE/colvars_init/$STRING_IMAGE_ID',
                        help='Base name for init colvar file (without extention)')
    args = parser.parse_args()

    input = resolve(args.input)
    output = resolve(args.output)
    output_init = resolve(args.output_init)
    frame = resolve(args.frame, cast=int)
    seed = resolve(args.seed, cast=int)
    swarm_size = int(args.swarm)
    colvars = resolve(args.colvars)
    colvars_init = resolve(args.colvars_init)

    if seed is not None:
        print('random seed', seed % (2**32 - 1))
        np.random.seed(seed % (2**32 - 1))

    initial_points = np.zeros(shape=(swarm_size, DEFAULT_N_ATOMS, 2))
    final_points = np.zeros(shape=(swarm_size, DEFAULT_N_ATOMS, 2))

    if args.init:
        for i in range(swarm_size):
            if args.mueller:
                x = interp(-0.5, 0.6, i / swarm_size)
                y = interp( 1.5, 0.0, i / swarm_size)
            else:
                x = interp(-0.98, 0.96, i / swarm_size)
                #y = interp( 0.06, 0.06, i / swarm_size)
                y = interp(0.3, 0.3, i / swarm_size)
            final_points[i, 0, 0] = x
            final_points[i, 0, 1] = y
    else:
        # normal propagation
        if input is None:
            raise ValueError('--input argument is missing')
        frame = load_frame(fname=input, frame_no=frame)
        initial_points[:, :, :] = frame[np.newaxis, :, :]
        for i in range(swarm_size):
            if args.mueller:
                final_points[i, 0, :] = propagate(frame[0, 0], frame[0, 1], int(args.steps), mueller_potential)
            else:
                #final_points[i, 0, :] = propagate_bistable(frame[0, 0], frame[0, 1], int(args.steps))
                final_points[i, 0, :] = propagate(frame[0, 0], frame[0, 1], int(args.steps), bistable_potential)

    # add orthogonal noisy dynamics (we can do it here after the dynamics, beause it's completely independent)
    if not args.init:
        initial_points = add_noise_atoms(initial_points)
        final_points = add_noise_atoms(final_points)

    if output is None:
        raise ValueError('--output argument is missing')
    mkdir(os.path.dirname(output))
    save(fname=output, frames=final_points)

    if not args.init:
        if output_init is None:
            raise ValueError('--output_init argument is missing')
        else:
            mkdir(os.path.dirname(output_init))
            save(fname=output_init, frames=initial_points)
        # save colvars
        mkdir(os.path.dirname(colvars))
        np.save(colvars + '.npy', to_structured(final_points))
        mkdir(os.path.dirname(colvars_init))
        np.save(colvars_init + '.npy', to_structured(initial_points))
