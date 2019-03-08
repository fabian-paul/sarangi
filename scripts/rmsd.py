import numpy as np
import mdtraj
import os
import random
from sarangi.string_archive import load_plan, load_image, root


def mkdir(folder):
    import errno
    try:
        os.mkdir(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_bias_defs(plan, sim_id, top_fname, keep_dups):
    branch, iter_, _, _ = sim_id.split('_')
    centers = {}
    springs = {}
    indices = {}
    identifiers_seen = set()
    print('loading centers')
    mkdir('%s/observables/%s_%03d/images' % (root(), branch, int(iter_)))
    max_index = max([max(image['atoms_1']) for image in plan['images']])
    for image in plan['images']:
        #print(image)
        current_indices = np.array(image['atoms_1']) - 1
        identifier = (image['prev_image_id'], image['prev_frame_number'], tuple(image['atoms_1']), float(image['spring']['RMSD']))
        if (not keep_dups) and identifier in identifiers_seen:  # TODO: depends on the order of images, FIXME
            print('skipping duplicate bias ' + image['id'])
            continue
        identifiers_seen.add(identifier)
        # test if pdb was already cached
        randstr = ''.join(random.choice('0123456789abcdef') for _ in range(10))
        cached_fname = '%s/observables/%s_%03d/images/%s.npy' % (root(), branch, int(iter_), image['id'])
        cached_fname_tmp = '%s/observables/%s_%03d/images/%s.%s.npy' % (root(), branch, int(iter_), randstr, image['id'])
        if os.path.exists(cached_fname):
            print('loading cached version of umbrella center', image['id'])
            xyz = np.load(cached_fname)
        else:
            print('extracting umbrella center', image['id'])
            xyz = load_image(image, top_fname=top_fname).xyz[0, :, :]
            np.save(cached_fname_tmp, xyz)
            os.rename(cached_fname_tmp, cached_fname)  # for NFS
        im_id = image['id']
        centers[im_id] = xyz[current_indices, :] #.reshape(len(current_indices) * 3)
        springs[im_id] = float(image['spring']['RMSD'])
        indices[im_id] = current_indices

    print('found', len(centers), 'centers')
    return centers, springs, indices


def compute_rmsd(fname_traj, sim_id, fname_top, cv_name, centers, springs, indices, update=False):

    dtype = [(name, np.float32) for name in centers.keys()]

    max_index = max([max(idx) for idx in indices.values()])
    traj = mdtraj.load(fname_traj, top=fname_top, atom_indices=np.arange(max_index + 1))
    traj_coords = traj.xyz
    n_frames = len(traj)

    rmsds = np.zeros(n_frames, dtype=dtype)

    branch, iter_, _, _ = sim_id.split('_')
    fname_obs = '%s/observables/%s_%03d/%s/%s.npy' % (root(), branch, int(iter_), cv_name, sim_id)

    old_rmsds = None
    old_names = []
    if update and os.path.exists(fname_obs):
        temp = np.load(fname_obs)
        #print('length old, new', len(temp), len(traj))
        if len(temp) == len(traj):
            old_rmsds = temp
            old_names = temp.dtype.names

    for name, center_coords in centers.items():
        if name in old_names:
            rmsds[name][:] = old_rmsds[name]
        else:
            n_atoms = centers[name].shape[0]
            #coords = traj.atom_slice(indices[name]).xyz #.reshape((n_frames, n_atoms*3))
            assert np.all(np.array(indices[name]) <= max_index)
            coords = traj_coords[:, indices[name], :]
            rmsd = np.linalg.norm(coords - center_coords[np.newaxis, :, :], axis=(1, 2)) * n_atoms**-0.5 * 10
            rmsds[name][:] = rmsd

    print(fname_traj, '->', fname_obs)
    np.save(fname_obs, rmsds)

    #fname_out = '%s/strings/%s_%03d/rmsd/%s.dat' % (root(), branch, iter_, sim_id)
    #with open(fname_out, 'w') as f:  # wb?
    #    f.write(' '.join([d[0] for d in dtype]))
    #    f.write('\n')
    #    for t in range(rmsds.shape[0]):
    #        f.write(' '.join(['%f'%rmsds[d[0]][t] for d in dtype]))
    #        f.write('\n')


if __name__ == '__main__':
    # TODO: eventually the set of all biases wil be larger and can involve many string itertations
    import argparse
    parser = argparse.ArgumentParser(description='compute isotropic RMSD order parameter / colvar',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--id', metavar='code', default='$STRING_IMAGE_ID',
                        help='simulation ID')
    parser.add_argument('--plan', metavar='path', default='$STRING_PLAN',
                        help='(in) path to plan file')
    parser.add_argument('--top', metavar='path', default='$STRING_SIM_ROOT/setup/system.pdb',
                        help='(in) file name of topology')
    parser.add_argument('--cvname', metavar='identifier', default='rmsd',
                        help='name under which to save results')
    parser.add_argument('--keepdups', default=False, action='store_true',
                        help='keep duplicate biases')
    parser.add_argument('trajectories', metavar='paths', default='out.dcd', nargs='+',
                        help='(in) file name(s) of the trajectory file(s) to be analyzed')


    args = parser.parse_args()

    traj_dir = os.path.split(args.trajectories[0])[0]
    if traj_dir == '':
        traj_dir = '.'

    if args.id == 'auto':
        sim_id = os.path.split(os.path.splitext(args.trajectories[0])[0])[1]
    else:
        sim_id = os.path.expandvars(args.id)
        if len(args.trajectories) != 1:
            raise ValueError('For the analysis of multiple trajectories, --id must be set to auto.')
    if args.plan == 'auto':
        fname_plan = traj_dir + '/plan.yaml'
        plan = load_plan(fname_plan=fname_plan, sim_id=None)
    else:
        plan = load_plan(fname_plan=os.path.expandvars(args.plan), sim_id=None)
    if args.top == 'auto':
        fname_top = traj_dir + '/../../setup/system.pdb'
    else:
        fname_top = os.path.expandvars(args.top)

    if 'STRING_SIM_ROOT' not in os.environ:
        os.environ['STRING_SIM_ROOT'] = root()

    centers, springs, indices = load_bias_defs(plan=plan, sim_id=sim_id, top_fname=fname_top, keep_dups=args.keepdups)
    
    # here we can do a potential mass operation
    # for mass operation, only allow sim_id=='auto'!
    for fname_traj in args.trajectories:
        if args.id == 'auto':
            sim_id = os.path.split(os.path.splitext(fname_traj)[0])[1]
        else:
            sim_id = os.path.expandvars(args.id)
        print('sim_id =', sim_id)
        compute_rmsd(centers=centers, indices=indices, springs=springs, fname_traj=fname_traj, sim_id=sim_id, fname_top=fname_top, cv_name=args.cvname)
