import subprocess
from . import load, root, is_sim_id, String


def main_transform(transform_and_save, cvname='colvars'):
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Compute ' + cvname,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--id', metavar='code', default='$STRING_IMAGE_ID',
                        help='simulation ID. If this is left empty or this points to an undefined environment variable,'
                             ' try to infer from file name.')
    parser.add_argument('trajectories', metavar='paths', default='out.dcd', nargs='+',
                        help='(in) file name(s) of the trajectory file(s) to be analyzed')
    parser.add_argument('--cvname', metavar='identifier', default=cvname,
                        help='name under which to save results')

    args = parser.parse_args()

    global_sim_id = os.path.expandvars(args.id)

    if '$' not in global_sim_id and global_sim_id != '':
        if len(args.trajectories) > 1:
            raise ValueError(
                'You specified the concrete simulation id %s but supplied multiple input trajectories. '
                'This is inconsistent.' % global_sim_id)

    for fname_traj in args.trajectories:
        if '$' in global_sim_id or global_sim_id == '':
            base_name = os.path.split(os.path.splitext(fname_traj)[0])[1]
            if is_sim_id(base_name):
                sim_id = base_name
            else:
                raise ValueError('Could not infer id from trajectory name %s.' % base_name)
        else:
            sim_id = global_sim_id
        branch, iter_, _, _ = sim_id.split('_')
        fname_base_out = '%s/observables/%s_%03d/%s/%s' % (root(), branch, int(iter_), args.cvname, sim_id)
        transform_and_save(fname_traj=fname_traj, fname_base_out=fname_base_out, sim_id=sim_id)


def main_update(image_id=None, ignore_colvar_traj=False, iteration=None):
    import os
    sim_root = root()
    if image_id is not None:
        branch, iteration, _, _ = image_id.split('_')
        string = String.load(branch=branch, iteration=int(iteration))
    else:
        if iteration is None:
            string = load()  # load highest iteration of the string
        else:
            string = String.load(branch='XCsw', iteration=iteration)

    observables = string.opaque['observables']

    for observable in observables:
        for image in string.images.values():
            #print('checking:', observable['name'], image.image_id, end=' ')
            if image_id is None or image.image_id == image_id:
                trajectory = image.base + '.dcd'  # TODO: have other file extensions that dcd, where to save this?
                if os.path.exists(trajectory):
                    fname_base_out = \
                        '{root}/observables/{branch}_{iteration:03d}/{name}/{image_id}'.format(root=sim_root,
                                                                                               name=observable['name'],
                                                                                               branch=string.branch,
                                                                                               iteration=string.iteration,
                                                                                               image_id=image.image_id)
                    print('checking', fname_base_out, end=' ')
                    if os.path.exists(fname_base_out + '.npy') or os.path.exists(fname_base_out + '.pdb') or (os.path.exists(fname_base_out + '.colvars.traj') and not ignore_colvar_traj) \
                        or os.path.exists(fname_base_out + '.pdb'):
                        print('exist. OK.')
                    else:
                        print('not found; making file')
                        full_command = \
                            '{command} --id {image_id} --cvname {name} {trajectory}'.format(command=observable['command'],
                                                                                          image_id=image.image_id,
                                                                                          name=observable['name'],
                                                                                          trajectory=trajectory)

                        print('running', full_command)
                        env = image._make_env(random_number=0)
                        #print(env)
                        env.update(os.environ)
                        subprocess.run(full_command, shell=True, env=env)

