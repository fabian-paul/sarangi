import subprocess
from . import load, root, is_sim_id, String
from .util import mkdir


def main_transform(transform_and_save, cvname='colvars', opt_kwargs=None):
    r'''Entry point for all scripts that compute collective variables. Defines common command line interface.

    :param transform_and_save:
        function to be called on every MD trajectory
    :param cvname:
        name of the default collective variable name, which in turn defines the subfolder where results are saved
    '''
    import os
    import argparse

    if opt_kwargs is None:
        opt_kwargs = {}

    parser = argparse.ArgumentParser(description='Compute ' + cvname,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--id', metavar='code', default='$STRING_IMAGE_ID',  # TODO: remove this default, since most of the time, we will run through `main_update` which does not rely on the env
                        help='simulation ID. If this is left empty or this points to an undefined environment variable,'
                             ' try to infer from file name.')
    parser.add_argument('trajectories', metavar='paths', default='out.dcd', nargs='+',
                        help='(in) file name(s) of the trajectory file(s) to be analyzed')
    parser.add_argument('--cvname', metavar='identifier', default=cvname,
                        help='name under which to save results')
    parser.add_argument('--init', action='store_true',
                        help='process intial swarm data; else process final data')


    args = parser.parse_args()

    global_sim_id = os.path.expandvars(args.id)

    if '$' not in global_sim_id and global_sim_id != '':
        # $ occurs if global_sim_id contains an unresolved environment variable, such as when running not in a jobfile
        if len(args.trajectories) > 1:
            raise ValueError(
                'You specified the concrete simulation id %s but supplied multiple input trajectories. '
                'This is inconsistent. Giving up.' % global_sim_id)

    for fname_traj in args.trajectories:
        if '$' in global_sim_id or global_sim_id == '':
            # infer form file name
            base_name = os.path.split(os.path.splitext(fname_traj)[0])[1]
            if base_name[-5:]=='_init':
                 base_name = base_name[0:-5]
                 init = True
            else:
                 init = False
            if is_sim_id(base_name):
                sim_id = base_name
            else:
                raise ValueError('Could not infer image id from trajectory name %s.' % base_name)
        else:
            sim_id = global_sim_id
            init = args.init
        if init:
            assert 'init' in fname_traj

        branch, iter_, _, _ = sim_id.split('_')
        if init:
            folder_out = '%s/observables/%s_%03d/%s_init/' % (root(), branch, int(iter_), args.cvname)
        else:
            folder_out = '%s/observables/%s_%03d/%s/' % (root(), branch, int(iter_), args.cvname)
        mkdir(folder_out)
        fname_base_out = folder_out + sim_id
        transform_and_save(fname_traj=fname_traj, fname_base_out=fname_base_out, sim_id=sim_id, **opt_kwargs)


def parse_args_update(argv=None):
    import argparse
    import sys
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--iteration', help='do not analyse current string but a past iteration. If prefixed by -, '
                                            'the number is treated as an offset relative to the highest iteration. '
                                            'Current iteration has offset -1 (like in Python lists).',
                        default='-1')
    parser.add_argument('--branch', help='select branch', default='AZ')
    parser.add_argument('--image', help='select image id. If given, overrides branch and iteration.')
    parser.add_argument('--ignore_colvars_traj',
                        help='Ignore *.colvars.traj files, proceed as if there were not there.', action='store_true')

    args = parser.parse_args(argv)

    options = {'iteration': int(args.iteration), 'branch': args.branch, 'image_id': args.image,
               'ignore_colvars_traj': args.ignore_colvars_traj}

    return options


def _process_trajectory(trajectory_fname, image, sim_root, command, observable_name, ignore_colvars_traj, init):
    import os
    if os.path.exists(trajectory_fname):
        if init:
            folder_out = \
                '{root}/observables/{branch}_{iteration:03d}/{name}_init/'.format(root=sim_root,
                                                                                  name=observable_name,
                                                                                  branch=image.branch,
                                                                                  iteration=image.iteration)
        else:
            folder_out = \
                '{root}/observables/{branch}_{iteration:03d}/{name}/'.format(root=sim_root,
                                                                             name=observable_name,
                                                                             branch=image.branch,
                                                                             iteration=image.iteration)

        fname_base_out = folder_out + image.image_id

        if init:
           init_arg = '--init'
        else:
           init_arg = ''

        print('checking', fname_base_out, end=' ')
        if os.path.exists(fname_base_out + '.npy') or os.path.exists(fname_base_out + '.pdb') or (
                os.path.exists(fname_base_out + '.colvars.traj') and not ignore_colvars_traj) \
                or os.path.exists(fname_base_out + '.pdb.gz'):
            print('exist. OK.')
        else:
            print('not found; making file')
            full_command = \
                '{command} --id {image_id} --cvname {name} {init_arg} {trajectory}'.format(command=command,
                                                                                           image_id=image.image_id,
                                                                                           name=observable_name,
                                                                                           init_arg=init_arg,
                                                                                           trajectory=trajectory_fname)

            print('running', full_command)
            mkdir(folder_out)
            env = image._make_env(random_number=0)
            # print(env)
            env.update(os.environ)
            subprocess.run(full_command, shell=True, env=env)


def main_update(image_id=None, branch=None, iteration=None, ignore_colvars_traj=False):
    'See `parse_args_update` for documentation.'

    sim_root = root()
    if image_id is not None:
        branch, iteration, _, _ = image_id.split('_')
        iteration = int(iteration)

    string = String.load(branch=branch, offset=iteration)

    observables = string.opaque['observables']

    for observable in observables:
        for image in string.images.values():
            #print('checking:', observable['name'], image.image_id, end=' ')
            if image_id is None or image.image_id == image_id:
                # TODO: have other file extensions than dcd; where to save this? Or just try a bunch of different formats xtc/nc/dcd?
                _process_trajectory(trajectory_fname=image.base + '.dcd', sim_root=sim_root,
                                    observable_name=observable['name'],
                                    command=observable['command'], image=image, ignore_colvars_traj=ignore_colvars_traj, init=False)
                # process the *_init.dcd file if it exists
                _process_trajectory(trajectory_fname=image.base + '_init.dcd', sim_root=sim_root,
                                    observable_name=observable['name'],
                                    command=observable['command'], image=image, ignore_colvars_traj=ignore_colvars_traj, init=True)
