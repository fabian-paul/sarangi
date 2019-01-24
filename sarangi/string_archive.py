#!/usr/bin/env python

import mdtraj
import os
import shutil
import warnings

def save_coor(traj, fname, frame=0):
    from struct import pack
    xyz = traj[frame].xyz[0, :, :]
    with open(fname, 'wb') as f:
        f.write(pack('<L', xyz.shape[0]))
        for atom in xyz:
            f.write(pack('<d', atom[0]*10.))
            f.write(pack('<d', atom[1]*10.))
            f.write(pack('<d', atom[2]*10.))


def save_xsc(traj, fname, frame=0):
    f = traj[frame]
    s = '# NAMD extended system configuration output file\n'
    s +='#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z s_x s_y s_z s_u s_v s_w\n'
    s += '%d ' % f.time[0]
    #print(f.unitcell_vectors.shape)
    s += '%f %f %f ' % tuple(f.unitcell_vectors[0, 0, :]*10.)
    s += '%f %f %f ' % tuple(f.unitcell_vectors[0, 1, :]*10.)
    s += '%f %f %f ' % tuple(f.unitcell_vectors[0, 2, :]*10.)
    s += '0 0 0 0 0 0 0 0 0\n'
    with open(fname, 'w') as file:
        file.write(s)
    #frame.save_netcdfrst(fname_dest)


def load_plan(fname_plan, sim_id):
    #branch, iteration, image_major, image_minor = prev_id.split('_')
    import yaml
    with open(fname_plan) as f:
        plan = yaml.load(f)
    return plan['strings'][0] # TODO


def store(plan, fname_trajectory, fname_colvars_traj, sim_id):
    # TODO: export to command line interface (???)
    root = os.environ['STRING_SIM_ROOT']
    branch, iteration, image_major, image_minor = sim_id.split('_')
    fname_archived = '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{image_major:03d}_{image_minor:03d}'.format(  # TODO: also have this as an environment variable...
                            root=root, branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
                         )
    shutil.copy(fname_trajectory, fname_archived+'.dcd')
    shutil.copy(fname_colvars_traj, fname_archived+'.colvars.traj')


def extract(plan, sim_id, fname_dest_coor, fname_dest_box, top):
    root = os.environ['STRING_SIM_ROOT']
    me = next(i for i in plan['images'] if i['id']==sim_id)
    prev_id = me['prev_image_id']
    branch, iteration, image_major, image_minor = prev_id.split('_')
    fname_dcd = '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{image_major:03d}_{image_minor:03d}.dcd'.format(
                    root=root, branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
                )
    frame_index = me['prev_frame_number']
    frame = mdtraj.load_frame(fname_dcd, index=frame_index, top=top)
    save_coor(frame, fname_dest_coor)
    save_xsc(frame, fname_dest_box)


def write_image(plan, sim_id, fname_dest_pdb, top_fname):
    root = os.environ['STRING_SIM_ROOT']
    me = next(i for i in plan['images'] if i['id'] == sim_id)
    if 'atoms_1' not in me:
        warnings.warn('String archivist was instructed to write an image file but the plan does not contain an atom '
                      'reference. Skipping this step.')
        return

    prev_id = me['prev_image_id']
    branch, iteration, image_major, image_minor = prev_id.split('_')
    fname_dcd = '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{image_major:03d}_{image_minor:03d}.dcd'.format(
        root=root, branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
    )
    frame_index = me['prev_frame_number']  # TODO: change me?
    frame = mdtraj.load_frame(fname_dcd, index=frame_index, top=top_fname)

    atoms = me['atoms_1']
    b_factors = [1 if (i + 1) in atoms else 0 for i in range(frame.top.n_atoms)]
    frame.save_pdb(fname_dest_pdb, bfactors=b_factors)


def write_colvar(colvars_file, colvars_template, sim_id, plan):
    if not os.path.exists(colvars_template):
        warnings.warn('String archivist was instructed to write create an colvar input file but not template was found.'
                      ' Skipping this step.')
        return

    # TODO: if we use RMSD with changing atoms these would have be written as well to the colvar file atoms { atomsset {...} }
    me = next(i for i in plan['images'] if i['id']==sim_id)
    with open(colvars_template) as f:
        config = ''.join(f.readlines()) + '\n'
    for restraint_name, restraint_value in me['node'].items():  # TODO: numpy conversion
        restraint_value_namd = str(restraint_value).replace('[', '(').replace(']', ')')
        config += 'harmonic {{\n'\
                  'name {restraint_name}_restraint\n'\
                  'colvars {restraint_name}\n'\
                  'forceconstant {spring}\n'\
                  'centers {restraint_value}\n}}\n'.format(restraint_name=restraint_name, restraint_value=restraint_value_namd, spring=me['spring'][restraint_name])
    with open(colvars_file, 'w') as f:
        f.write(config)
    # add extra stuff, like changing images of the path collective variabel later;
    # this would involve extracting a frame and converting it to pdb or xyz format (looks like here is the good place to do this)!!!


if __name__ == '__main__': 
    import argparse
    # the plan or part of the plan might be in the current directory
    parser = argparse.ArgumentParser(description='archive trajectory segments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparser = parser.add_subparsers(title='commands', dest='command')
    storer = subparser.add_parser('store')
    storer.add_argument('--id', metavar='code', default='$STRING_IMAGE_ID',
                        help='simulation ID')
    storer.add_argument('--plan', metavar='path', default='$STRING_PLAN',
                        help='path to plan file')
    storer.add_argument('--trajectory', metavar='path', default='out.dcd',
                        help='file name of the trajectory file to be stored')
    storer.add_argument('--colvarstraj', metavar='path', default='out.colvars.traj',
                        help='file name of the colvar trajectory file to be stored')

    extractor = subparser.add_parser('extract')
    extractor.add_argument('--id', metavar='code', default='$STRING_IMAGE_ID',
                           help='simulation ID')
    extractor.add_argument('--plan', metavar='path', default='$STRING_PLAN',
                           help='(in) path to plan file')
    extractor.add_argument('--colvars_template', metavar='path',
                           default='$STRING_SIM_ROOT/setup/colvars.template',
                           help='(in) file name of the colvar template')
    extractor.add_argument('--colvars', metavar='path',
                           default='colvars.in',
                           help='(out) file name of the colvar configuration file')
    extractor.add_argument('--coordinates', metavar='path', default='in.coor',
                           help='(out) file name to which the restart coordinates are extracted')
    extractor.add_argument('--box', metavar='path', default='in.xsc',
                           help='(out) file name to which the restart box information is extracted')
    extractor.add_argument('--top', metavar='path', default='$STRING_SIM_ROOT/setup/system.pdb',
                           help='(in) file name of topology')
    extractor.add_argument('--image', metavar='path',
                           default='image.pdb',
                           help='(out) file name for the image coordinated')


    args = parser.parse_args()

    #top = os.path.expandvars(args.topology)

    if args.command == 'store':
        sim_id = os.path.expandvars(args.id)
        plan = load_plan(fname_plan=os.path.expandvars(args.plan), sim_id=sim_id)
        store(plan=plan, sim_id=sim_id, fname_trajectory=args.trajectory, fname_colvars_traj=args.colvarstraj)
    elif args.command == 'extract':
        sim_id = os.path.expandvars(args.id)
        top = os.path.expandvars(args.top)
        plan = load_plan(fname_plan=os.path.expandvars(args.plan), sim_id=sim_id)
        extract(plan=plan, sim_id=sim_id, fname_dest_coor=args.coordinates, fname_dest_box=args.box, top=top)

        write_colvar(plan=plan, sim_id=sim_id, colvars_file=args.colvars, colvars_template=os.path.expandvars(args.colvars_template))

        write_image(plan=plan, sim_id=sim_id, fname_dest_pdb=args.image, top_fname=top)

    print('end achivist')
