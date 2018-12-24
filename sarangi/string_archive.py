#!/usr/bin/env python

import mdtraj
import os
import shutil

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
    # TODO: export to command line interface
    root = os.environ['STRING_SIM_ROOT']
    branch, iteration, image_major, image_minor = sim_id.split('_')
    fname_archived = '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{image_major:02d}_{image_minor:02d}'.format(  # TODO: also have this as an environment variable...
                            root=root, branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
                         )
    shutil.copy(fname_trajectory, fname_archived+'.dcd')
    shutil.copy(fname_colvars_traj, fname_archived+'.colvars.traj')


def extract(plan, sim_id, fname_dest_coor, fname_dest_box, top):
    root = os.environ['STRING_SIM_ROOT']
    me = next(i for i in plan['images'] if i['id']==sim_id)
    prev_id = me['prev_image_id']
    branch, iteration, image_major, image_minor = prev_id.split('_')
    fname_dcd = '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{image_major:02d}_{image_minor:02d}.dcd'.format(
                    root=root, branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
                )
    frame_index = me['prev_frame_number']
    frame = mdtraj.load_frame(fname_dcd, index=frame_index, top=top)
    save_coor(frame, fname_dest_coor)
    save_xsc(frame, fname_dest_box)


def write_colvar(colvars_file, colvars_template, sim_id, plan):
    me = next(i for i in plan['images'] if i['id']==sim_id)
    with open(colvars_template) as f:
        config = ''.join(f.readlines()) + '\n'
    for restraint_name, restraint_value in me['node'].items():  # TODO: numpy conversion
        config += 'harmonic {{\n'\
                  'name {restraint_name}_restraint\n'\
                  'colvars {restraint_name}\n'\
                  'forceconstant 10\n'\
                  'centers {restraint_value}\n}}\n'.format(restraint_name=restraint_name, restraint_value=restraint_value)
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
                           help='path to plan file')
    extractor.add_argument('--colvars_template', metavar='path',
                        default='$STRING_SIM_ROOT/setup/colvars.template',
                        help='file name of the colvar template')
    extractor.add_argument('--colvars', metavar='path',
                        default='colvars.in',
                        help='file name of the colvar template')
    extractor.add_argument('--coordinates', metavar='path', default='in.coor',
                        help='file name to which the restart coordinates are extracted')
    extractor.add_argument('--box', metavar='path', default='in.xsc',
                        help='file name to which the restart box information is extracted')
    extractor.add_argument('--top', metavar='path', default='$STRING_SIM_ROOT/setup/system.pdb',
                        help='file name of topology')

    args = parser.parse_args()

    #top = os.path.expandvars(args.topology)

    if args.command == 'store':
        sim_id = os.path.expandvars(args.id)
        plan = load_plan(fname_plan=os.path.expandvars(args.plan), sim_id=sim_id)
        store(plan=plan, sim_id=sim_id, fname_trajectory=args.trajectory, fname_colvars_traj=args.colvarstraj)
    elif args.command == 'extract':
        sim_id = os.path.expandvars(args.id)
        plan = load_plan(fname_plan=os.path.expandvars(args.plan), sim_id=sim_id)
        extract(plan=plan, sim_id=sim_id, fname_dest_coor=args.coordinates, fname_dest_box=args.box, top=os.path.expandvars(args.top))
        write_colvar(plan=plan, sim_id=sim_id, colvars_file=args.colvars, colvars_template=os.path.expandvars(args.colvars_template))

    print('end achivist')
