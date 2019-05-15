#!/usr/bin/env python

import mdtraj
import os
import shutil
import warnings
import sarangi
from sarangi.util import root


__all__ = ['save_coor', 'save_xsc', 'store', 'extract', 'write_image_coordinates', 'write_colvar',
           'load_initial_coordinates']


def sim_id_to_seq(s):
    fields = s.split('_')
    return float(fields[2] + '.' + fields[3])


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
    s += '#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z s_x s_y s_z s_u s_v s_w\n'
    s += '%d ' % f.time[0]
    # print(f.unitcell_vectors.shape)
    s += '%f %f %f ' % tuple(f.unitcell_vectors[0, 0, :]*10.)
    s += '%f %f %f ' % tuple(f.unitcell_vectors[0, 1, :]*10.)
    s += '%f %f %f ' % tuple(f.unitcell_vectors[0, 2, :]*10.)
    s += '0 0 0 0 0 0 0 0 0\n'
    with open(fname, 'w') as file:
        file.write(s)
    # frame.save_netcdfrst(fname_dest)


def store(fname_trajectory, fname_colvars_traj, sim_id):
    branch, iteration, image_major, image_minor = sim_id.split('_')
    fname_archived = '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{image_major:03d}_{image_minor:03d}'.format(
        root=root(), branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
    )
    shutil.copy(fname_trajectory, fname_archived+'.dcd')
    fname_archived = '{root}/observables/{branch}_{iteration:03d}/colvars/{branch}_{iteration:03d}_{image_major:03d}_{image_minor:03d}'.format(
        root=root(), branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
    )
    shutil.copy(fname_colvars_traj, fname_archived+'.colvars.traj')


def extract(image, fname_dest_coor, fname_dest_box, top, number=-1):
    if isinstance(image, sarangi.image.Image):
        prev_id = image.previous_image_id
        frame_index = image.previous_frame_number
    elif isinstance(image, dict):
        prev_id = image['prev_image_id']
        frame_index = image['prev_frame_number']
    else:
        raise ValueError('"image" is neither a python dict nor a sarangi Image object. Don\'t know how to handle it.')
    branch, iteration, image_major, image_minor = prev_id.split('_')
    fname_dcd = '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{image_major:03d}_{image_minor:03d}.dcd'.format(
        root=root(), branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
    )
    if number >= 0:
        frame_index = number  # overwrite plan information by user selection
    frame = mdtraj.load_frame(fname_dcd, index=frame_index, top=top)
    save_coor(frame, fname_dest_coor)
    save_xsc(frame, fname_dest_box)


def load_initial_coordinates(image, top_fname):
    if isinstance(image, sarangi.image.Image):
        prev_id = image.previous_image_id
        frame_index = image.previous_frame_number
    elif isinstance(image, dict):
        prev_id = image['prev_image_id']
        frame_index = image['prev_frame_number']
    else:
        raise ValueError('"image" is neither a python dict nor a sarangi Image object. Don\'t know how to handle it.')
    branch, iteration, image_major, image_minor = prev_id.split('_')
    fname_dcd = '{root}/strings/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{image_major:03d}_{image_minor:03d}.dcd'.format(
        root=root(), branch=branch, iteration=int(iteration), image_major=int(image_major), image_minor=int(image_minor)
    )
    frame = mdtraj.load_frame(fname_dcd, index=frame_index, top=top_fname)
    return frame


def write_image_coordinates(image, fname_dest_pdb, top_fname):
    if not isinstance(image, sarangi.image.CartesianImage):
        warnings.warn('String archivist was instructed to write an image file but the plan does not contain an atom '
                      'reference. Skipping this step.')
        return
    frame = load_initial_coordinates(image, top_fname)
    atoms = image.colvars_def['cartesian']['atomnumbers']
    b_factors = [1 if (i + 1) in atoms else 0 for i in range(frame.top.n_atoms)]
    frame.save_pdb(fname_dest_pdb, bfactors=b_factors)


def write_colvar(colvars_file, string, image):
    default_colvars_template = '$STRING_SIM_ROOT/setup/colvars.template'  # code should select which colvars to use

    if string.colvars_def is not None and 'template_file' in string.colvars_def:
        colvars_template = os.path.expandvars(string.colvars_def['template_file'])
    else:
        colvars_template = os.path.expandvars(default_colvars_template)

    if not os.path.exists(colvars_template):
        warnings.warn('String archivist was instructed to write a colvars input file but no template was found. '
                      'Skipping this step and hoping for the best.')
        return

    with open(colvars_template) as f:
        config = ''.join(f.readlines()) + '\n'

    config += image.write_namd_conf(cwd=os.getcwd())

    with open(colvars_file, 'w') as f:
        f.write(config)


def write_re_config(string, group_id='$STRING_GROUP_ID', bias_conf_fname='bias.conf'):
    # TODO: move to main lib
    images = string.group[group_id].images_ordered
    config = 'set num_replicas %d\n' % len(images)
    config += 'proc replica_bias { i } {\n  switch $i {\n'
    for i, im in enumerate(images):
        config += '  %d {\n    return {' % i
        for restraint_name in im.fields:
            spring_value = im.spring[restraint_name]
            if im.node is not None:
                center_value = im.node[restraint_name]
            else:
                warnings.warn('Spring constant was defined but no umbrella center. Using the default 0.0.')
                center_value = '0.0'
            spring_value_namd = str(spring_value).replace('[', '(').replace(']', ')')
            center_value_namd = str(center_value).replace('[', '(').replace(']', ')')
            config += '    {restraint_name}_restraint {{ centers {center} forceconstant {spring} }}\\\n'.format(
                                                                    restraint_name=restraint_name,
                                                                    center=center_value_namd,
                                                                    spring=spring_value_namd)
        config += '    }}\n'
    config += '}\n}\n'
    with open(bias_conf_fname, 'w') as f:
        f.write(config)


if __name__ == '__main__': 
    import argparse
    # the plan or part of the plan might be in the current directory
    parser = argparse.ArgumentParser(description='archive trajectory segments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparser = parser.add_subparsers(title='commands', dest='command')
    storer = subparser.add_parser('store', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    storer.add_argument('--id', metavar='code', default='$STRING_IMAGE_ID',
                        help='simulation ID')
    storer.add_argument('--plan', metavar='path', default='$STRING_PLAN',
                        help='path to plan file')
    storer.add_argument('--trajectory', metavar='path', default='out.dcd',
                        help='file name of the trajectory file to be stored')
    storer.add_argument('--colvarstraj', metavar='path', default='out.colvars.traj',
                        help='file name of the colvar trajectory file to be stored')

    extractor = subparser.add_parser('extract', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    extractor.add_argument('--id', metavar='code', default='$STRING_IMAGE_ID',
                           help='simulation ID')
    extractor.add_argument('--frame', metavar='number', default='-1',
                           help='frame number to extract')
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
                           help='(out) file name for the umbrella center (image) coordinates')

    re_extractor = subparser.add_parser('replica_extract', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    re_extractor.add_argument('--id', metavar='code', default='$STRING_GROUP_ID',
                              help='group ID')
    re_extractor.add_argument('--plan', metavar='path', default='$STRING_PLAN',
                              help='(in) path to plan file')
    re_extractor.add_argument('--top', metavar='path', default='$STRING_SIM_ROOT/setup/system.pdb',
                              help='(in) file name of topology')
    re_extractor.add_argument('--config', metavar='path', default='bias.conf',
                              help='(out) file name for bias configuration file to generate')

    re_storer = subparser.add_parser('replica_store', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    re_storer.add_argument('--id', metavar='code', default='$STRING_GROUP_ID',
                           help='group ID')
    re_storer.add_argument('--plan', metavar='path', default='$STRING_PLAN',
                           help='(in) path to plan file')

    args = parser.parse_args()

    if args.command == 'store':
        sim_id = os.path.expandvars(args.id)
        store(fname_trajectory=args.trajectory, fname_colvars_traj=args.colvarstraj, sim_id=sim_id)

    elif args.command == 'extract':
        sim_id = os.path.expandvars(args.id)
        top = os.path.expandvars(args.top)
        string = sarangi.String.load_form_fname(fname=os.path.expandvars(args.plan))
        image = string.images[sim_id]
        extract(image=image, fname_dest_coor=args.coordinates, fname_dest_box=args.box, top=top, number=int(args.frame))

        write_colvar(image=image, string=string, colvars_file=args.colvars)

        write_image_coordinates(image=image, fname_dest_pdb=args.image, top_fname=top)

    elif args.command == 'replica_extract':
        top = os.path.expandvars(args.top)
        group_id = os.path.expandvars(args.id)
        string = sarangi.String.load_form_fname(fname=os.path.expandvars(args.plan))

        # write bias definition
        write_re_config(string=string, group_id=group_id, bias_conf_fname=args.config)

        # write initial conditions
        for i, image in enumerate(string.group[group_id].images_ordered):
            print('replica-extract: %s -> in.%d.coor' % (image.image_id, i))
            extract(image=image, fname_dest_coor='in.%d.coor' % i, fname_dest_box='in.%d.xsc' % i, top=top)

    if args.command == 'replica_store':
        group_id = os.path.expandvars(args.id)
        string = sarangi.String.load_form_fname(fname=os.path.expandvars(args.plan))
        for i, image in enumerate(string.group[group_id].images_ordered):
            print('replica-store: out.%d.sort.dcd -> %s' % (i, image['id']))
            store(fname_trajectory='out.%d.sort.dcd' % i, fname_colvars_traj='out.%d.sort.colvars.traj' % i, sim_id=image.image_id)
            # TODO: reorder by Hamiltonian (dcd and colvars.traj) ?
            # TODO: store more data, like the history of biases? ?

    print('end archivist')
