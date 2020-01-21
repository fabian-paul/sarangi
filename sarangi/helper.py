from .sarangi import All


__all__ = ['nodes_to_trajs', 'export_to_path_metadynamics_abf']


def _compute_pathcv_parameters(fnames_pdbs, atom_indices):
    import mdtraj
    deltas = []
    N = len(atom_indices)
    for fname_a, fname_b in zip(fnames_pdbs[0:-1], fnames_pdbs[1:]):
        a = mdtraj.load(fname_a)
        b = mdtraj.load(fname_b)
        deltas.append(np.linalg.norm(a.xyz[0, atom_indices, :]-b.xyz[0, atom_indices, :])*(N**-0.5))
    return np.mean(deltas)


def export_to_path_collective_variable(string, overlap_matrix=None, folder_out='metad_images', selection='not water', overlap_options=None):
    r'''Generate a sequence of PDB files to be used with path metadynamics or path ABF.

    Parameters
    ----------
    string: sarangi.String
        Base string.

    overlap_matrix: nd.array or None
        Precomputed overlap matrix or None. If None, will compute overlaps
        with the options algorithm='units' and fields=string.images_ordered[0].fields
        To load overlap data from disk, use `sarangi.overlap.load_matrix`.

    folder_out: string
        Folder where to place the PDB files.

    overlap_options: dict or None

    Notes
    -----
    This function computes the widest path through the overlap matrix,
    find realizations that are close to the swarm averages and exports
    them as PDB files.
    Must be run on a machine that has access to structures
    '''
    import mdtraj
    from .util import widest_path, mkdir
    from .sarangi import root
    top = mdtraj.load_topology(root() + '/setup/system.pdb')
    atom_indices = top.select(selection)
    if overlap_matrix is None:
        overlap_options['matrix'] = True
        if 'fields' not in overlap_options:
            overlap_options['fields'] = string.images_ordered[0].fields
        if 'algorithm' not in overlap_options:
            overlap_options['algorithm'] = 'units'
        overlap_matrix = string.overlap(**overlap_options)
    else:
        if not overlap_matrix.shape[0] == overlap_matrix.shape[1] == len(string):
            raise ValueError('Shape of user-specified matrix does not match number of images in the string.')
    p = widest_path(overlap_matrix, regularize=True)
    # TODO: interpolation options?
    mkdir(folder_out)
    fnames_pdb = []
    for i_running, i in enumerate(p):
        md_data_path = string.images_ordered[i].previous_base + '.dcd'
        step = string.images_ordered[i].previous_frame_number
        # im, step, _ = sarangi.sarangi.find_realization_in_string([string], s.images_ordered[i].node)
        # md_data_path = im.base + '.dcd'
        frame = mdtraj.load_frame(md_data_path, step, top=top, atom_indices=atom_indices)
        fname_out = '{folder_out}/{i_running:03d}.pdb'.format(folder_out=folder_out, i_running=i_running)
        fnames_pdb.append(fname_out)
        frame.save_pdb(fname_out)

    # TODO: calculate the path CV parameter d here?


def nodes_to_trajs(string, fname='nodes.pdb', fields=All):
    'Convert nodes to a PDB trajecectory. Works for any 3-D nodes, that do not necessarily have to be atoms.'
    pdb_fmt = '{ATOM:<6}{serial_number:>5} {atom_name:<4}{alt_loc_indicator:<1}{res_name:<3} ' \
              '{chain_id:<1}{res_seq_number:>4}{insert_code:<1}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}'

    frames = []
    for i_node, image in enumerate(string.images_ordered):
        frame = 'MODEL      {model:>3}\n'.format(model=i_node)
        node = image.node
        for i, field in enumerate(node.dtype.names):
            if field in fields:
                frame += (pdb_fmt.format(
                        ATOM='ATOM',
                        serial_number=i,
                        atom_name='C',
                        alt_loc_indicator=' ',
                        res_name=field[0:3],
                        chain_id='A',
                        res_seq_number=i,
                        insert_code=' ',
                        x=node[field][0, 0],
                        y=node[field][0, 1],
                        z=node[field][0, 2],
                        occupancy=1.0,
                        temp_factor=0.0) + '\n')
        frame += 'ENDMDL\n'
        frames.append(frame)

    with open(fname, 'w') as f:
        for frame in frames:
            f.write(frame)

    return frames
