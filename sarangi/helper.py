from .sarangi import All


__all__ = ['nodes_to_trajs']


def nodes_to_trajs(string, fname='nodes.pdb', fields=All):
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
