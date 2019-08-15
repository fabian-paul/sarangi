import numpy as np
import os
import datetime
import shutil
import subprocess
from . import colvars
from . import image
from . import util
from . import sarangi

try:
    from .short_path import path as shortest_path
except ImportError:
    def shortest_path(cv, start, stop, lambada):
        # TODO: debug me!
        import scipy.spatial
        import scipy.sparse.csgraph
        cost = np.exp(lambada * scipy.spatial.distance_matrix(cv, cv)) - 1.  # this will fill up memory and crash
        _, pred = scipy.sparse.csgraph.dijkstra(cost, directed=False, indices=[start], return_predecessors=True)
        path = [stop]
        u = stop
        while pred[u] != start:
            u = pred[u]
            path.append(u)
        return [start] + path[::-1]


__all__ = ['import_trajectory']


def get_colvars(traj_fnames, branch, command, fields=util.All, cvname='colvars'):
    'Generate the colvars for all input trajectories (load if already precomputed, or compute them).'

    cv_folder = '{root}/observables/{branch}_000/{cvname}/'.format(root=util.root(), cvname=cvname, branch=branch)
    util.mkdir(cv_folder)

    cvs = []

    for i_part, traj_fname in enumerate(traj_fnames):
        image_id = '{branch}_000_000_{i_part:03d}'.format(branch=branch, i_part=i_part)

        for ext in ['.npy', '.npz', '.colvars.traj']:
            cv_fname = cv_folder + '/' + image_id + ext
            print('seaching for', cv_fname, '...')
            if os.path.exists(cv_fname):
                print('found')
                break
        else:  # file not found
            full_command = '{command} --id {image_id} --cvname {cvname} {fname_traj}'.format(command=command,
                                                                                             cvname=cvname,
                                                                                             image_id=image_id,
                                                                                             fname_traj=traj_fname)
            print('running', full_command)
            #os.environ['STRING_SIM_ROOT'] = util.root()
            env = {'STRING_SIM_ROOT': util.root()}
            env.update(os.environ)
            result = subprocess.run(os.path.expandvars(full_command), shell=True, env=env)
            # print('execution result:', result)
            if result.returncode != 0:
                raise RuntimeError('Colvar script returned with error code %d.' % result.returncode)

        cvs.append(colvars.Colvars(folder=cv_folder, base=image_id, fields=fields))

    return cvs


def collect_frames(fnames_traj_in, fname_traj_out, traj_idx, frame_idx, top_fname='{root}/setup/system.pdb'):
    'Collect individual frames from multiple trajectories and write them to disk as a single trajectory.'
    import mdtraj
    top_fname = top_fname.format(root=util.root())
    top = mdtraj.load_topology(top_fname)

    traj_out = None
    for i_traj in np.unique(traj_idx):
        traj = mdtraj.load(fnames_traj_in[i_traj], top=top)
        print('collecting frames from', fnames_traj_in[i_traj])
        i_current_frames = frame_idx[traj_idx == i_traj]
        assert np.all(i_current_frames < len(traj))
        if traj_out is None:
            traj_out = traj[i_current_frames]
        else:
            traj_out += traj[i_current_frames]

    print('saving')
    traj_out.save(fname_traj_out)


def import_trajectory(traj_fnames, branch, fields=util.All, swarm=True, end=-1, cvname='colvars', max_images=200, min_rmsd=0.3,
           mother_string=None, command=None, compress=True, stride=1, n_clusters=50000):
    r'''Import trajectory into Sarangi and turn it into a String object (make directory structure and plan file).

    Parameters
    ----------
    traj_fnames : str or list of str
        File name of the input trajectory. Can also be a list of file names, in which case the contents
        of the files is concatenated (in the exact order given by the list).
    branch : str
        Short label for the new string, e.g. 'AZ'
    fields : list or `sarangi.All`
        Selection of collective variable names. If `All` is given, use all colvars that the `command` outputs.
    swarm : bool
        If true, propagate using the swarm of trajectories method. If false, used permanent biases (umbrella sampling).
    end : int
        stop looking at the data past this frame (default=-1, i.e. use all frames)
    cvname : str
        Subfolder name in the "observables" folder where collective variables are stored.
    max_images : int
        New string will not contain more that this number of images.
    min_rmsd : float
        Stop adding images to the initial string, once the RMSD bottleneck reaches this value (in Angstrom).
    mother_string : sarangi.String
        String object, will copy basic configuration information from that string.
    command : str
        Executable that computes collective variables. Program must accept --id --cvname options.
        Can be omitted in which case the command will be taken from configuration information in the
        mother string. Either `command` or `mother_string` must be set.
    compress: bool
        If true, only import into the project the MD frames (full coordinares) that will actually be used as
        starting points for MD integration.
        If false, copy the whole input trajectories into the project (sub-)folder.
        # TODO: allow to input collective variable file directly
    stride : int
        Only use every n'th frame from the input trajectories.
    n_clusters : int
        Reduce maximum number of input data points to this number by running k-means clustering.
        If the number of input data points is lower than this number, directly use all of them.

    Returns
    -------
    s : sarangi.String
        A new String object, ready to propagate.
    '''
    # TODO do some concatenation? (first convert to cvs, then concatenate; don't forget to translate indices)

    # TODO: should we have some pre-clustering to reduce the data in a first step?
    # Preclustering reduces the complexity of the shortest path algorithm form N*N to N*M where M is a fixed
    # number of clusters. This comes at the cost of extra code complexity (more indexing operations)

    if isinstance(traj_fnames, str):
        traj_fnames = [traj_fnames]

    if mother_string is not None and 'observables' in mother_string.opaque:
        command = [obs['command'] for obs in mother_string.opaque['observables'] if obs['name'] == 'colvars']
    else:
        if command is None:
            raise ValueError('One of the parameters "command" or "mother_string" must be set.')

    cv_trajs = get_colvars(traj_fnames=traj_fnames, branch=branch, command=command, fields=fields, cvname=cvname)
    real_fields = cv_trajs[0].fields
    real_cv_dims = cv_trajs[0].dims
    # In the next few steps, we will concatenate the trajectories. But first make some indices that will allow
    # us to undo the concatenation.
    frame_indices = np.concatenate([np.arange(len(cv_traj))[::stride] for cv_traj in cv_trajs])[0:end]
    traj_indices = np.concatenate([np.zeros(len(cv_traj), dtype=int)[::stride] + i for i, cv_traj in enumerate(cv_trajs)])[0:end]
    cvs_linear = np.concatenate([cv_traj.as2D(fields=fields)[::stride] for cv_traj in cv_trajs])[0:end]
    #del cv_trajs

    if len(cvs_linear) > n_clusters:
        print('Collective variable trajectory contains many frames (>100000). Reducing data with k-means first.')
        import sklearn.cluster
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(cvs_linear)
        center_indices = [np.argmin(np.linalg.norm(cvs_linear - c, axis=1)) for c in kmeans.cluster_centers_]
        frame_indices = frame_indices[center_indices]
        traj_indices = traj_indices[center_indices]
        cvs_linear = kmeans.cluster_centers_

    # select images using subtropical shortest path algorithm
    image_indices_linear = denoise(cvs_linear, max_images=max_images, min_rmsd=min_rmsd)
    centers = cvs_linear[image_indices_linear, :]

    # import frames into the project ...
    util.mkdir('{root}/strings/{branch}_000/'.format(root=util.root(), branch=branch))
    if not compress:
        # ... either as complete trajectories
        for traj_index, traj_fname in enumerate(traj_fnames):
            ext = os.path.splitext(traj_fname)[1]
            fname_dest = '{root}/strings/{branch}_000/{branch}_000_000_{traj_index:03d}.{ext}'.format(root=util.root(),
                                                                                                      branch=branch,
                                                                                                      traj_index=traj_index,
                                                                                                      ext=ext)
            shutil.copy(traj_fname, fname_dest)
    else:
        # ... or just the relevant frames
        fname_traj_out = '{root}/strings/{branch}_000/{branch}_000_000_000.dcd'.format(root=util.root(), branch=branch)
        collect_frames(fnames_traj_in=traj_fnames, fname_traj_out=fname_traj_out,
                       traj_idx=traj_indices[image_indices_linear], frame_idx=frame_indices[image_indices_linear])
        # colvars are outdated now, we need to recreate the file
        # TODO: delete temporary cv files ...
        fname_cv_out = '{root}/observables/{branch}_000/{cvname}/{branch}_000_000_000.npy'.format(root=util.root(), branch=branch, cvname=cvname)
        np.save(fname_cv_out, util.flat_to_structured(cvs_linear, fields=real_fields, dims=real_cv_dims))

    if mother_string is not None:
        new_string = mother_string.empty_copy(iteration=0, branch=branch)
    else:
        new_string = sarangi.String(branch=branch, iteration=0, images={}, image_distance=min_rmsd, previous=None,
                                    colvars_def=None, opaque={'observables': [{'command': command, 'name': cvname}]})

    # create String object / plan file
    for running_index, (center, index_linear) in enumerate(zip(centers, image_indices_linear)):
        if compress:
            frame_index = running_index  # just renumber sequentially
            traj_index = 0
        else:
            frame_index = frame_indices[index_linear]
            traj_index = traj_indices[index_linear]
        node = cv_trajs[traj_index]._colvars[frame_index:frame_index + 1]
        spring = util.load_structured({name: 10. for name in cv_trajs[traj_index].fields})
        im = image.CompoundImage(image_id='{branch}_001_{index:03d}_000'.format(branch=branch, index=running_index),
                                 previous_image_id='{branch}_000_000_{traj_index:03d}'.format(branch=branch, traj_index=traj_index),
                                 previous_frame_number=frame_index, group_id=None, node=node, spring=spring, swarm=swarm)
        new_string.add_image(im)

    time = datetime.datetime.strftime(datetime.datetime.now(), '%Y/%m/%d %I:%M')
    new_string.write_yaml(
        message='{time}: New string from initialized from trajectory "{traj_fname}".'.format(time=time, traj_fname=traj_fnames[0]))
    return new_string


def denoise_once(cv, lambada, geomf=None):
    '''Denoise trajectory with subtropical shortest-path algorithm that discourages large jumps in output

    Parameters
    ----------
    cv : np.ndarray((T, n))

    lambada : float
        Parameter that controls level of detail. The higher, the more detail is
        kept (less denoising).  The distance in the shortest-path algorithm is computed as
        dist(i, j) = exp(lambada*geomf*||x_i - x_j||) - 1

    geomf : float or None
        If None, set to geomf = 1./sqrt(n/3)

    Returns
    -------
    list of integers : [i1, i2, ...]
    indices such that [cv[i1, :], cv[i2, :], ...] are the denoised trajectory
    '''
    if geomf is None:
        geomf = (cv.shape[1]/3)**-0.5
    path = shortest_path(cv, 0, cv.shape[0] - 1, lambada * geomf)
    assert np.all(np.array(path) >= 0)
    jumps = []
    for a, b in zip(path[0:-1], path[1:]):
        d = np.linalg.norm(cv[a, :] - cv[b, :])
        jumps.append(d)
    return path, jumps


def denoise(cv, max_images, min_rmsd, max_iter=20, verbose=False):
    lambada = 1.0  # starting value
    lowest_infeasible_top_bound = 100.  # smallest upper bound on the parameter, such that the constraints are fulfilled
    highest_feasible_bottom_bound = 0.  # highest lower bound on the parameter in the infeasible region
    path = [0, len(cv)]

    for _ in range(max_iter):
        try:
            path, jumps = denoise_once(cv, lambada)
        except OverflowError:
            error = np.inf
            n_images = 0
            feasible = False
        else:
            error = np.median(jumps)
            n_images = len(path)
            if verbose:
                print('^=', highest_feasible_bottom_bound, '_=', lowest_infeasible_top_bound)
                print('Denoising attempt with parameter {lambada} yielded {n_images} images and median jump {median}.'.format(
                lambada=lambada, n_images=n_images, median=error))
            feasible = error >= min_rmsd and n_images <= max_images and not np.any(np.isnan(jumps))

        if verbose:
            if feasible:
                print('This is feasible')
            else:
                print('This is infeasible')

        if feasible:
            if lambada > highest_feasible_bottom_bound:  # if we found a new upper boundary, adjust it
                if verbose:
                    print('increased bottom bound', highest_feasible_bottom_bound, '->', lambada)
                highest_feasible_bottom_bound = lambada
        else:  # infeasible
            if lambada < lowest_infeasible_top_bound:  # if we found a new lower boundary, adjust it
                if verbose:
                    print('decreased top bound', lowest_infeasible_top_bound, '->', lambada)
                lowest_infeasible_top_bound = lambada

        if feasible and abs(n_images - max_images) < max_images/10. and abs(error - min_rmsd) < min_rmsd/10.:
            # feasible and good enough -> done
            return path

        if feasible:  # feasible but not good
            # increase lambada
            lambada = 0.5*(lowest_infeasible_top_bound + lambada)
        else:  # neither good nor feasible
            # decrease params
            lambada = 0.5*(highest_feasible_bottom_bound + lambada)

    return path

#def screen(cv, lambadas=None):
#    n_atoms = 1  # TODO
#    N = cv.shape[0]
#    #D = np.zeros((N, N))
#    #for i, x in enumerate(cv):
#    #    for j, y in enumerate(cv):
#    #        D[i,j] = np.linalg.norm(x-y) * n_atoms**-0.5
#
#    means = []
#    lens = []
#    maxs = []
#    bads = []
#    paths = []
#    if lambadas is None:
#        lambadas = list(range(1,10))#[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
#    for lambada in lambadas:
#        #graph = scipy.sparse.csr_matrix(np.exp(lambada*D)-1)  # exponential loss function
#        #dist_matrix, predecessors = scipy.sparse.csgraph.dijkstra(graph, indices=N-1, return_predecessors=True, directed=False)
#        #i1, i2 = N-1, 0
#        #path = []
#        path = shortest_path(cv, 0, cv.shape[0]-1, param=lambada)
#        jumps = []
#        #i = i2
#        #while i != i1:
#        #    path.append(i)
#        #    i = predecessors[i]
#        #path.append(i1)
#        max_d = 0
#        for a, b in zip(path[0:-1], path[1:]):
#            d = np.exp(np.linalg.norm(a-b)*lambada)
#            max_d = max(max_d, d)
#            jumps.append(d)
#        #print('lambda', lambada)
#        #print('max D', max_d)
#        #print('length of path', len(path))
#        #print('mean jump', np.median(jumps))
#        maxs.append(max_d)
#        means.append(np.mean(jumps))
#        lens.append(len(path))
#       bads.append(np.count_nonzero(np.array(jumps) > 0.5))
#        paths.append(path)
#
#    return lambadas, maxs, means, lens, bads, paths



#def generate(image_id='AZ_000_000_000', subdir='', fields=None, length=100):
#    pcoord = Colvars(subdir + '/' + image_id, fields=fields)
#    # TODO: insert the string generation code here
#    pass

#import numpy as np
#import mdtraj

#selection = [0, 86, 790, 890, 1072, 1283, 1581, 1571, 1764, 2026, 2047, 2103, 2152, 2184, 2832, 2837, 3378, 2856, 3407, 3785, 4404, 5047, 5391, 5404, 5736, 5768, 5928, 6208, 6239, 6358, 6771, 6922, 7099, 7350, 7623, 7772, 7922, 7898, 8272, 8501, 8964, 9659, 9658, 9646, 10051, 9952, 10223, 10146, 10242, 10163, 10150, 10152, 10269, 10268, 10265, 10276, 10280, 10307, 10283, 10470, 10477, 10571, 10500, 10492, 10998, 11116, 11589, 11791, 11832, 12242, 12625, 12723, 13099, 13116, 13457, 13607, 14014, 14377, 14828, 14874, 14880, 14889, 14890, 15032, 14981, 15425, 15404, 15613, 16280, 16286, 16296, 16291, 16115, 16072, 16047, 15816, 15930, 16960, 16961, 17066, 17071, 17566, 17457, 18508, 19009, 18758, 18977, 18917, 18921, 19488, 19497, 19522, 19534, 19547, 19551, 20246, 20570, 20557, 20551, 20553, 20954, 21000]

#pdb_fname = '../setup/crystal.pdb'

#print('loading pdb')
#pdb = mdtraj.load(pdb_fname)

#out_traj = None

#traj_fname = 'crystal.2.dcd'

#iter = mdtraj.iterload(traj_fname, top=pdb)
#t = 0
#for chunk in iter:
#    print('.')
#    delta = len(chunk)
#    idx = np.arange(t, t + delta)
#    for i,k in enumerate(idx):
#       if k in selection:
#           if out_traj is None:
#               out_traj = chunk[i]
#           else:
#               out_traj += chunk[i]
#    t += delta

#print('saving')
#out_traj.save_dcd('path.dcd')

# print(traj.unitcell_lengths.shape, traj.unitcell_angles.shape)
# xyz_out = traj.xyz[i_current_frames, :, :]
# unitcell_lengths_out = traj.unitcell_lengths[i_current_frames, :]
# unitcell_angles_out = traj.unitcell_angles[i_current_frames, :]

# xyz_out = np.concatenate((xyz_out, traj.xyz[i_current_frames, :, :]))
# unitcell_lengths_out = np.concatenate((unitcell_lengths_out, traj.unitcell_lengths[i_current_frames, :]))
# unitcell_angles_out = np.concatenate((unitcell_angles_out, traj.unitcell_angles[i_current_frames, :]))

    #order = np.argsort(traj_idx, frame_idx)
    #frame_idx = frame_idx[order]
    #traj_idx = traj_idx[order]


    #xyz_out = None
    #unitcell_lengths_out = None
    #unitcell_angles_out = None
    #mdtraj.Trajectory(xyz_out, top, time=None, unitcell_lengths=unitcell_lengths_out, unitcell_angles=unitcell_angles_out).save(fname_traj_out)
    # from scratch:
    # dtype_spring = [(name, np.float64) for name in centers.dtype.names]
    # spring = np.zeros(1, dtype=dtype_spring)
    #for name in centers.dtype.names:
    #    spring[name] = 3