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
    def shortest_path(cv, start, stop, lambada, logspace=False):  # fig leaf fall-back implementation
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


__all__ = ['import_trajectory', 'parse_args_import_trajectory']


def get_colvars(traj_fnames, branch, command, fields=util.All, cvname='colvars', write_to_disk=True, use_existing_file=True):
    'Generate the colvars for all input trajectories (load if already precomputed, or compute them).'

    if not write_to_disk:
        import uuid
        branch = uuid.uuid4().hex[0:10]  # to generate temporary colvar files, select some phantasy branch mame
        # branch = 'TEMP'

    cv_folder = '{root}/observables/{branch}_000/{cvname}/'.format(root=util.root(), cvname=cvname, branch=branch)
    util.mkdir(cv_folder)

    cvs = []

    for i_part, traj_fname in enumerate(traj_fnames):
        image_id = '{branch}_000_000_{i_part:03d}'.format(branch=branch, i_part=i_part)

        for ext in ['.npy', '.npz', '.colvars.traj']:
            cv_fname = cv_folder + '/' + image_id + ext
            print('seaching for', cv_fname, '...')
            if os.path.exists(cv_fname) and use_existing_file:
                print('found')
                break
        else:  # file not found, recompute..
            full_command = '{command} --id {image_id} --cvname {cvname} {fname_traj}'.format(command=command,
                                                                                             cvname=cvname,
                                                                                             image_id=image_id,
                                                                                             fname_traj=traj_fname)
            os.environ['STRING_SIM_ROOT'] = util.root()
            full_command = os.path.expandvars(full_command)
            print('running', full_command)
            env = {'STRING_SIM_ROOT': util.root()}
            env.update(os.environ)
            result = subprocess.run(full_command, shell=True, env=env)
            # print('execution result:', result)
            if result.returncode != 0:
                raise RuntimeError('Colvar script returned with error code %d.' % result.returncode)

        cvs.append(colvars.Colvars(folder=cv_folder, base=image_id, fields=fields))

    return cvs


def collect_frames(fnames_traj_in, fname_traj_out, traj_idx_unordered, frame_idx_unordered, top_fname='{root}/setup/system.pdb'):
    'Collect individual frames from multiple trajectories and write them to disk as a single trajectory.'
    import mdtraj
    top_fname = top_fname.format(root=util.root())
    top = mdtraj.load_topology(top_fname)

    #
    order = np.argsort(traj_idx_unordered, kind='mergesort')
    traj_idx_ordered = traj_idx_unordered[order]
    frame_idx_ordered = frame_idx_unordered[order]

    # print a littls summary
    #for i_traj in np.sort(np.unique(traj_idx)):
    #    print('In trajectory piece %d, will collect %d frames.' % (i_traj, np.count_nonzero(traj_idx == i_traj)))

    traj_out = None
    # We process input trajectories one by one, still the unordered input indices might jump erratically from trajectory
    # to trajectory. That's why we had to first sort all indices by trajectory index.
    for i_traj in np.sort(np.unique(traj_idx_ordered)):
        traj = mdtraj.load(fnames_traj_in[i_traj], top=top)
        print('collecting %d frames from %s.' % (np.count_nonzero(traj_idx_ordered == i_traj), fnames_traj_in[i_traj]))
        i_current_frames = frame_idx_ordered[traj_idx_ordered == i_traj]
        assert np.all(i_current_frames < len(traj))
        if traj_out is None:
            traj_out = traj[i_current_frames]
        else:
            traj_out += traj[i_current_frames]

    print('saving coordinate data to', fname_traj_out)
    # Since we reordered the indices, we have to undo this here, to give back to the caller a file with
    # the actual input ordering that was requested.
    inv_order = np.argsort(order)
    traj_out[inv_order].save(fname_traj_out)


def import_trajectory(traj_fnames, branch, fields=util.All, swarm=True, end=-1, cvname='colvars', max_images=300, min_rmsd=0.3,
           mother_string=None, command=None, compress=True, stride=1, n_clusters=50000, lambada=None):
    r'''Import trajectory into Sarangi and turn it into a String object (make directory structure and plan file).

    Parameters
    ----------
    traj_fnames : str or list of str
        File name of the input trajectory. Can also be a list of file names, in which case the
        files are treated as parts of the same trajectory (in the exact order given by the list).
    branch : str
        Short label for the new string, e.g. 'AZ'. Can be any number of letters long.
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
        Stop adding images to the initial string, once the RMSD bottleneck reaches or drops below this value (in Angstrom).
    mother_string : sarangi.String (optional)
        String object, will copy basic configuration information from that string.
    command : str
        Executable that computes collective variables. Program must accept --id --cvname options.
        Can be omitted in which case the command will be taken from configuration information in the
        mother string. Either `command` or `mother_string` must be set.
    compress: bool
        If true, only import into the project the MD frames (full coordinares) that will actually be used as
        starting points for MD integration.
        If false, copy the whole input trajectories into the project (sub-)folder even if most frames are
        nerver used.
        # TODO: allow to input collective variable file directly
    stride : int
        Only use every n'th frame from the input trajectories.
    n_clusters : int
        Reduce maximum number of input data points to this number by running k-means clustering.
        If the number of input data points is lower than this number, directly use all of them.
    lambada : float or None
        Exponential scaling parameter lambda for the shortest path algorithm. If given, will
        override max_images and min_rmsd.

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

    cv_trajs = get_colvars(traj_fnames=traj_fnames, branch=branch, command=command, fields=fields, cvname=cvname, write_to_disk=not compress)
    #print('CV loader returned with %d trajs.' % len(cv_trajs))
    real_fields = cv_trajs[0].fields
    real_cv_dims = cv_trajs[0].dims
    # In the next few steps, we will concatenate the trajectories. But first make some indices that will allow
    # us to undo the concatenation.
    frame_indices = np.concatenate([np.arange(len(cv_traj))[::stride] for cv_traj in cv_trajs])[0:end]
    traj_indices = np.concatenate([np.zeros(len(cv_traj), dtype=int)[::stride] + i for i, cv_traj in enumerate(cv_trajs)])[0:end]
    #print('Indices of the all trajectories are:', np.unique(traj_indices))
    real_fields = cv_trajs[0].fields  # convert user selection (which may be All into a concrete list of field names)
    del fields
    cvs_linear = np.concatenate([cv_traj.as2D(fields=real_fields)[::stride] for cv_traj in cv_trajs])[0:end]
    #del cv_trajs

    if len(cvs_linear) > n_clusters:
        # TODO: test me!
        print('Collective variable trajectory contains many frames (>100000). Reducing data with k-means first.')
        import sklearn.cluster
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(cvs_linear)
        center_indices = [np.argmin(np.linalg.norm(cvs_linear - c, axis=1)) for c in kmeans.cluster_centers_]
        frame_indices = frame_indices[center_indices]
        traj_indices = traj_indices[center_indices]
        cvs_linear = kmeans.cluster_centers_

    # select images using soft-max capacity path algorithm
    if np.isinf(float(max_images)):
        image_indices_linear = np.arange(cvs_linear.shape[0])
    else:
        if lambada is None:
            print('finding connected path...')
            image_indices_linear = denoise(cvs_linear, max_images=max_images, min_rmsd=min_rmsd)
        else:
            print('finding connected path (in expert mode) ...')
            image_indices_linear, _ = denoise_once(cv=cvs_linear, lambada=lambada)
        #centers = cvs_linear[image_indices_linear, :]
        print('Length of path is %d. [%d, ..., %d]' % (len(image_indices_linear), image_indices_linear[0], image_indices_linear[-1]))

    # import frames into the project ...
    util.mkdir('{root}/strings/{branch}_000/'.format(root=util.root(), branch=branch))
    if not compress:
        # ... either as complete trajectories
        print('Importing frames in compressed from. Note that if the collective variables space is changed later, reimporting might be needed.')
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
                       traj_idx_unordered=traj_indices[image_indices_linear], frame_idx_unordered=frame_indices[image_indices_linear])
        # colvars are outdated now, we need to recreate the file
        # TODO: delete temporary cv files ...
        #fname_cv_out = '{root}/observables/{branch}_000/{cvname}/{branch}_000_000_000.npy'.format(root=util.root(), branch=branch, cvname=cvname)
        # TODO: should we overwrite ...
        # np.save(fname_cv_out, util.flat_to_structured(cvs_linear, fields=real_fields, dims=real_cv_dims))
        print('checking self-consistecy of the imported (compressed) coordinate data...')
        # as a side effect the next line writes a collective variable file that corresponds to the starting structures into the project directory structure
        ok = self_consistency_compressed(fname_traj_out, branch=branch, command=command, fields=real_fields, cvname=cvname,
                 cv_centers_linear=cvs_linear[image_indices_linear], frame_indices=frame_indices[image_indices_linear], traj_indices=traj_indices[image_indices_linear])
        if ok:
            print('OK')
        else:
            raise RuntimeError('Self-consistency test of coordinate data failed. Imported frames are not identical to the ones selected by the shortest path algorithm. Something went wrong.')

    if mother_string is not None:
        new_string = mother_string.empty_copy(iteration=1, branch=branch)
    else:
        new_string = sarangi.String(branch=branch, iteration=1, images={}, image_distance=min_rmsd, previous=None,
                                    colvars_def=None, opaque={'observables': [{'command': command, 'name': cvname}]})

    # create String object / plan file
    for running_index, image_index_linear in enumerate(image_indices_linear):
        # get node in structured format; cv_trajs variable always refers to the original (uncompressed) format
        i = frame_indices[image_index_linear]
        node = cv_trajs[traj_indices[image_index_linear]]._colvars[i:i + 1]
        if compress:
            frame_index_out = running_index  # just renumber sequentially
            traj_index_out = 0  # there is only one "trajectory" i.e. file with all input coordinates
        else:
            frame_index_out = frame_indices[image_index_linear]
            traj_index_out = traj_indices[image_index_linear]
        # test self-consistency of nodes
        if not np.allclose(util.structured_to_flat(node, fields=real_fields), cvs_linear[image_index_linear, np.newaxis, :]):
            #print(running_index, traj_index, )
            print('structured:', util.structured_to_flat(node, fields=real_fields))
            print('unstructured:', cvs_linear[image_index_linear, np.newaxis, :])
            raise RuntimeError('The structured and the unstructured representations of the CV data became inconsistent. This should never happen. Stopping plan generation.')
        spring = util.dict_to_structured({name: 10. for name in real_fields})
        im = image.CompoundImage(image_id='{branch}_001_{index:03d}_000'.format(branch=branch, index=running_index),
                                 previous_image_id='{branch}_000_000_{traj_index:03d}'.format(branch=branch, traj_index=traj_index_out),
                                 previous_frame_number=frame_index_out, group_id=None, node=node, spring=spring, swarm=swarm)
        new_string.add_image(im)
    # save String object / plan file to disk
    time = datetime.datetime.strftime(datetime.datetime.now(), '%Y/%m/%d %H:%M')
    new_string.write_yaml(
        message='{time}: New string from initialized from trajectory "{traj_fname}" ...'.format(time=time, traj_fname=traj_fnames[0]))
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


def gss(f, a, b, tol=1e-3, goal=float('-inf'), max_iter=float('inf'), verbose=False):
    'Minimize the function given by f(x)[0] under the condition f(x)[1] (Golden section search).'
    import math
    hist_x = []
    hist_f = []
    hist_cond = []
    hist_aux = []
    gr = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc, cond_c, aux_c = f(c)
    fd, cond_d, aux_d = f(d)
    iter_ = 0
    while abs(fc - fd) > tol or not (cond_c or cond_d):
        fc, cond_c, aux_c = f(c)  # TODO: move down
        fd, cond_d, aux_d = f(d)  # TODO: move down
        if fc <= goal and cond_c:
            print('Reached precision goal.')
            return c, aux_c, hist_x, hist_f, hist_cond, hist_aux
        if fd <= goal and cond_d:
            print('Reached precision goal.')
            return d, aux_d, hist_x, hist_f, hist_cond, hist_aux
        # This is the golden section algorithm with a single modification:
        # For large x, we will typically violate the constraint. The
        # violation can be resolved by decreasing x. Therefore we select
        # the left interval, if the point in the middle of the right interval
        # is infeasible. That's all. No such reasoning is needed for violations
        # in the left interval, since we can always go further left.
        if fc < fd or not cond_d:  # the "or not cond_d" expression is the modification to the original GSS algorithm
            b = d
            hist_x.append(d)
            hist_f.append(fd)
            hist_cond.append(cond_d)
            hist_aux.append(aux_d)
        else:
            a = c
            hist_x.append(c)
            hist_f.append(fc)
            hist_cond.append(cond_c)
            hist_aux.append(aux_c)

        if iter_ > max_iter:
            break

        c = b - (b - a) / gr
        d = a + (b - a) / gr
        iter_ += 1

    if iter_ > max_iter:
        print('Reached maximum number of iterations.')
    else:
        if cond_c and not cond_d:
            print('Converged to the constraint boundary.')
        else:
            print('Found minimum off the constraint boundary.')

    if (fc < fd and cond_c and cond_d) or (cond_c and not cond_d):
        return c, aux_c, hist_x, hist_f, hist_cond, hist_aux
    else:
        return d, aux_d, hist_x, hist_f, hist_cond, hist_aux


def denoise(cv, max_images=200, min_rmsd=0.1, max_iter=float('inf'), verbose=True):
    def f_(x):
        geomf = (cv.shape[1]/3.)**-0.5
        p = shortest_path(cv, 0, cv.shape[0] - 1, x*geomf, logspace=True)
        delta = [np.linalg.norm(cv[i, :] - cv[j, :]) for i, j in zip(p[0:-1], p[1:])]
        score = np.mean(delta)*geomf
        cond = (len(p) < max_images)
        if verbose:
            print('Path optimzation with parameter %f produced an average rmsd of %f and path length of %d.'%(x, score, len(p)))
        return score, cond, p
    mem = {}  # TODO: implement better solution
    def f_memoized(x):
        if x in mem:
            return mem[x]
        else:
            y = f_(x)
            mem[x] = y
            return y
    l, result, _, _, _, _ = gss(f_memoized, 0.05, 200, goal=min_rmsd, max_iter=max_iter, verbose=verbose)
    return result


def self_consistency_compressed(traj_fname, branch, command, fields, cvname, cv_centers_linear, frame_indices, traj_indices):
    # this type of check is only run, if we are "compressing" i.e. importing individual frames, so we can rewrite the cv file
    cv = get_colvars([traj_fname], branch=branch, command=command, fields=fields, cvname=cvname, write_to_disk=True, use_existing_file=False)[0]
    #print('shape of the read-back colvars is:', cv.as2D(fields=fields).shape)
    #print('expected shape is:', cv_centers_linear.shape)
    is_ok = np.allclose(cv.as2D(fields=fields), cv_centers_linear, atol=1E-6)
    if not is_ok:
        recomputed = cv.as2D(fields=fields)
        for i, (i_frame, i_traj) in enumerate(zip(frame_indices, traj_indices)):
            print(i_traj, i_frame, 'error:', np.max(np.abs(recomputed[i, :] - cv_centers_linear[i, :])))
    return is_ok


def parse_args_import_trajectory(argv=None):
    import argparse
    import sys
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Make new string by importing trajectory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--branch', required=True, help='Short label for the new string, e.g. \'AZ\'')
    parser.add_argument('--noswarm', action='store_true',
                        help='By default, propagate using the swarm of trajectories method. If set, use permanent biases (umbrella sampling).')
    parser.add_argument('--end', default=-1, type=int,
                        help='stop looking at the data past this frame (default=-1, use all frame)')
    parser.add_argument('--max_images', default=300,
                        help='New string will not contain more that this number of images. Set to inf to use all frames in the input data.')
    parser.add_argument('--min_rmsd', default=0.3, type=float,
                        help='Stop adding images to the intial string, once the RMSD bottleneck reaches this value (in Angstrom).')
    parser.add_argument('--command', type=str,
                        help='Executable that computes collective variables. Program must accept --id --cvname options.')
    parser.add_argument('--mother_string', type=str,
                        help='String object, will copy basic configuration information from that string.')
    parser.add_argument('--cvname', default='colvars', type=str,
                        help='Subfolder name where collective variables are stored.')
    parser.add_argument('--fields', nargs='+', type=str, default=['All'],
                        help='Selection of collective variable names. If All is given, use all colvars that the command outputs.')
    parser.add_argument('--nocompress', action='store_true',
                        help='By default, import into the project only the frames that are needed for propagation. If set, import all data.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Only cosider every n\'th frame in the input trajectory.')
    parser.add_argument('--nclusters', default=50000, type=int,
                        help='Maximum number of input data points to use in path search. If input contains more points, reduce with k-means clustering.')
    parser.add_argument('--lambada', default=None, type=float,
                        help='Exponential scaling parameter for the shortest path algorithm. If given, will override max_images and min_rmsd.')
    parser.add_argument('trajname', nargs='+', type=str,
                        help='File name of the input trajectory. If more than one name is given, contents is concatenated (in the order of this list)')

    args = parser.parse_args(argv)

    if args.mother_string is None:
        mother_string = None
    else:
        from sarangi import String
        mother_string = String.load(args.mother_string)  # TODO: currently can't pass iteration number, FIXME

    if args.fields == ['All']:
        from sarangi import util
        fields = util.All
    else:
        fields = args.fields

    args = {'traj_fnames': args.trajname, 'branch': args.branch, 'fields': fields, 'swarm': not args.noswarm, 'end': args.end,
            'cvname': args.cvname, 'max_images': args.max_images, 'min_rmsd': args.min_rmsd, 'mother_string': mother_string,
            'command': args.command, 'compress': not args.nocompress, 'stride': args.stride, 'n_clusters': args.nclusters,
            'lambada': args.lambada}
    return args


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