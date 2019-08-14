import numpy as np
import os
import subprocess
from .short_path import path as shortest_path
from . import colvars
from . import image
from . import util
from . import sarangi


def spinup(fname_traj, branch, fields=util.All, swarm=True, end=-1, cvname='colvars', max_images=200, min_rmsd=0.3,
           mother_string=None, command=None):
    r'''Import trajectory into Sarangi and turn it into a string.

    Parameters
    ----------
    fname_traj : str
        File name of the input trajectory.
    branch : str
        Short label for the new string, e.g. 'AZ'
    fields : list or All
        Selection of collective variable names. If All is given, use all colvars that the command outputs.
    swarm : bool
        If true, propagate using the swarm of trajectories method. If false, used permanent biases (umbrella sampling).
    end : int
        stop looking at the data past this frame (default=-1, use all frame)
    cvname : str
        Subfolder name where collective variables are stored.
    max_images : int
        New string will not contain more that this number of images.
    min_rmsd : float
        Stop adding images to the intial string, once the RMSD bottleneck reaches this value (in Angstrom).
    mother_string : sarangi.String
        String object, will copy basic configuration information from that string.
    command : str
        Executable that computes collective variables. Program must accept --id --cvname options.
        # TODO: allow to input collective variable file directly
    '''
    # TODO do some concatenation? (first convert to cvs, then concatenate; don't forget to translate indices)

    if mother_string is not None:
        command = [obs['command'] for obs in mother_string.opaque['observables'] if obs['name'] == 'colvars']
    else:
        if command is None:
            raise ValueError('One of the parameters "command" or "mother_string" must be set.')
    # file name for initial colvars 'observables/AZ_000/AZ_000_000_000.npy' etc.

    cv_folder = '{root}/observables/{cvname}/{branch}_000'.format(root=util.root(), cvname=cvname, branch=branch)
    image_id = '{branch}_000_000_000'.format(branch=branch)

    for ext in ['.npy', '.npz', '.colvars.traj']:
        cv_fname = cv_folder + '/' + image_id + ext
        if os.path.exists(cv_fname):
            break
    else:  # file not found
        full_command = '{command} --id {image_id} --cvname {cvname} {fname_traj}'.format(command=command,
                                                                                         cvname=cvname,
                                                                                         image_id=image_id,
                                                                                         fname_traj=fname_traj)
        util.mkdir(cv_folder)
        print('running', full_command)
        env = {'STRING_SIM_ROOT': util.root(), }  # TODO: 'STRING_SARANGI_SCRIPTS' and more?
        env.update(os.environ)
        subprocess.run(full_command, shell=True, env=env)

    cv = colvars.Colvars(
            folder=cv_folder,
            base='{branch}_000_000_000.npy'.format(branch=branch),
            fields=fields)

    image_indices = denoise(cv.as2D(fields=fields)[0:end, :], max_images=max_images, min_rmsd=min_rmsd)
    centers = [cv[i] for i in image_indices]

    if mother_string is not None:
        new_string = mother_string.empty_copy(iteration=0, branch=branch)
    else:
        new_string = sarangi.String(branch=branch, iteration=0, images={}, image_distance=min_rmsd, previous=None,
                                    colvars_def=None, opaque=None)

    # from scrach:
    # dtype_spring = [(name, np.float64) for name in centers.dtype.names]
    # spring = np.zeros(1, dtype=dtype_spring)
    #for name in centers.dtype.names:
    #    spring[name] = 3
    for running_index, (center, index) in enumerate(zip(centers, image_indices)):
        node = np.zeros(1, dtype=center.dtype)
        node[0] = center
        spring = util.load_structured({name: 10. for name in cv.fields})
        im = image.CompoundImage(image_id='{branch}_001_{index:03d}_000'.format(branch=branch, index=running_index),
                                 previous_image_id='{branch}_000_000_000'.format(branch=branch),
                                 previous_frame_number=index, group_id=None, node=node, spring=spring, swarm=swarm)
        new_string.add_image(im)

    new_string.write_yaml(
        message='New string from initialized from trajectory "{fname_traj}".'.format(fname_traj=fname_traj))
    return new_string


def denoise_once(cv, param):
    '''Denoise trajectory with modified shortest-path algorithm that discourages large jumps in output

    Parameters
    ----------
    cv : np.ndarray((T, n))

    param : float
        level of detail, the higher, the more details is kept (less denoising)
        The distance in the shortest-path algorithm is computed as
        dist(i, j) = exp(param*||x_i - x_j||*geom) - 1
        where geom = 1./sqrt(n).

    Returns
    -------
    list of integers : [i1, i2, ...]
    indices such that [cv[i1, :], cv[i2, :], ...] are the denoised trajectory
    '''
    dim = cv.shape[1]
    path = shortest_path(cv, 0, cv.shape[0] - 1, param=param*dim**-0.5)
    jumps = []
    for a, b in zip(path[0:-1], path[1:]):
        d = np.exp(np.linalg.norm(a - b)*param*dim**-0.5)
        jumps.append(d)
    return path, jumps


def denoise(cv, max_images, min_rmsd, max_iter=20):
    param = 1.0  # starting value
    lowest_infeasible_top_bound = 1000.  # smallest upper bound on the parameter, such that the constraints are fulfilled
    highest_feasible_bottom_bound = 0.  # highest lower bound on the parameter in the infeasible region
    path = [0, len(cv)]

    for _ in range(max_iter):
        path, jumps = denoise_once(cv, param)
        error = np.median(jumps)
        n_images = len(path)
        print('Denoising attempt with parameter {param} yielded {n_images} and median jump {median}.'.format(
            param=param, n_images=n_images, median=error))

        feasible = error >= min_rmsd and n_images <= max_images
        if feasible:
            if param > highest_feasible_bottom_bound:  # if we found a new upper boundary, adjust it
                highest_feasible_bottom_bound = param
        else:  # infeasible
            if param < lowest_infeasible_top_bound:  # if we found a new lower boundary, adjust it
                lowest_infeasible_top_bound = param

        if feasible and abs(n_images - max_images) < max_images/10. and abs(error - min_rmsd) < min_rmsd/10.:
            # feasible and good enough -> done
            return path

        if feasible:  # feasible but not good
            # increase param
            param = 0.5*(lowest_infeasible_top_bound + param)
        else:  # neither good nor feasible
            # decrease params
            param = 0.5*(highest_feasible_bottom_bound + param)

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
