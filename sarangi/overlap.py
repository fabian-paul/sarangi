from sarangi import root
from sarangi.colvars import overlap_svm, Colvars
from sarangi.util import exactly_2d

__all__ = ['main_overlap', 'parse_args_overlap']


def load_matrix(branch: str, iteration: int, subdir: str='colvars', reduction='min', return_image_ids=False):
    r'''Get the full overlap matrix of all images in the string.

    Parameters
    ----------
    branch: str
        branch name
    iteration: int
        interation number
    subdir: str
        subdirectory of overlap folder (usually corresponds to the subdir
        of the name name in the observables folder)
    reduction: str
        Currently only 'min' is supported. Computes the minimum overlap
        of all observables for each pair if images.
    return_image_ids: bool
        return list of image_id in addition of to the overlap matrix
        Since this procedure loads precomputed data form disk, it is worth to
        check this return value to test if the list is complete.

    Return
    ------
    Depending on the value of return_image_ids:
    * matrix
    * matrix, image_id

    Undefined elements of matrix are set to nan (if data was not found on disk).
    '''
    import os
    import csv
    import numpy as np
    matrix = {}
    seen_image_ids = set()
    folder_overlap = '{root}/overlap/{branch}_{iteration:03d}/{subdir}'.format(
        root=root(), branch=branch, iteration=int(iteration), subdir=subdir)

    for fname in os.listdir(folder_overlap):
        if not fname.endswith('.csv'):
            continue
        image_id_a = os.path.splitext(fname)[0]
        seen_image_ids.add(image_id_a)
        with open(folder_overlap + '/' + fname) as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            data = list(reader)
        for line in data:
            image_id_b = line[0]
            seen_image_ids.add(image_id_b)
            overlap = [float(x) for x in line[1:]]
            # TODO: should we support using no reductions, so that we return a tensor?
            if reduction == 'min':
                matrix[(image_id_a, image_id_b)] = np.min(overlap)
            else:
                raise RuntimeError('reduction not implemented')

    # convert to numpy array
    seen_image_ids = sorted(seen_image_ids)
    n = len(seen_image_ids)
    numpy_matrix = np.zeros((n, n)) + np.nan
    for i, a in enumerate(seen_image_ids):
        for j, b in enumerate(seen_image_ids):
            if (a, b) in matrix:
                numpy_matrix[i, j] = matrix[(a, b)]
            elif (b, a) in matrix:
                numpy_matrix[i, j] = matrix[(b, a)]
            numpy_matrix[j, i] = numpy_matrix[i, j]
    if return_image_ids:
        return numpy_matrix, seen_image_ids
    else:
        return numpy_matrix
    # TODO; have some numpy format...
    #symm_matrix = {}
    #for key, overlap in matrix.items():
    #    symm_matrix[key] = overlap
    #    symm_matrix[(key[1], key[0])] = overlap
    #


def load_tensor(branch: str, iteration: int, subdir: str='colvars'):
    raise NotImplementedError('load_tensor not implemented yet')


def load_pair(image_id_a: str, image_id_b: str, subdir='colvars', try_swapped: bool=True):
    import csv
    branch, iteration, _, _ = image_id_a.split('_')
    fname = '{root}/overlap/{branch}_{iteration:03d}/{subdir}/{image_id}.csv'.format(
        root=root(), branch=branch, iteration=int(iteration), subdir=subdir, image_id=image_id_a)
    try:
        with open(fname) as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            data = list(reader)
        for line in data:
            if line[0] == image_id_b:
                return {h.strip(): float(o) for h, o in zip(headers[1:], line[1:])}
    except FileNotFoundError:
        pass  # pass and fall through
    # did not find image_id_b
    if try_swapped:
        return load_pair(image_id_b, image_id_a, subdir=subdir, try_swapped=False)
    else:
        raise FileNotFoundError('Could not find overlap recorded for image pair %s, %s.' % (image_id_a, image_id_b))


def compute_and_encode_overlap(a: Colvars, b: Colvars):
    res = {}
    for field in a.fields:
        res[field] = float(overlap_svm(exactly_2d(a[field], add_frame_axis=False),
                                       exactly_2d(b[field], add_frame_axis=False)))
    return res


def write_csv(fname: str, data: dict):
    some_image_id = next(iter(data.keys()))
    fields = data[some_image_id].keys()
    with open(fname, 'w') as f:
        f.write('image_id, ' + ', '.join(fields) + '\n')
        for image_id, overlap in data.items():
            line = ', '.join(['%f'%data[image_id][field] for field in fields])
            f.write(image_id + ', ' + line + '\n')


def without_ext(fname: str):
    import os
    return os.path.join(os.path.dirname(fname), os.path.basename(fname).split(os.extsep)[0])


def file_is_up_to_date(new: str, old: list):
    import os
    if not os.path.exists(new):
        return False
    new_mtime = os.path.getmtime(new)
    for fname_old in old:
       if os.path.exists(fname_old):
           old_mtime = os.path.getmtime(fname_old)
           if old_mtime > new_mtime:
               return False
    return True


def main_overlap(branch: str, iteration: int, image_id: str=None, subdir: str=None, refresh: bool=False):
    r'''Precomputes the overlap between between one image and all others. Stores one csv file per image.

    Parameters
    ----------
    branch : str
        Branch name
    iteration : int
        String iteration number
    image_id : str or None
        identifier of image to compute overlap for
        Can be None in which case, all images in the given combintation of
        branch_iteration will be processed.
    subdir : str
        subdir of $STRING_SIM_ROOT/observables/<branch>_<iteration>/
        Restrict computation to this subdir
    refresh : bool
        If False, do not recreate files that are already up to date.
    '''
    import os
    from sarangi.util import mkdir

    folder_obs = '{root}/observables/{branch}_{iteration:03d}'.format(
        root=root(), branch=branch, iteration=int(iteration))

    if subdir is not None:
        subdirs = [subdir]
    else:
        subdirs = os.listdir(folder_obs)
    for subdir_ in subdirs:
        if os.path.isdir(folder_obs + '/' + subdir_) and '_init' not in subdir_:
            folder_out = '{root}/overlap/{branch}_{iteration:03d}/{subdir}'.format(root=root(), branch=branch, iteration=int(iteration), subdir=subdir_)
            mkdir(folder_out)
            images = os.listdir(folder_obs + '/' + subdir_)
            for image in images:
                image_id_ = without_ext(image)
                if image_id is not None and image_id_ != image_id:
                    continue  # if user selected specific image id, skip all others
                fname_image = folder_obs + '/' + subdir_ + '/' + image
                fname_out = '{folder_out}/{image_id}.csv'.format(folder_out=folder_out, image_id=image_id_)
                other_images = os.listdir(folder_obs + '/' + subdir_)
                fnames_other_images = [folder_obs + '/' + subdir_ + '/' + f for f in other_images]
                if file_is_up_to_date(fname_out, fnames_other_images + [fname_image]) and not refresh:
                    continue  # if results are already on disk and we were not asked to refresh, skip
                a = Colvars(folder=folder_obs + '/' + subdir_, base=image_id_)
                results = {}
                # now loop over all bs
                debug_count = 0
                for other_image in other_images:
                    other_image_id = without_ext(other_image)
                    if other_image_id == image_id_:  # *Do not* restrict to one triangle of the matrix, since we might be running this on incomplete data 
                        continue
                    b = Colvars(folder=folder_obs + '/' + subdir_, base=other_image_id)
                    print('processing', image_id_, '<->', other_image_id, '(', subdir_, ')', len(b), 'frames')

                    results[other_image_id] = compute_and_encode_overlap(a, b)
                    debug_count += 1
                    #if debug_count > 1:
                    #    break
                if len(results) > 0:
                    write_csv(fname_out, results)

        #fname_out2 = '{folder_out}/{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}.yaml'.format(
        #    folder_out=folder_out, branch=branch, iteration=int(iteration), id_minor=int(id_minor), id_major=int(id_major))
        #with open(fname_out2, 'w') as f:
        #    yaml.dump(results, f, width=1000, default_flow_style=None)


def parse_args_overlap(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='compute overlap in collective variable space', prog='overlap',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--branch', metavar='ID',
                        help='branch id of images to compute overlap for')
    parser.add_argument('--iteration', metavar='number',
                        help='iteration number of images to compute overlap for')
    parser.add_argument('--image', metavar='ID',
                        help='Only compute overlap for the image with specific ID. ID format is <branch>_<iteration>_<major>_<minor>. If given, overrides --branch and --iteration.')
    parser.add_argument('--refresh', action='store_true',
                        help='Even if overlap file already exist, recompute.')
    parser.add_argument('--subdir', metavar='name',
                        help='restrict overlap computation to colvars in subdir; be default loop over all subdirs')

    args = parser.parse_args(args=argv)

    if args.image is not None:
        branch, iter_, _, _ = args.image.split('_')
        iter_ = int(iter_)
    else:
        if args.branch is None or args.iteration is None:
            raise ValueError('Must either provide --image or the pair of --branch and --iteration.')
        branch = args.branch
        iter_ = int(args.iteration)

    return {'image_id': args.image,
            'branch': branch,
            'iteration': iter_,
            'refresh': args.refresh,
            'subdir': args.subdir}

if __name__ == '__main__':
    main_overlap(**parse_args_overlap())
