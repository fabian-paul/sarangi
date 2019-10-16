from sarangi import root
from sarangi.colvars import overlap_svm, Colvars

__all__ = ['main_overlap', 'parse_args_overlap']


def compute_and_encode_overlap(a: Colvars, b: Colvars):
    res = {}
    for field in a.fields:
        res[field] = overlap_svm(a[field], b[field])
    return res


def main_overlap(image_id):
    import os
    import yaml
    branch, iteration, id_major, id_minor = image_id.split('_')
    fname_colvars = '{root}/observables/{branch}_{iteration:03d}'.format(
        root=root(), branch=branch, iteration=int(iteration))
    results = {}
    for subdir in os.listdir(fname_colvars):
        if os.path.isdir(subdir):
            a = Colvars(folder=fname_colvars + '/' + subdir, base=image_id)

            # now loop over all bs
            other_images = os.listdir(fname_colvars + '/' + subdir)
            for other_image in other_images:
                other_image_id = os.path.splitext(other_image)[0]
                b = Colvars(folder=fname_colvars + '/' + subdir, base=other_image_id)

                results[other_image_id] = compute_and_encode_overlap(a, b)

    fname_out = '{root}/overlap/{branch}_{iteration:03d}/{branch}_{iteration:03d}_{id_major:03d}_{id_minor:03d}.yaml'.format(
        root=root(), branch=branch, iteration=int(iteration), id_minor=int(id_minor), id_major=int(id_major))

    with open(fname_out, 'w') as f:
        yaml.dump(results, f, width=1000, default_flow_style=None)


def parse_args_overlap(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='compute overlap in collective variable space', prog='overlap',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image', metavar='ID', required=True,
                        help='image ID in format <branch>_<iteration>_<major>_<minor>')

    args = parser.parse_args(args=argv)

    return {'image': args.image}


if __name__ == '__main__':
    main_overlap(**parse_args_overlap())
