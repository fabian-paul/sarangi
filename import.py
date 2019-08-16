from sarangi.generation import import_trajectory
import argparse

parser = argparse.ArgumentParser(description='Make new string by importing trajectory',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--branch', required=True, help='Short label for the new string, e.g. \'AZ\'')
parser.add_argument('--noswarm', action='store_true',
                    help='By default, propagate using the swarm of trajectories method. If set, use permanent biases (umbrella sampling).')
parser.add_argument('--end', default=-1, type=int,
                    help='stop looking at the data past this frame (default=-1, use all frame)')
parser.add_argument('--max_images', default=300, type=int,
                    help='New string will not contain more that this number of images.')
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

args = parser.parse_args()

if args.mother_string is None:
    mother_string = None
else:
    from sarangi import String
    mother_string = String.load(args.mother_string)

if args.fields == ['All']:
    from sarangi import util
    fields = util.All
else:
    fields = args.fields

import_trajectory(args.trajname, args.branch, fields=fields, swarm=not args.noswarm, end=args.end, cvname=args.cvname,
                  max_images=args.max_images, min_rmsd=args.min_rmsd, mother_string=mother_string, command=args.command,
                  compress=not args.nocompress, stride=args.stride, n_clusters=args.nclusters, lambada=args.lambada)


