import os
import time
import argparse


def parse_argv():
    parser = argparse.ArgumentParser()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    parser.add_argument('-nd', '--noise_dim', metavar='dim', type=int, default=100,
                        help='how much (random)info is used to generate animage')
    parser.add_argument('-gl1', '--gen_l1', metavar='filters', type=int, default=128,
                        help="configure generator's first Conv2DTranspose layer")
    parser.add_argument('-gl2', '--gen_l2', metavar='filters', type=int, default=64,
                        help="configure generator's second Conv2DTranspose layer")

    parser.add_argument('-dl1', '--disc_l1', metavar='filters', type=int, default=64,
                        help="configure discriminator's first Conv2D layer")
    parser.add_argument('-dl2', '--disc_l2', metavar='filters', type=int, default=128,
                        help="configure discriminator's second Conv2D layer")

    parser.add_argument('-l', '--log', help='save detailed trainning log (.csv file)',
        dest='log_path', metavar='path')  # default: defined below
    parser.add_argument('-gm', '--gen_model', help='save trainned moldel (.h5)',
        dest='gen_model_path', metavar='path')  # default: defined below
    parser.add_argument('-dm', '--disc_model', help='save trainned moldel (.h5)',
        dest='disc_model_path', metavar='path')  # default: defined below
    parser.add_argument('-i', '--img_dir', help='save pics from trained models every epoch',
        dest='img_dir', metavar='path')  # default: defined below
    parser.add_argument('-s', '--statistics', help='where to store statistics file (.csv)',
        dest='statistics_path', metavar='path', default='statistics.csv')

    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = f'{args.noise_dim}__{args.gen_l1}_{args.gen_l2}__{args.disc_l1}_{args.disc_l2}__{timestamp}.csv'

    if args.gen_model_path is None:
        args.gen_model_path = f'{args.noise_dim}__{args.gen_l1}_{args.gen_l2}__{args.disc_l1}_{args.disc_l2}__{timestamp}_gen.h5'

    if args.disc_model_path is None:
        args.disc_model_path = f'{args.noise_dim}__{args.gen_l1}_{args.gen_l2}__{args.disc_l1}_{args.disc_l2}__{timestamp}_disc.h5'

    if args.img_dir is None:
        args.img_dir = f'{args.noise_dim}__{args.gen_l1}_{args.gen_l2}__{args.disc_l1}_{args.disc_l2}__{timestamp}'

    args.log_path = os.path.abspath(args.log_path)
    args.gen_model_path = os.path.abspath(args.gen_model_path)
    args.disc_model_path = os.path.abspath(args.disc_model_path)
    args.statistics_path = os.path.abspath(args.statistics_path)

    return args


def main():
    options = parse_argv()
    for key in options.__dict__:
        print('{}: {}'.format(key, options.__dict__[key]))


if __name__ == '__main__':
    main()
