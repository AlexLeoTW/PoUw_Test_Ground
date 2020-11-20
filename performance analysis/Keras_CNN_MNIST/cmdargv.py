import os
import time
import argparse


def parse_argv():
    parser = argparse.ArgumentParser()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    parser.add_argument('-conv1', help='configure first Conv2D layer ex. 32 2',
        nargs='+', type=int, metavar=('filters', 'kernel_size'), default=[32, 2])
    parser.add_argument('-conv2', help='configure second Conv2D layer ex. 64',
        type=int, metavar='filters', default=64)
    parser.add_argument('-pool', help='configure MaxPooling2D pool_size ex. 2',
        type=int, metavar='pool_size', default=2)
    parser.add_argument('-dense', help='configure units of dense layer ex. 128',
        type=int, metavar='units', default=128)
    parser.add_argument('-l', '--log', help='save detailed trainning log (.csv file)',
        dest='log_path', metavar='path')  # default: defined below
    parser.add_argument('-m', '--model', help='save trainned moldel (.h5)',
        dest='model_path', metavar='path')  # default: defined below
    parser.add_argument('-s', '--statistics', help='where to store statistics file (.csv)',
        dest='statistics_path', metavar='path', default='statistics.csv')

    parser.add_argument('--allow_growth', help='whether to set allow_growth for TensorFlow',
            action='store_true', default=False)
    parser.add_argument('--fp16', help='whether to use mixed_precision / mixed_float16 training',
        action='store_true', default=False)

    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = '{}_{}_{}_{}_{}_{}.csv'.format(
            args.conv1[0],  # conv1_filters
            args.conv1[1],  # conv1_kernel_size
            args.conv2, args.pool, args.dense, timestamp
        )

    if args.model_path is None:
        args.model_path = '{}_{}_{}_{}_{}_{}.h5'.format(
            args.conv1[0],  # conv1_filters
            args.conv1[1],  # conv1_kernel_size
            args.conv2, args.pool, args.dense, timestamp
        )

    args.log_path = os.path.abspath(args.log_path)
    args.model_path = os.path.abspath(args.model_path)
    args.statistics_path = os.path.abspath(args.statistics_path)

    return args


def main():
    options = parse_argv()
    for key in options.__dict__:
        print('{}: {}'.format(key, options.__dict__[key]))


if __name__ == '__main__':
    main()
