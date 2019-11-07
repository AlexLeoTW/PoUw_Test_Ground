import os
import sys
import time
import argparse


def error_and_exit(parser, msg):
    parser.print_usage()
    print('{prog}: error: {msg}'.format(prog=sys.argv[0], msg=msg))
    sys.exit(1)


def parse_argv(argv):
    parser = argparse.ArgumentParser()
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

    parser.add_argument('--aug', help='whether to randomly augment images during training',
        action='store_true', default=False)
    parser.add_argument('-conv', help='configure Conv2D layer ex. 32 4',
        nargs='+', type=int, metavar=('filters', 'kernel_size'), default=[32, 4])
    parser.add_argument('-conv_num', help='how many Conv2D layers',
        type=int, metavar='num', default=4)
    parser.add_argument('-pool', help='configure MaxPooling2D pool_size ex. 2',
        type=int, metavar='size', default=2)
    parser.add_argument('--stack', help='how Conv2D layers are stacked',
        choices=['independent', '2_in_a_row'], metavar='type', default='2_in_a_row')
    # parser.add_argument('-dense', help='configure units of dense layer ex. 128',
    #     type=int, metavar='units', default=128)
    parser.add_argument('-l', '--log', help='save detailed trainning log (.csv file)',
        dest='log_path', metavar='path')
    parser.add_argument('-m', '--model', help='save trainned moldel (.h5)',
        dest='model_path', metavar='path')
    parser.add_argument('-s', '--statistics', help='where to store statistics file (.csv)',
        dest='statistics_path', metavar='path', default='statistics.csv')

    parser.add_argument('--allow_growth', help='whether to set allow_growth for TensorFlow',
        action='store_true', default=False)

    args = parser.parse_args()

    if args.conv_num < 1:
        error_and_exit(parser, 'conv_num must gratter then 0')

    if args.stack == '2_in_a_row' and args.conv_num % 2 != 0:
        error_and_exit(parser, 'conv_num must be multiples of 2 when stacking mode \"2_in_a_row\"')

    if args.log_path is None:
        args.log_path = '{aug}_{filters}_{kernel_size}_{conv_num}_{pool}_{stack}_{timestamp}.csv'.format(
            aug='aug' if args.aug else 'none',
            filters=args.conv[0], kernel_size=args.conv[1], conv_num=args.conv_num,
            pool=args.pool, stack=args.stack,
            timestamp=timestamp
        )

    if args.model_path is None:
        args.model_path = '{aug}_{filters}_{kernel_size}_{conv_num}_{pool}_{stack}_{timestamp}.h5'.format(
            aug='aug' if args.aug else 'none',
            filters=args.conv[0], kernel_size=args.conv[1], conv_num=args.conv_num,
            pool=args.pool, stack=args.stack,
            timestamp=timestamp
        )

    args.log_path = os.path.abspath(args.log_path)
    args.model_path = os.path.abspath(args.model_path)
    args.statistics_path = os.path.abspath(args.statistics_path)

    return args


def main(argv):
    options = parse_argv(argv)
    print(options)


if __name__ == '__main__':
    main(sys.argv)