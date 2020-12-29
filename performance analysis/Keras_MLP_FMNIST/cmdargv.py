import os
import time
import argparse


def parse_argv():
    parser = argparse.ArgumentParser()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    parser.add_argument('-hidden', nargs='+', help='configure size of hidden layer ex. 256',
                        type=int, metavar='units', default=128)

    parser.add_argument('-l', '--log', help='save detailed trainning log (.csv file)',
                        dest='log_path', metavar='path')  # default: see below
    parser.add_argument('-m', '--model', help='save trainned moldel (.h5)',
                        dest='model_path', metavar='path')  # default: see below
    parser.add_argument('-s', '--statistics', help='where to store statistics file (.csv)',
                        dest='statistics_path', metavar='path', default='statistics.csv')

    parser.add_argument('--allow_growth', help='whether to set allow_growth for TensorFlow',
                        action='store_true', default=False)
    parser.add_argument('--fp16', help='whether to use mixed_precision / mixed_float16 training',
                        action='store_true', default=False)

    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = '{num_hidden}_{timestamp}.csv'.format(
            num_hidden='_'.join(map(lambda x: str(x), args.hidden)), timestamp=timestamp
        )

    if args.model_path is None:
        args.model_path = '{num_hidden}_{timestamp}.h5'.format(
            num_hidden='_'.join(map(lambda x: str(x), args.hidden)), timestamp=timestamp
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
