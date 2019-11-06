import argparse
import sys
import os
import time

data_path = os.path.join(os.path.dirname(__file__), 'dataset')


def parse_argv(argv):
    parser = argparse.ArgumentParser()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    parser.add_argument('-t', '--train', help='enable trainning', action='store_true')
    parser.add_argument('-e', '--eval', help='enable test', action='store_true')
    parser.add_argument('--step', help='step size', default=30, type=int,
                        metavar='num_steps', dest='num_steps')
    parser.add_argument('--batch', help='trainning batch size', default=20, type=int,
                        metavar='batch_size', dest='batch_size')
    parser.add_argument('--embd', help='embedding vector size & first LSTM hidden layer size',
                        type=int, metavar='embd_size', dest='embedding_size', default=500)
    parser.add_argument('--lstm2', help='configure hidden size of second LSTM layer',
                        type=int, metavar='hidden_size', dest='lstm2_size', default=500)
    parser.add_argument('--cudnn', help='use CuDNN version of LSTM layer instead',
                        action='store_true')

    parser.add_argument('-l', '--log', help='save detailed trainning log (.csv file)',
        metavar='path', dest='log_path')

    parser.add_argument('-m', '--model', help='save/load trainned moldel (.h5 file)',
        metavar='path', dest='model_path')

    parser.add_argument('-s', '--statistics', help='where to store statistics file (.csv)',
        metavar='path', dest='statistics_path', default='statistics.csv')

    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = '{}_{}_{}_{}_{}.csv'.format(args.num_steps, args.batch_size, args.embedding_size, args.lstm_size, timestamp)

    if args.model_path is None:
        args.model_path = '{}_{}_{}_{}_{}.h5'.format(args.num_steps, args.batch_size, args.embedding_size, args.lstm_size, timestamp)

    args.log_path = os.path.abspath(args.log_path)
    args.model_path = os.path.abspath(args.model_path)
    args.statistics_path = os.path.abspath(args.statistics_path)

    return args


def error_if_not_exist(paths):
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if not os.path.exists(path):
            raise IOError(path)


def main(argv):
    options = parse_argv(argv)

    for key in options.__dict__:
        print('{}: {}'.format(key, options.__dict__[key]))


if __name__ == '__main__':
    main(sys.argv)
