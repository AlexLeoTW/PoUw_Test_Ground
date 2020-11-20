import os
import time
import argparse


def parse_argv():
    parser = argparse.ArgumentParser()
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

    parser.add_argument('--feature', help='cut texts after this number of words',
                        type=int, metavar='max_features', dest='max_features', default=20000)
    parser.add_argument('--embd', help='embedding vector size & first LSTM hidden layer size',
                        type=int, metavar='embd_size', dest='embd_size', default=128)
    parser.add_argument('--type', help='what kind of recurrent neurons to use',
        choices=['SimpleRNN', 'GRU', 'LSTM', 'CuDNNGRU', 'CuDNNLSTM'],
        metavar='neuron', default='LSTM')

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
        args.log_path = '{type}_{max_features}_{embd_size}_{timestamp}.csv'.format(
            type=args.type, max_features=args.max_features, embd_size=args.embd_size,
            timestamp=timestamp
        )

    if args.model_path is None:
        args.model_path = '{type}_{max_features}_{embd_size}_{timestamp}.h5'.format(
            type=args.type, max_features=args.max_features, embd_size=args.embd_size,
            timestamp=timestamp
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
