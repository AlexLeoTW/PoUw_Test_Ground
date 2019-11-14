import argparse
import os
import re


def _match_in_list(regex, lis):
    loc = map(lambda name: re.search(regex, name) is not None, lis)
    return list(loc).index(True)


def parse_argv():
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='where statistics.csv is stored')
    parser.add_argument('-params', help='which columns are params(variables) in statistics.csv',
        nargs='+', type=str, metavar='param', default=None)
    parser.add_argument('--acc', help='calumn name for accuracy in train log',
        metavar='col_name', default=None)

    args = parser.parse_args()

    return args


def auto_params():
    args = parse_argv()

    if not args.params:
        with open(args.path, 'r') as file:
            colnames = file.readline()
            colnames = colnames.split(',')
            args.params = colnames[:_match_in_list('^acc_score', colnames)]

    if not args.acc:
        with open(args.path, 'r') as file:
            _void = file.readline()
            log_path = file.readline().strip().split(',')[-1]

            dir = os.path.dirname(args.path)
            log_file = os.path.basename(log_path)
            log_path = os.path.join(dir, log_file)

            with open(log_path, 'r') as log:
                colnames = log.readline().split(',')
                args.acc = colnames[_match_in_list('^val_.*acc', colnames)]

    return args


def _main():
    options = auto_params()
    print(options)


if __name__ == '__main__':
    _main()
