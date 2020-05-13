import argparse


def parse_argv():
    parser = argparse.ArgumentParser()

    parser.add_argument('statistics', help='where to store statistics file (.csv)',
        default='statistics.csv')
    parser.add_argument('--inplace', '-i', help='do you want to modify input (.csv) file',
        action='store_true')
    parser.add_argument('--output', '-o', help='output file path. default: stdout',
        default=None, metavar='path')

    args = parser.parse_args()

    if args.inplace:
        args.output = args.statistics

    return args


def _main():
    options = parse_argv()
    for key, value in options.__dict__.items():
        print(f'{key} = {value}')


if __name__ == '__main__':
    _main()

# TODO: add `-r` option
