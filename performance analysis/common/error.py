import sys
import os


def error_and_exit(parser, msg):

    parser.print_usage()

    prog = sys.argv[0]
    print(f'{prog}: error: {msg}', file=sys.stderr)
    sys.exit(1)


def error_if_not_exist(paths):
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if not os.path.exists(path):
            raise IOError(f'{path} not exist')
