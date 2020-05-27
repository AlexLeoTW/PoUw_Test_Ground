import argparse
import functools


class _Select(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):

        if len(values) == 0:
            return

        # split: 'a=1' form
        values = [value.split('=') for value in values]
        values = functools.reduce(lambda x, y: x+y, values)

        # make values a dict
        iter_values = iter(values)
        select = dict(zip(iter_values, iter_values))
        setattr(namespace, self.dest, select)


def parse_argv():
    parser = argparse.ArgumentParser()

    parser.add_argument('statistics', metavar='path', default='statistics.csv',
        help='where statistics file (.csv) stored')
    parser.add_argument('-s', '-select', dest='select', nargs='+', action=_Select,
        help='prerequisites for graphs to draw. ex. a=1 b=2')

    args = parser.parse_args()

    return args


def _main():
    options = parse_argv()
    for key, value in options.__dict__.items():
        print(f'{key} = {value}')


if __name__ == '__main__':
    _main()

# TODO: make print() using f-strings
