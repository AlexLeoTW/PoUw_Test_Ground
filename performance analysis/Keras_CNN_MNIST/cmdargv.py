import argparse, sys, os

def parse_argv(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-conv1', help='configure first Conv2D layer ex. 32 2',
        nargs='+', type=int, metavar=('filters', 'kernel_size'), default=[32, 2])
    parser.add_argument('-conv2', help='configure second Conv2D layer ex. 64',
        type=int, metavar='filters', default=64)
    parser.add_argument('-pool', help='configure MaxPooling2D pool_size ex. 2',
        type=int, metavar='pool_size', default=2)
    parser.add_argument('-dense', help='configure units of dense layer ex. 128',
        type=int, metavar='units', default=128)
    parser.add_argument('-s', '--statistics', help='where to store statistics file (.csv)',
        default='statistics.csv')

    args = parser.parse_args()

    return args

def main(argv):
    options = parse_argv(argv)
    print(options)

if __name__ == '__main__':
    main(sys.argv)
