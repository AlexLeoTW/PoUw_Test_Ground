import os
import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate


def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='path to statistics.csv')
    parser.add_argument('-s', '--stat', metavar='target', nargs='+',
                        default=['preprocess_time'],
                        help='which columns to target')
    parser.add_argument('-a', '--alpha', metavar='percentage', type=int, default=5,
                        help='how many percentage of the outlier to remove')
    parser.add_argument('-d', '--output', metavar='path',
                        help='where you waant the output be stored')

    args = parser.parse_args()

    return args


def trim_one(array):
    mean = np.mean(array)
    darray = np.abs(array - mean)
    argsort = np.argsort(darray)

    max_idx = argsort[-1]
    print(f'mean = {mean}, trim array[{max_idx}] {array[max_idx]}')

    return np.delete(array, max_idx)


def trim(array, percentage=None, inclusive=True):
    array = array.copy()

    num_trim = (len(array) * percentage)
    num_trim = np.floor(num_trim) if inclusive else np.ceil(num_trim)
    print(f'alpha = {percentage:.0%}, num_trim = {num_trim}')

    for num in range(int(num_trim)):
        array = trim_one(array)

    return array


def main():
    options = parse_argv()
    percentage = options.alpha / 100

    stats = pd.read_csv(options.path)
    result = {}

    for target in options.stat:
        print(target)
        print('-' * 80)
        result[target] = trim(stats[target].to_numpy(), percentage)

    print('=' * 80)
    # print(tabulate(result, headers="keys"))

    table = {'': ['avg', 'max', 'min', 'stdiv']}
    for key in result:
        avg = np.mean(result[key])
        max = np.max(result[key])
        min = np.min(result[key])
        stdiv = np.std(result[key])

        table[key] = [ avg, max, min, stdiv ]
    print(tabulate(table, headers="keys"))

    if options.output:
        with open(options.output, 'w') as file:
            file.write(pd.DataFrame(result).to_csv())


if __name__ == '__main__':
    main()
