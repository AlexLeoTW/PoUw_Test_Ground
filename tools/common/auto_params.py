import argparse
import os
import re

col_name_regex_map = {
    'statistics': {
        'acc_score': '^acc_score'
    },
    'epochs': {
        'val_loss': '^val_loss',
        'val_acc': '^val_.*acc.*',
        'loss': '^loss',
        'acc': '^(?!val).*acc'
    }
}


# return loc of the first element matched
def _match_in_list(regex, lis):
    loc = map(lambda name: re.search(regex, name) is not None, lis)
    return list(loc).index(True)


def auto_params(col_names):
    regex = col_name_regex_map['statistics']['acc_score']
    return col_names[:_match_in_list(regex, col_names)]


def auto_params_from_file(path):
    with open(path, 'r') as file:
        colnames = file.readline()
        colnames = colnames.split(',')
        return auto_params(colnames)


def _first_log_path_from_statistics(path):
    with open(path, 'r') as file:
        file.readline()  # skip first line
        log_path = file.readline().strip().split(',')[-1]

        # join log_path column with path inputed
        dir = os.path.dirname(path)
        log_file = os.path.basename(log_path)
        log_path = os.path.join(dir, log_file)

        return log_path


def auto_acc_loss(col_names):
    acc_loss = col_name_regex_map['epochs'].copy()

    for key in acc_loss:
        regex = acc_loss[key]
        acc_loss[key] = col_names[_match_in_list(regex, col_names)]

    return acc_loss


def auto_acc_loss_from_file(path, is_statistics=True):
    if is_statistics:
        path = _first_log_path_from_statistics(path)

    with open(path, 'r') as log:
        colnames = log.readline().split(',')
        return auto_acc_loss(colnames)


def parse_argv(auto_params=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='where statistics.csv is stored')
    parser.add_argument('-params', help='which columns are params(variables) in statistics.csv',
        nargs='+', type=str, metavar='param', default=None)
    parser.add_argument('--acc', help='calumn name for accuracy in train log',
        metavar='col_name', default=None)

    args = parser.parse_args()

    if auto_params and not args.params:
        args.params = auto_params_from_file(args.path)

    if auto_params and not args.acc:
        args.acc = auto_acc_loss_from_file(args.path)['val_acc']

    return args


def _main():
    options = parse_argv(auto_params=True)
    for key in options.__dict__:
        print('{}: {}'.format(key, options.__dict__[key]))


if __name__ == '__main__':
    _main()
