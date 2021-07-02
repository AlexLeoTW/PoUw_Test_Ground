import os
import argparse
import numpy as np
import pandas as pd
import yaml

import statistics as stat_tools
import acc_req_descend as acc


def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to source config (.yaml)')

    args = parser.parse_args()
    return args


def _find_best_param_val(avg_hit_df, params):
    hit_time = []
    best_val = []

    for param in params:
        mean = avg_hit_df.groupby(param).mean().reset_index()
        best = mean.sort_values(by='end_time').iloc[0]

        hit_time.append(best['end_time'])
        best_val.append(best[param])

    idxmin = np.argmin(hit_time)

    # print(params[idxmin], best_val[idxmin], hit_time[idxmin])

    return params[idxmin], best_val[idxmin], hit_time[idxmin]


def _ctrl_params(avg_hit_df, params, num=None):
    df_copy = avg_hit_df.copy()

    if num is None:
        num = len(params)

    for itr in range(num):
        best_p, best_v, best_t = _find_best_param_val(df_copy, params)
        df_copy = df_copy[df_copy[best_p] == best_v]

    return best_t


def _get_row(statistics):
    avg_hit_df = stat_tools.find_first_hits_avg(statistics, acc.acc_requirement)
    avg_hit_time = avg_hit_df['end_time'].mean()
    hit_times = [avg_hit_time]

    for num in [1, 2, None]:
        best_t = _ctrl_params(avg_hit_df, statistics.params, num=num)
        hit_times.append(best_t - avg_hit_time)

    return hit_times


def _main():
    options = parse_argv()

    with open(options.path, 'r') as stream:
        src = yaml.safe_load(stream)

    rows = []
    for file in src:
        statistics = stat_tools.Statistics(path=file['path'], params=file['params'])
        rows.append([file['hosts']] + _get_row(statistics))

    result =  pd.DataFrame(data=rows, columns=['host', 'avg', '1', '2', 'All'])
    print(result)


if __name__ == '__main__':
    _main()
