import numpy as np

import statistics as stat_tools
import acc_req_descend as acc
from collections.abc import Iterable


def get_values_df(path, params=None):
    stats = stat_tools.Statistics(path=path, params=params)
    values_df = stat_tools.find_first_hits_avg(stats, acc.acc_requirement)
    params = stats.params

    # collect 'end_time', 'val_acc' of last epoch
    final_end_time = [xs[-1] for xs, ys in stat_tools.gen_log_avg_dots(stats)]
    final_val_acc = [ys[-1] for xs, ys in stat_tools.gen_log_avg_dots(stats)]

    values_df.loc[:, 'final_end_time'] = final_end_time
    values_df.loc[:, 'final_val_acc'] = final_val_acc

    return params, values_df


def iter_gooups(values_df, groupby, data_row):
    values_df = _insert_x_idx(values_df, groupby)
    num_groups, size_group = cnt_groups(values_df, groupby)
    bar_width, x_loc_iter = _get_x_loc_iter(num_groups=num_groups,
                                            size_group=size_group)

    for label in values_df[groupby].unique():
        local_df = values_df[values_df[groupby] == label]

        x_loc = next(x_loc_iter)
        xs = x_loc[local_df['x_idx']]

        hight = local_df[data_row]

        yield xs, hight, bar_width, label


def _insert_x_idx(values_df, groupby):
    params = values_df.columns[:-4]
    params = params.drop(groupby).to_list()

    idx_src = values_df[params].drop_duplicates()
    idx_src.loc[:, 'x_idx'] = np.arange(0, idx_src.shape[0])
    values_df = values_df.merge(idx_src, on=params, how='left')
    col_x_idx = values_df.pop('x_idx')
    values_df.insert(0, 'x_idx', col_x_idx)

    return values_df


def _get_x_loc_iter(num_groups, size_group):
    bar_width = 1 / (size_group + 1)

    x_loc_base = np.arange(num_groups)
    x_loc_base = x_loc_base + 0.5 * bar_width   # align left boarder to 0
    x_loc_base = x_loc_base - (size_group / 2) * bar_width  # center

    def x_loc_iter():
        for nth in range(size_group):
            yield x_loc_base + nth * bar_width

    return bar_width, x_loc_iter()


def cnt_groups(values_df, groupby):
    params = values_df.columns[:-4]
    params = params.drop(groupby).to_list()
    idx_src = values_df[params].drop_duplicates()

    num_groups = idx_src.shape[0]
    size_group = values_df[groupby].nunique()

    return num_groups, size_group


def get_ticklabels(values_df, exclude):
    exclude = list(exclude) + ['x_idx']

    params = values_df.columns[:-4].values
    params = list(filter(lambda param: param not in exclude, params))

    ticks = values_df[params].drop_duplicates()
    ticks = [_series_to_label(row) for index, row in ticks.iterrows()]

    return ticks


def _series_to_label(ser):
    kv_pair = zip(ser.index, ser.values)
    kv_str = map(lambda kv: f'{kv[0]}: {kv[1]}', kv_pair)

    return '\n'.join(kv_str)


# TODO: make this a class
# class Perf():
#
#     def __init__(self, path, params):
#         stats = stat_tools.Statistics(path=path, params=params)
#         values_df = stat_tools.find_first_hits_avg(stats, acc.acc_requirement)
#
#         # collect 'end_time', 'val_acc' of last epoch
#         final_end_time = [xs[-1] for xs, ys in stat_tools.gen_log_avg_dots(stats)]
#         final_val_acc = [ys[-1] for xs, ys in stat_tools.gen_log_avg_dots(stats)]
#
#         values_df.loc[:, 'final_end_time'] = final_end_time
#         values_df.loc[:, 'final_val_acc'] = final_val_acc
#
#         self.params = stats.params
#         self.stats = values_df
#
#     def groupby(by):
#         pass
#
#
# class PrefGroup():
#
#     def __init__(self, local_df, params):
