import os
import numpy as np
import pandas as pd
import copy
from scipy.optimize import fsolve

from error import perror
from auto_params import auto_params, auto_acc_loss

# ==============================================================================
# ||                   class:Statistics & related functions                   ||
# ==============================================================================

def _test_criteria(statistics, criteria):
    # if non-existed key/column specified ==> raise error
    if not set(criteria.keys()).issubset(set(statistics.columns)):
        # selected keys - presented/selectable keys = non-existed
        missing_keys = set(criteria.keys()) - set(statistics.columns)

        perror(f'key: {missing_keys} doesn\'t exist')
        raise KeyError('some columns not found')


def _log_auto_rename(df):
    # get map: general_name --> specific name
    new_names = auto_acc_loss(df.columns)
    # reverse map: specific name --> general_name
    new_names = dict(zip(new_names.values(), new_names.keys()))
    return df.rename(columns=new_names)


def _group_index(df, by):
    groups = df[by]
    groups.insert(len(by), column='count', value=[1] * groups.shape[0])
    groups = groups.groupby(by).count()
    groups = groups.reset_index()

    end_index = np.add.accumulate(groups['count'].values)
    start_index = end_index - groups['count'].values
    end_index = end_index - 1   # count vs. index

    return start_index, end_index


def _df_select_by_values(df, criteria):
    selection = pd.Series([True] * df.shape[0])

    for key, value in criteria.items():
        selection = selection & (df[key] == value)

    return df[selection]


class Statistics():
    # terms.
    # - statistics: statistics.csv
    # - log: train log / XX_XX_XX_20YY-MM-DD_HH:MM:SS.csv

    def __init__(self, path, params=None, normalize_colname=True):
        self.dirname = os.path.dirname(path)
        self.statistics = pd.read_csv(path)
        self.params = params if params else auto_params(self.statistics.columns.to_list())
        self.params_combination = None  # will update later
        self.normalize_colname = normalize_colname

        # sort
        self.statistics = self.statistics.sort_values(by=self.params)
        self.statistics = self.statistics.reset_index(drop=True)

        self._update_params_combination()

    def select_by_values(self, criteria, inplace=False):
        if inplace:
            stat_obj = self
        else:
            stat_obj = copy.deepcopy(self)

        stat_obj.statistics = _df_select_by_values(stat_obj.statistics, criteria)
        stat_obj.statistics.reset_index(drop=True, inplace=True)
        stat_obj._update_params_combination()

        return stat_obj

    def logs_gen(self):
        # iter through available training logs,
        #   and yield a log for each one of them
        for index, row in self.statistics.iterrows():
            log = self.read_log(row['log_path'])
            group = row[self.params].to_dict()

            yield group, log

    def grouped_logs_gen(self):
        # iter through available training logs,
        #   and yield a log for each group (by params)
        idx_group_start, idx_group_end = _group_index(self.statistics, by=self.params)

        for start, end in zip(idx_group_start, idx_group_end):
            logs = []

            for index, row in self.statistics.loc[start:end].iterrows():
                log = self.read_log(row['log_path'])
                logs.append(log)

            group = self.statistics.iloc[start][self.params].to_dict()

            yield group, logs

    def read_log(self, filename):
        log = pd.read_csv(os.path.join(self.dirname, filename))
        # print(f'reading "{filename}"')

        if self.normalize_colname:
            return _log_auto_rename(log)
        else:
            return log

    def _update_params_combination(self):
        params_combination = self.statistics[self.params].drop_duplicates()
        self.params_combination = params_combination


# ==============================================================================
# ||                  functions for plotting statistics_obj                   ||
# ==============================================================================

# used by: acc_growth, first_hit
def get_log_dots(stat_obj, x='end_time', y='val_acc'):
    xs = np.array([])
    ys = np.array([])

    for group, log in stat_obj.logs_gen():
        xs = np.append(xs, log[x])
        ys = np.append(ys, log[y])

    return xs, ys


# used by: N/A
def gen_log_dots(stat_obj, x='end_time', y='val_acc'):

    for group, log in stat_obj.logs_gen():
        xs = log[x]
        ys = log[y]

        return xs, ys


# used by: acc_growth
def get_log_avg_dots(stat_obj, x='end_time', x_mode='offset',
                               y='val_acc', y_mode='avg'):
    xs = np.array([])
    ys = np.array([])

    for group, logs in stat_obj.grouped_logs_gen():
        avg_x = _col_avg(logs, col_name=x, mode=x_mode)
        avg_y = _col_avg(logs, col_name=y, mode=y_mode)

        xs = np.append(xs, avg_x)
        ys = np.append(ys, avg_y)

    return xs, ys


# used by: first_hit
def gen_log_avg_dots(stat_obj, x='end_time', x_mode='offset',
                               y='val_acc', y_mode='avg'):

    for group, logs in stat_obj.grouped_logs_gen():
        avg_x = _col_avg(logs, col_name=x, mode=x_mode)
        avg_y = _col_avg(logs, col_name=y, mode=y_mode)

        yield avg_x, avg_y


def _col_avg(logs, col_name, mode='offset'):
    # get the (math)aveaage value of a specific col_name
    #   avg: average over values at each index position
    #   offset: aveaage over epoch time (time[t] - time[t-1]) and add them back up
    #   avg_max: running max of "avg"
    cols = [np.array(log[col_name]) for log in logs]

    if mode=='offset':
        cols_from_zero = [np.insert(col, 0, 0) for col in cols]
        cols_diff = [np.diff(col) for col in cols_from_zero]

        avg_diff = np.sum(cols_diff, axis=0) / len(cols_diff)
        cols_avg = np.add.accumulate(avg_diff)

        return cols_avg

    elif mode=='avg':
        cols_avg = np.sum(cols, axis=0) / len(cols)
        return cols_avg

    elif mode=='avg_max':
        cols_avg = np.sum(cols, axis=0) / len(cols)
        max_avg = np.maximum.accumulate(cols_avg)
        return max_avg

    else:
        raise ValueError('average mode not supported')


# ==============================================================================
# ||            functions for plotting first_hits (dep injection)             ||
# ==============================================================================


def find_first_hits(stat_obj, acc_req, reverse=None,
                    time='end_time', acc='val_acc'):

    rows = []

    for group, logs in stat_obj.logs_gen():
        hit_x, hit_y = _find_first_hit(logs[time].values, logs[acc].values,
                                       acc_req, reverse)

        # note: values can be a str() or int()
        new_row = list(group.values()) + [hit_x, hit_y]
        rows.append(new_row)

    # column name = [param1, param2, .... , time, acc]
    cal_names = np.append(stat_obj.params, [time, acc])

    df = pd.DataFrame(rows, columns=cal_names)

    return df


def find_first_hits_avg(stat_obj, acc_req, reverse=None,
                        time='end_time', acc='val_acc'):

    rows = []

    for group, logs in stat_obj.grouped_logs_gen():

        avg_time = _col_avg(logs, col_name=time, mode='offset')
        avg_acc = _col_avg(logs, col_name=acc, mode='avg_max')

        hit_x, hit_y = _find_first_hit(avg_time, avg_acc, acc_req, reverse)

        # note: values can be a str() or int()
        new_row = list(group.values()) + [hit_x, hit_y]
        rows.append(new_row)


    # column name = [param1, param2, .... , time, acc]
    cal_names = np.append(stat_obj.params, [time, acc])

    df = pd.DataFrame(rows, columns=cal_names)

    return df


def _find_first_hit(times, accs, acc_req, reverse=None):
    if reverse is None:
        reverse = _build_reverse_acc_req(acc_req)

    # always consider best 'acc' so far
    accs = np.maximum.accumulate(accs)

    # just in case a set of training can't passs acc_req() entire time
    max_acc = accs[-1]
    times = np.append(times, reverse(max_acc))
    accs = np.append(accs, max_acc)

    passed = accs > acc_req(times)
    idx_first_passed = np.argmax(passed)

    return times[idx_first_passed], accs[idx_first_passed]


def _build_reverse_acc_req(acc_req):
    def reverse(acc):
        # acc_req(time) == acc  ==>  acc_req(time) - acc == 0
        acc_req_offset = (lambda time: acc_req(time) - acc)

        return fsolve(acc_req_offset, 1e-3)[0]

    return reverse
