import os
import pandas as pd
import numpy as np
import abc
from auto_params import auto_params, auto_acc_loss


def _test_criteria(statistics, criteria, fix_criteria=False):
    # if non-existed key/column specified ==> raise error
    if not set(criteria.keys()).issubset(set(statistics.columns)):
        # selected keys - presented/selectable keys = non-existed
        missing_keys = set(criteria.keys()) - set(statistics.columns)
        raise KeyError(f'key: {missing_keys} doesn\'t exist')

    if fix_criteria:
        # make selected prerequisites value fit statistics.csv
        for key, value in criteria.items():
            try:
                criteria[key] = statistics[key].dtypes.type(value)
            except ValueError:
                raise ValueError(f'"{key}" should be "{statistics[key].dtypes.name}" type')


def select_by_values(df, criteria):
    selection = pd.Series([True] * df.shape[0])

    for key, value in criteria.items():
        selection = selection & (df[key] == value)

    return df[selection]


def avg_statistics(path, params):
    df = pd.read_csv(path)
    df = df.groupby(params).mean().reset_index()
    return df


def _group_by_params(df, params, sorted=False):
    local_df = df if sorted else df.sort_values(by=params)

    params_combination = local_df[params]
    params_combination.insert(len(params), column='count', value=[1] * params_combination.shape[0])
    params_combination = params_combination.groupby(params).count()
    params_combination = params_combination.reset_index()

    end_index = np.add.accumulate(params_combination['count'].values)
    start_index = end_index - params_combination['count'].values
    end_index = end_index - 1   # count vs. index

    return params_combination[params], start_index, end_index


def _log_auto_rename(df):
    # get map: general_name --> specific name
    new_names = auto_acc_loss(df.columns)
    # reverse map: specific name --> general_name
    new_names = dict(zip(new_names.values(), new_names.keys()))
    return df.rename(columns=new_names)


class Job(abc.ABC):
    @classmethod
    def exec_every_log(params: pd.Series, log: pd.DataFrame) -> None:
        raise NotImplementedError

    @classmethod
    def exec_every_avg(params: pd.Series, log: pd.DataFrame) -> None:
        raise NotImplementedError

    @classmethod
    def exec_every_params_combination(params: pd.Series, logs: list) -> None:
        raise NotImplementedError


class Statistics():
    def __init__(self, path, params=None, sort=False, normalize_colname=False):
        self.dirname = os.path.dirname(path)
        self.statistics = pd.read_csv(path)
        self.params = params if params else auto_params(self.statistics.columns.to_list())
        self.params_combination = self.statistics[self.params].drop_duplicates()
        self.sort = sort
        self.normalize_colname = normalize_colname

        if sort:
            self.statistics = self.statistics.sort_values(by=self.params)
            self.statistics = self.statistics.reset_index(drop=True)

    def select_by_values(self, criteria):
        criteria_tmp = criteria.copy()
        _test_criteria(self.statistics, criteria_tmp, fix_criteria=True)

        self.statistics = select_by_values(self.statistics, criteria)
        self.params_combination = self.statistics[self.params].drop_duplicates()

    def exec_every_log(self, job):
        for index, row in self.statistics.iterrows():
            log = self.read_log(row['log_path'])
            job.exec_every_log(row[self.params], log)  # do(params: dict, log: pd.DataFrame)

    def exec_every_avg(self, job):
        if self.sort:
            sorted_stat = self.statistics
            params_combination, start_index, end_index = _group_by_params(self.statistics, self.params, sorted=True)
        else:
            sorted_stat = self.get_sorted()
            params_combination, start_index, end_index = _group_by_params(self.statistics, self.params, sorted=False)

        for start, end in zip(start_index, end_index):
            log_sum = None

            for index, row in sorted_stat.loc[start:end].iterrows():
                log = self.read_log(row['log_path'])
                log_sum = log_sum + log if log_sum is not None else log

            log_avg = log_sum / (end - start + 1)
            job.exec_every_avg(row[self.params], log_avg)  # do(params: dict, log: pd.DataFrame)

    def exec_every_params_combination(self, job):
        if self.sort:
            sorted_stat = self.statistics
            params_combination, start_index, end_index = _group_by_params(self.statistics, self.params, sorted=True)
        else:
            sorted_stat = self.get_sorted()
            params_combination, start_index, end_index = _group_by_params(self.statistics, self.params, sorted=False)

        for start, end in zip(start_index, end_index):
            logs = []

            for index, row in sorted_stat.loc[start:end].iterrows():
                log = self.read_log(row['log_path'])
                logs.append(log)

            job.exec_every_params_combination(sorted_stat.loc[start, self.params], logs)

    def deep_collect(self, cols, avg=False):
        self.temp = pd.DataFrame([], columns=(self.params + cols))

        class Collect(Job):
            def exec_every_log(params, log):
                # expand params(Series) to DataFrame
                params = pd.DataFrame([params] * log.shape[0], index=range(log.shape[0]))
                new_rows = params.merge(log, left_index=True, right_index=True, validate='one_to_one')
                new_rows = new_rows[self.temp.columns]

                self.temp = self.temp.append(new_rows, ignore_index=True)

            def exec_every_avg(params, log):
                Collect.exec_every_log(params, log)

        if avg:
            self.exec_every_avg(Collect)
        else:
            self.exec_every_log(Collect)

        self.temp.reset_index(drop=True, inplace=True)

        collected = self.temp
        del self.temp

        return collected

    def read_log(self, filename):
        log = pd.read_csv(os.path.join(self.dirname, filename))
        log = _log_auto_rename(log) if self.normalize_colname else log

        return log

    def get_sorted(self):
        if self.sort:
            return self.statistics
        else:
            sorted_stat = self.statistics.sort_values(by=self.params)
            sorted_stat = sorted_stat.reset_index(drop=True)
            return sorted_stat
