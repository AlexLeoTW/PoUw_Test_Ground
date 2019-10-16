import os
import sys
import pandas as pd


class Statistics(object):

    def __init__(self, path, params):
        # self.base_path = os.path.basename(path)
        self.params = params
        self.statistics = pd.read_csv(path)
        self.params_combination = self.statistics[self.params].drop_duplicates()
        self.temp = None

    def select_by_values(self, criteria):
        selection = pd.Series([True] * self.statistics.shape[0])
        for key, value in criteria.items():
            selection = selection & (self.statistics[key] == value)

        return self.statistics[selection]

    def exec_every_result(self, job):
        # loop through non-repeating [params] combinations
        for index, row in self.params_combination.iterrows():
            selected_rows = self.select_by_values(row.to_dict())

            # loop through all test results matching [params] combinations
            for index, row in selected_rows.iterrows():
                log = pd.read_csv(row['log_path'])
                job(row[self.params].to_dict(), log)  # job(param, avg)

    def exec_every_avg(self, job):
        # loop through non-repeating [params] combinations
        for index, row in self.params_combination.iterrows():
            selected_rows = self.select_by_values(row.to_dict())
            log_sum = None

            # loop through all test results matching [params] combinations
            for index, row in selected_rows.iterrows():
                log = pd.read_csv(row['log_path'])

                if log_sum is not None:
                    log_sum = log_sum + log
                else:
                    log_sum = log

            log_avg = log_sum / selected_rows.shape[0]
            job(row[self.params].to_dict(), log_avg)  # job(param, avg)

    def deep_collect(self, cols, avg=False):
        self.temp = pd.DataFrame([], columns=(self.params + cols))

        def collect(params, per_test_log):
            for key, value in params.items():
                per_test_log[key] = [value] * per_test_log.shape[0]

            per_test_log = per_test_log[self.temp.columns]

            self.temp = self.temp.append(per_test_log)

        if avg:
            self.exec_every_avg(collect)
        else:
            self.exec_every_result(collect)

        self.temp = self.temp.reset_index(drop=True)

        collected = self.temp
        self.temp = None

        return collected


def main():
    # path = 'statistics.csv'
    # params = ['conv1_filters', 'conv1_kernel_size', 'conv2_filters', 'pool', 'dense']
    path = sys.argv[1]
    params = sys.argv[2:]
    print('path = {}'.format(path))
    print('params = {}'.format(params))
    statistics = Statistics(path, params)

    # fix log_path
    statistics.statistics['log_path'] = statistics.statistics['log_path'].apply(lambda log_path: os.path.join(os.path.dirname(path), os.path.basename(log_path)))

    collected = statistics.deep_collect(['val_acc', 'end_time'])
    print(collected[collected['val_acc'] > 0.98].sort_values(by=['end_time']))


if __name__ == '__main__':
    main()
