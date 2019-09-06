import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import functools
from statistics import Statistics

figsize = [15, 8]
# statistics_path = 'statistics.csv'
# params = ['conv1_filters', 'conv1_kernel_size', 'conv2_filters', 'pool', 'dense']
statistics_path = sys.argv[1]
params = sys.argv[2:]

statistics = Statistics(statistics_path, params)

# fix log_path
statistics.statistics['log_path'] = statistics.statistics['log_path'].apply(lambda log_path: os.path.join(os.path.dirname(statistics_path), os.path.basename(log_path)))


def draw_per_avg():
    collected = statistics.deep_collect(['val_acc', 'end_time'])
    graph_end_time = collected.sort_values(by='end_time', ascending=False).iloc[0]['end_time']
    plt.figure(figsize=figsize)

    for index, row in statistics.params_combination.iterrows():
        figname = '_'.join(map(str, row.values))
        # scatter on the background
        plt.scatter(collected['end_time'].tolist(), collected['val_acc'].tolist(), c='#808080', alpha=0.5)

        # plt.axhline(y=0.98)

        # step: acc growth
        results = statistics.select_by_values(row.to_dict())        # bunch of path
        results = [pd.read_csv(x) for x in results['log_path']]     # bunch of DataFrame
        results = [x[['val_acc', 'end_time']] for x in results]     # select only ['val_acc', 'end_time']
        avg = functools.reduce(lambda x, y: x + y, results) / len(results)  # divide by
        avg = avg.append(avg.iloc[avg.shape[0]-1])                  # duplicate last record
        avg.iloc[avg.shape[0]-1]['end_time'] = graph_end_time       # change end_time of last record
        plt.step(avg['end_time'].tolist(), avg['val_acc'].tolist(), where='post', label=figname)

        plt.xlabel("end_time(s)")
        plt.ylabel("val_acc(%)")
        plt.legend(title='_'.join(params))

        # plt.show()
        plt.savefig(os.path.join(os.path.dirname(statistics_path), '{}.jpg'.format(figname)))
        plt.cla()


def draw_per_param_avg():
    plt.figure(figsize=figsize)
    collected = statistics.deep_collect(['val_acc', 'end_time'])
    collected_avg = statistics.deep_collect(['val_acc', 'end_time'], avg=True)

    for param in params:

        plt.scatter(collected['end_time'].tolist(), collected['val_acc'].tolist(), c='#808080', alpha=0.5)

        for conv1_filters in collected_avg[param].drop_duplicates().values:

            selected = collected_avg[collected_avg[param] == conv1_filters]
            plt.scatter(selected['end_time'], selected['val_acc'], label=str(conv1_filters))

        plt.xlabel("end_time(s)")
        plt.ylabel("val_acc(%)")
        plt.legend(title=param)
        plt.savefig(os.path.join(os.path.dirname(statistics_path), 'param_{}.jpg'.format(param)))
        print('param = {}'.format(param))
        plt.cla()


draw_per_param_avg()
