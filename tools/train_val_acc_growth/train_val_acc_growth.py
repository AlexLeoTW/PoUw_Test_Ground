import os
import cmdargv
from functools import reduce
from tabulate import tabulate
from statistics import Statistics, Job
import matplotlib.pyplot as plt
import numpy as np

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
colors_iter = iter(colors)

options = cmdargv.parse_argv()
statistics = Statistics(options.statistics, sort=True, normalize_colname=True)

# if notning specified ==> select all
if options.select is None:
    options.select = {}
else:
    print('\nPrerequisite(s):')
    print(tabulate([options.select.values()], options.select.keys(), tablefmt="github"))

statistics.select_by_values(options.select)


class FindMinMax(Job):
    def __init__(self):
        self.min = np.array([1, 1])
        self.max = np.array([0, 0])

    def exec_every_log(self, params, logs):
        self.min = np.minimum(self.min, logs[['val_acc', 'acc']].min(axis='index').values)
        self.max = np.maximum(self.max, logs[['val_acc', 'acc']].max(axis='index').values)

    def get(self):
        # round to 0.01
        minmax = np.array([self.min.min(), self.max.max()])
        minmax = np.round(minmax * 100) / 100

        return minmax[0], minmax[1]


minmax = FindMinMax()
statistics.exec_every_log(minmax)
min, max = minmax.get()
print(f'min, max = {min}, {max}')


class Draw(Job):
    def exec_every_params_combination(params, logs):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        # 16_2_16_2_64.jpg
        savepath = os.path.join(
            os.path.dirname(options.statistics),
            reduce(lambda x, y: f'{x}_{y}', params.values) + '.jpg'
        )
        # 'conv1_filters: 16 / conv1_kernel_size: 2 / conv2_filters: 16 / pool: 2 / dense: 64'
        title = map(lambda x: f'{x[0]}: {x[1]}', params.items())
        title = reduce(lambda x, y: f'{x} / {y}', title)

        for index, log in zip(range(len(logs)), logs):
            ax.plot(range(log.shape[0]), log['val_acc'], label=f'#{index} val_acc')
            ax.scatter(range(log.shape[0]), log['acc'], label=f'#{index} train acc')

        ax.set_title(title, fontsize=10, y=1.03)
        ax.legend(loc='lower right')
        ax.set(xlabel='Epochs', ylabel='Accuracy', ylim=[min, max])
        plt.savefig(savepath)
        plt.cla()


statistics.exec_every_params_combination(Draw)
