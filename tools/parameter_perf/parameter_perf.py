import numpy as np
from auto_params import parse_argv
from statistics import avg_statistics, select_by_values
from collections import Iterable
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

targets = ['acc_score(%)', 'train_time']


def _main():
    options = parse_argv()
    avg = avg_statistics(options.path, options.params)

    for param in options.params:
        print(f'param = {param}')

        p_values = avg[param].drop_duplicates().to_list()
        labels = [str(value) for value in p_values]
        print(f'labels = {labels}')

        values = _collect_values(avg, param, p_values, targets)

        fig = _draw_plot(labels, values, annotate=True, scale=True)
        fig.tight_layout()
        fig.savefig(
            os.path.join(os.path.dirname(options.path), f'pref_{param}.jpg'))
        plt.cla()


def _draw_plot_vertical(labels, values, annotate=False, scale=False):
    assert len(labels) == len(values)

    values = np.array(values)

    len_group = values.shape[1]
    num_categories = len(labels)
    bar_width = 1.0 / (num_categories + 1)

    fig, ax = plt.subplots(figsize=(2 * len_group, 5))
    x_base = np.arange(len_group)
    bottom = values.min() - (values.max() - values.min()) * 0.1
    bottom = np.floor(bottom * 1000.0) / 1000.0

    for index, value, label in zip(range(num_categories), values, labels):
        x_start = bar_width * index + bar_width
        xs = x_base + x_start

        if scale:
            rects = ax.bar(xs, value-bottom, width=bar_width, label=label)
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda x, pos: x + bottom))
        else:
            rects = ax.bar(xs, value, width=bar_width, label=label)

        if annotate:
            autolabel(rects, ax)

    ax.set_xticks([])   # hide x_ticks
    ax.legend()

    return fig


def _collect_values(df, param, p_values, targets):
    values = {
        # 'acc_score(%)': [[], [], [], ...],
        # 'train_time': [[], [], [], ...],
        # ...
    }

    for p_value in p_values:
        this_value = select_by_values(df, {param: p_value})
        _append_values(values, this_value, targets)

    return values


def _append_values(values, df, targets):
    for target in targets:
        target_value = df[target].to_list()

        if target in values:
            values[target].append(target_value)
        else:
            values[target] = [target_value]


def _draw_plot(labels, values, annotate=False, scale=False):
    max_len_group = max([len(values[x][0]) for x in values])
    len_target = len(values.keys())
    print(f'max_len_group = {max_len_group}')
    print(f'figsize = {(2 * max_len_group, 5 * len_target)}')
    fig, axs = plt.subplots(len(values), 1,
                            figsize=(2 * max_len_group, 5 * len_target))
    fig.tight_layout()

    for ax, target in zip(axs, values):
        _draw_subplot(ax, labels, values[target], annotate, scale)

    return fig


def _draw_subplot(ax, labels, values, annotate=False, scale=False):
    assert len(labels) == len(values)

    values = np.array(values)

    len_group = values.shape[1]
    num_categories = len(labels)
    bar_width = 1.0 / (num_categories + 1)
    x_base = np.arange(len_group)

    bottom = values.min() - (values.max() - values.min()) * 0.1
    bottom = np.floor(bottom * 1000.0) / 1000.0

    for index, value, label in zip(range(num_categories), values, labels):
        x_start = bar_width * index + bar_width
        xs = x_base + x_start

        if scale:
            rects = ax.bar(xs, value-bottom, width=bar_width, label=label)
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda x, pos: np.around(x + bottom, decimals=3)))
        else:
            rects = ax.bar(xs, value, width=bar_width, label=label)

        if annotate:
            _draw_annotate(rects, ax, value)

    ax.set_xticks([])   # hide x_ticks
    ax.legend()


def _draw_annotate(rects, ax, value):
    for rect, val in zip(rects, value):
        height = rect.get_height()
        ax.annotate(f'{val:.3f}',  # f'{val:.3%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', annotation_clip=True)


def iterable(obj):
    return isinstance(obj, Iterable)


if __name__ == '__main__':
    _main()
