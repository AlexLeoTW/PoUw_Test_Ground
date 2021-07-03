## Warning: this is TF2.0+ Only ###

import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import reduce

import tf_tricks
import statistics as stat_tools
import plot_color as c


def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='path to statistics.csv')
    parser.add_argument('--params', metavar='col_name', nargs='+', default=None,
                        help='plot the forground of the graph with averaged data')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='not showing figure, just save them')
    parser.add_argument('--avg', action='store_true',
                        help='plot averaged startup_time for each model')
    parser.add_argument('-t', '--trans', dest='facecolor', action='store_true',
                        help='use transparent background')
    parser.add_argument('--3d', dest='mode', action='store_const', const='3d',
                        help='plot the figure in "3D"')
    parser.add_argument('--2d', dest='mode', action='store_const', const='2d',
                        help='plot the figure in "2D" (default behavior)')
    parser.add_argument('--drop', metavar='type', nargs='*', type=str, default=[],
                        help='ignore certain type(s) of neuron')

    args = parser.parse_args()

    args.facecolor = c.transparent if args.facecolor else c.facecolor
    args.mode = '2d' if args.mode is None else args.mode

    return args


def _num_trainable_variables(trainable_variables):
    is_model = isinstance(trainable_variables, tf.keras.models.Model)
    is_layer = isinstance(trainable_variables, tf.python.keras.engine.base_layer.Layer)

    if is_model or is_layer:
        trainable_variables = trainable_variables.trainable_variables

    if not type(trainable_variables) == list:
        trainable_variables = [trainable_variables]

    shapes = [variable.shape.as_list() for variable in trainable_variables]
    trainables = map(lambda layer: reduce(lambda x, y: x*y, layer), shapes)

    return sum(trainables)


def num_trainable_variables_categorized(model):
    result = {'Conv': 0, 'RNN': 0, 'Dense': 0, 'Other': 0}

    for layer in model.layers:
        if isinstance(layer, tf.python.keras.layers.convolutional.Conv):
            result['Conv'] += _num_trainable_variables(layer)
        elif isinstance(layer, tf.python.keras.layers.recurrent.RNN):
            result['RNN'] += _num_trainable_variables(layer)
        elif isinstance(layer, tf.python.keras.layers.Dense):
            result['Dense'] += _num_trainable_variables(layer)
        else:
            result['Other'] += _num_trainable_variables(layer)

    return result


def _load_model(dirname, filename):
    model_path = os.path.join(dirname, filename)
    return tf.keras.models.load_model(model_path)


def _split_df(df, by='startup_time_avg', sections=2):
    num_rows = df.shape[0]
    idx_s = np.array_split(np.arange(0, num_rows), sections)
    sorted_df = df.sort_values(by=[by])
    return [sorted_df.iloc[idx] for idx in idx_s]


def _print_df(df):
        # print entire dataframe
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            print(df)


def get_startup_time_df(statistics, avg=True):
    base_dir = statistics.dirname
    values = []

    for index, group, rows in statistics.itergroups():
        # read the first model, then count num_variables
        log_name = rows.iloc[0]['log_path']
        model_name = f'{log_name.rsplit(".", 1)[0]}.h5'

        print(f'reading {model_name} ...')
        model = _load_model(base_dir, model_name)
        num_variables = num_trainable_variables_categorized(model)
        group.update(num_variables)

        if avg:
            startup_time_avg = np.average(rows['startup_time'].values)
            group['startup_time'] = startup_time_avg
            values.append(group)

        else:
            for index, row in rows.iterrows():
                group['startup_time'] = row['startup_time']
                values.append(group)

    startup_time_df = pd.DataFrame(values)

    return startup_time_df


def _drop_all_zero_cols(df):
    zeros = (df == 0).all(axis='index')
    zero_col = zeros[zeros].index.tolist()
    return df.drop(zero_col, axis='columns')


def draw_fig_3d(startup_time_df, facecolor=c.white):
    startup_time_df = _drop_all_zero_cols(startup_time_df)
    dfs = _split_df(startup_time_df, by='startup_time_avg', sections=2)

    x_label = startup_time_df.columns[0]
    y_label = startup_time_df.columns[1]
    z_label = startup_time_df.columns[2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    for df in dfs:
        ax.scatter(df[x_label].values, df[y_label].values, df[z_label].values)

    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    return fig


def draw_scatter(ax, startup_time_df, type_label, time_label='startup_time'):
    dfs = _split_df(startup_time_df, by=time_label, sections=2)

    for df in dfs:
        ax.scatter(df[type_label].values, df[time_label].values,
                   c=c.bg_dot_color, alpha=c.bg_dot_alpha)


def draw_trend(ax, startup_time_df, type_label, time_label='startup_time',
               deg=1):
    xs = startup_time_df[type_label].values
    ys = startup_time_df[time_label].values

    # find polynomial
    trend = np.poly1d(np.polyfit(xs, ys, deg=deg))
    # evenly spaced xs
    alt_xs = np.linspace(min(xs), max(xs), num=50)

    ax.plot(alt_xs, trend(alt_xs), color=c.deep_gray, linestyle='--')


def draw_grouped_avg(ax, startup_time_df, type_label, time_label='startup_time',
                     sections=2):
    mean_df = startup_time_df.groupby(type_label).mean().reset_index()
    dfs = _split_df(mean_df, by=time_label, sections=sections)

    for df, color in zip(dfs, c.iter_fg_dot_color()):
        ax.scatter(df[type_label].values, df[time_label].values,
                   c=color, edgecolor='white', s=90)


def draw_startup_2d(ax, startup_time_df, type_label, time_label='startup_time'):
    draw_scatter(ax, startup_time_df, type_label, time_label)
    # draw_trend(ax, startup_time_df, type_label, time_label)
    draw_grouped_avg(ax, startup_time_df, type_label, time_label)

    ax.set(xlabel=type_label, ylabel=time_label)
    ax.grid(True)


def draw_fig_2d(startup_time_df, facecolor=c.white, drop=[]):
    startup_time_df = _drop_all_zero_cols(startup_time_df)
    startup_time_df = startup_time_df.drop(columns=drop)

    num_types = len(startup_time_df.columns) - 1  # 'startup_time'
    print(f'num_types = {num_types}')
    time_label = startup_time_df.columns[-1]

    fig = plt.figure(figsize=(7 * num_types, 6))
    axs = fig.subplots(nrows=1, ncols=num_types, sharex='col')

    for ax, type_label in zip(axs, startup_time_df.columns[0:-1]):
        draw_startup_2d(ax, startup_time_df, type_label, time_label)

    fig.set_facecolor(facecolor)
    for ax in axs:
        ax.set_facecolor(facecolor)

    return fig


def _main():
    tf_tricks.allow_growth()
    options = parse_argv()

    base_dir = os.path.dirname(options.path)
    statistics = stat_tools.Statistics(path=options.path, params=options.params)
    startup_time_df = get_startup_time_df(statistics, avg=options.avg)
    print(startup_time_df)

    print('writing startup_time.csv ...')
    with open(os.path.join(base_dir, 'startup_time.csv'), 'w') as file:
        file.write(startup_time_df.to_csv())

    startup_time_df = startup_time_df[['Conv', 'RNN', 'Dense', 'Other', 'startup_time']]

    if options.mode == '3d':
        print('draw_fig_3d')
        fig = draw_fig_3d(startup_time_df, facecolor=options.facecolor)
    elif options.mode == '2d':
        print('draw_fig_2d')
        fig = draw_fig_2d(startup_time_df, facecolor=options.facecolor, drop=options.drop)
    else:
        print(f'Unknown mode "{mode}".')

    # showing figure in window
    if not options.quiet:
        plt.show()

    # save image to the same directory as statistics.csv
    image_path = os.path.join(base_dir, 'startup_time.png')
    fig.savefig(image_path)

    # close current figure before drawing again
    plt.close(fig)


if __name__ == '__main__':
    _main()
