import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections.abc import Iterable

import plot_color as c
from perf import get_values_df, iter_gooups, cnt_groups, get_ticklabels


def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='path to statistics.csv')
    parser.add_argument('--params', metavar='col_name', nargs='+', default=None,
                        help='plot the forground of the graph with averaged data')
    parser.add_argument('-t', '--trans', dest='facecolor', action='store_true',
                        help='use transparent background')

    args = parser.parse_args()

    args.facecolor = c.transparent if args.facecolor else c.facecolor

    return args


def _draw_annotate(ax, rects, annotate='{:.2f}'):
    y_min, y_max = ax.get_ylim()
    y_upper = y_max - (y_max - y_min) * 0.1

    for rect in rects:
        height = rect.get_height()
        ax.annotate(annotate.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', annotation_clip=True)

        if height > y_upper:
            y_upper = height + (y_max - y_min) * 0.2
            ax.set_ylim(top=y_upper)


def draw_bars(ax, values_df, groupby, data_row, on=True, label=False, annotate='{:.2f}'):
    colors = c.iter_fg_dot_color() if on else c.iter_fg_dot_off_color()
    gooups_iter = iter_gooups(values_df, groupby, data_row=data_row)
    print(f'\tdata_row = {data_row}')

    # print(values_df)

    for xs, hight, bar_width, _label in gooups_iter:
        rects = ax.bar(x=xs, height=hight, color=next(colors), width=bar_width,
                       label=_label if label else None)
        _draw_annotate(ax, rects, annotate)


def draw_acc_bars(ax, values_df, groupby):
    draw_bars(ax, values_df, groupby, data_row='final_val_acc', on=False)
    draw_bars(ax, values_df, groupby, data_row='val_acc', on=True, label=True)

    ax.set_ylabel('val_acc')
    ax.set_xticks([])


def draw_time_bars(ax, values_df, groupby):
    draw_bars(ax, values_df, groupby, data_row='final_end_time', on=False, annotate='{:.0f}')
    draw_bars(ax, values_df, groupby, data_row='end_time', on=True, label=True, annotate='{:.1f}')

    num_groups, size_group = cnt_groups(values_df, groupby)
    ax.set_ylabel('end_time')
    ax.set_xticks(range(num_groups))
    ax.set_xticklabels(get_ticklabels(values_df, exclude=[groupby]))


def draw_fig(values_df, by_param, facecolor=c.white):
    num_groups, size_group = cnt_groups(values_df, by_param)
    fig = plt.figure(figsize=(num_groups*size_group, 6))

    grid = gridspec.GridSpec(2, 1)
    grid.update(wspace=0.025, hspace=0.01)

    ax_acc = fig.add_subplot(grid[0])
    ax_time = fig.add_subplot(grid[1], sharex=ax_acc)

    draw_acc_bars(ax_acc, values_df, groupby=by_param)
    draw_time_bars(ax_time, values_df, groupby=by_param)

    fig.set_facecolor(facecolor)
    ax_acc.set_facecolor(facecolor)
    ax_time.set_facecolor(facecolor)

    # basic description of the figure
    ax_acc.legend(title=by_param, loc='lower right')
    ax_time.legend(title=by_param, loc='lower right')

    # x-axis / button labels
    if size_group < 3:
        plt.setp(ax_time.get_xticklabels(), rotation=90, ha="center")
        # make space for mega-sized xtick text
        fig.subplots_adjust(bottom=0.3)
    # make space for mega-sized xtick text
    fig.subplots_adjust(bottom=0.2)

    return fig


def _main():
    options = parse_argv()

    params, values_df = get_values_df(options.path, params=options.params)
    base_dir = os.path.dirname(options.path)

    for param in params:
        print(f'param = {param}')

        fig = draw_fig(values_df, by_param=param, facecolor=options.facecolor)

        # save image to the same directory as statistics.csv
        image_path = os.path.join(base_dir, f'pref_{param}.png')
        fig.savefig(image_path)

        # close current figure before drawing again
        plt.close(fig)


if __name__ == '__main__':
    _main()
