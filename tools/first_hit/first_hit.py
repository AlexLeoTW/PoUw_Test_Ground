import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

import statistics as stat_tools
import plot_color as c
import acc_req_descend as acc


figsize = [15, 8]
cnt_zorder = 1
important_coords = np.array([[0, 0], [0, 1]])  # coolection of points MUST be in the fig


def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='path to statistics.csv')
    parser.add_argument('--params', metavar='col_name', nargs='+', default=None,
                        help='plot the forground of the graph with averaged data')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='not showing figure, just save them')
    parser.add_argument('-t', '--trans', dest='facecolor', action='store_true',
                        help='use transparent background')
    parser.add_argument('-s', '--focus', action='store_true',
                        help='show only range with first hit')
    parser.add_argument('--dark', action='store_true',
                        help='use dark mode')

    args = parser.parse_args()

    if args.dark:
        c.dark_mode()

    args.facecolor = c.transparent if args.facecolor else c.facecolor

    return args


def _set_zorder(artist):
    global cnt_zorder

    if isinstance(artist, Iterable):
        # assuming ax.plot / matplotlib.lines.Line2D
        artist[0].set_zorder(cnt_zorder)
    else:
        # assuming ax.scatter / matplotlib.collections.PathCollection
        artist.set_zorder(cnt_zorder)

    cnt_zorder = cnt_zorder + 1


def _remove_nan_points(points):
    idx_nan = np.isnan(points).any(axis=1)
    return points[~idx_nan]


def _add_2_important(points):
    global important_coords

    points = np.atleast_2d(points)
    points = _remove_nan_points(points)
    important_coords = np.append(important_coords, points, axis=0)


def _get_border(padding=0.1):
    # print(f'important_coords = {important_coords}')
    max_x = important_coords[:, 0].max()
    min_x = important_coords[:, 0].min()
    padding_x = (max_x - min_x) * padding

    max_y = important_coords[:, 1].max()
    min_y = important_coords[:, 1].min()
    padding_y = (max_y - min_y) * padding

    x_lim = (max(0, min_x - padding_x), max_x + padding_x)
    y_lim = (max(0, min_y - padding_y), max_y + padding_y)

    return x_lim, y_lim


def draw_bg_dots(ax, statistics):
    print('\tdraw_bg_dots')
    bg_xs, bg_ys = stat_tools.get_log_dots(statistics)
    path = ax.scatter(bg_xs, bg_ys, c=c.bg_dot_color, alpha=c.bg_dot_alpha)
    _set_zorder(path)


def draw_avg_dots(ax, statistics, color=None):
    print('\tdraw_avg_dots')
    for xs, ys in stat_tools.gen_log_avg_dots(statistics):
        path = ax.scatter(xs, ys, c=color)
        _set_zorder(path)

    return path


def draw_max_acc_line(ax, statistics, color=None):
    print('\tdraw_max_acc_line')
    for xs, ys in stat_tools.gen_log_avg_dots(statistics, y_mode='avg_max'):
        lines = ax.plot(xs, ys, c=color)
        _set_zorder(lines)

    return lines


def draw_first_hit(ax, statistics, color=None):
    print('\tdraw_first_hit')
    hit_df = stat_tools.find_first_hits_avg(statistics, acc.acc_requirement)
    hit_df = hit_df.dropna(axis='index')
    hit_xs, hit_ys = hit_df['end_time'], hit_df['val_acc']

    path = ax.scatter(hit_xs, hit_ys, c=color, edgecolor='white', s=72)

    # in case no hit was found
    if not hit_df.empty:
        _add_2_important((max(hit_xs), max(hit_ys)))
    else:
        _add_2_important((ax.get_xlim()[1], ax.get_ylim()[1]))

    _set_zorder(path)

    return path


def _set_label(artist, label):
    if isinstance(artist, Iterable):
        # assuming ax.plot / matplotlib.lines.Line2D
        artist[0].set_label(label)
    else:
        # assuming ax.scatter / matplotlib.collections.PathCollection
        artist.set_label(label)


def draw_group(fn, ax, statistics, param, on=True, label=False, **kwargs):
    param_vals = statistics.params_combination[param].unique()
    colors = c.iter_fg_dot_color() if on else c.iter_fg_dot_off_color()

    for param_val in param_vals:
        local_stat = statistics.select_by_values({param: param_val})
        color = next(colors)

        obj = fn(ax, local_stat, color=color, **kwargs)

        if label:
            _set_label(obj, label=param_val)


def draw_acc_req_line(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xs, ys = acc.get_acc_req_xy(x_range=(ax.get_xlim()), y_range=(0, 1))

    lines = ax.plot(xs, ys, **c.acc_req_line)

    _set_zorder(lines)


def draw_fig(statistics, param, facecolor=c.white, focus=False):
    fig, ax = plt.subplots(figsize=figsize)

    # draw the gray dots (no avg) in the background
    draw_bg_dots(ax, statistics)

    draw_group(draw_avg_dots, ax, statistics, param, on=False)
    draw_group(draw_max_acc_line, ax, statistics, param, on=False)

    draw_acc_req_line(ax)

    draw_group(draw_first_hit, ax, statistics, param, on=True, label=True)

    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    ax.set(xlabel="end_time(s)", ylabel="val_acc(%)")
    ax.legend(title=param, loc='upper right')

    fig.tight_layout()

    if focus:
        x_lim, y_lim = _get_border()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

    return fig


def main():
    options = parse_argv()

    statistics = stat_tools.Statistics(path=options.path, params=options.params)
    base_dir = os.path.dirname(options.path)

    for param in statistics.params:
        print(f'param = {param}')

        fig = draw_fig(statistics, param=param, facecolor=options.facecolor, focus=options.focus)

        # showing figure in window
        if not options.quiet:
            plt.show()

        # save image to the same directory as statistics.csv
        image_path = os.path.join(base_dir, f'first_hit_{param}.png')
        fig.savefig(image_path)

        # close current figure before drawing again
        plt.close(fig)


if __name__ == '__main__':
    main()
