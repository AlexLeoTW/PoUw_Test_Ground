import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections.abc import Iterable

import statistics as stat_tools
import plot_color as c
import acc_req_descend as acc


figsize = [15, 8]
# figsize = None
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

    args = parser.parse_args()

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

    if not hit_df.empty:
        _add_2_important((max(hit_xs), max(hit_ys)))
    else:
        _add_2_important((ax.get_xlim()[1], ax.get_ylim()[1]))

    _set_zorder(path)

    return path


def draw_group(fn, ax, stat_objs, on=True, label=None, **kwargs):
    colors = c.iter_fg_dot_color() if on else c.iter_fg_dot_off_color()
    iter_label = iter(label) if label is not None else None

    for stat_obj in stat_objs:
        color = next(colors)

        obj = fn(ax, stat_obj, color=color, **kwargs)

        if iter_label:
            _set_label(obj, label=next(iter_label))


def _set_label(artist, label):
    if isinstance(artist, Iterable):
        # assuming ax.plot / matplotlib.lines.Line2D
        artist[0].set_label(label)
    else:
        # assuming ax.scatter / matplotlib.collections.PathCollection
        artist.set_label(label)


def draw_acc_req_line(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xs, ys = acc.get_acc_req_xy(x_range=(ax.get_xlim()), y_range=(0, 1))

    plot_arg = {'c': c.deep_gray, 'linestyle': '--', 'linewidth': '2'}
    lines = ax.plot(xs, ys, **plot_arg)

    _set_zorder(lines)


def draw_acc_growth(ax, statistics, by_param):
    param_vals = statistics.params_combination[by_param].unique()
    stat_objs = [statistics.select_by_values({by_param: param_val}) for param_val in param_vals]

    draw_bg_dots(ax, statistics)
    draw_group(draw_avg_dots, ax, stat_objs, on=False)
    draw_group(draw_max_acc_line, ax, stat_objs, on=False)
    draw_group(draw_first_hit, ax, stat_objs, label=param_vals, on=True)
    draw_acc_req_line(ax)

    ax.legend(title=by_param, loc='upper left')
    ax.set_ylabel('val_acc(%)')


def draw_box_plot(ax, statistics, by_param):
    df = stat_tools.find_first_hits(statistics, acc.acc_requirement)
    df = df.dropna(axis='index')
    colors = c.iter_fg_dot_color()

    data = []
    param_vals = []
    for param_val, local_df in df.groupby(by_param).__iter__():
        data.append(local_df['end_time'])
        param_vals.append(param_val)

    box_plot = ax.boxplot(data, vert=False, labels=param_vals,
                          notch=False, patch_artist=True)

    for box, flier, color in zip(box_plot['boxes'], box_plot['fliers'], colors):
        box.set_facecolor(color)
        flier.set_markerfacecolor(color)

    ax.set(xlabel='end_time(s)', ylabel=by_param)


def draw_fig(statistics, by_param, facecolor=c.white):
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    grid = gridspec.GridSpec(2, 1, figure=fig)
    grid.update(wspace=0.025, hspace=0.01)

    ax_acc = fig.add_subplot(grid[0])
    ax_bar = fig.add_subplot(grid[1], sharex=ax_acc)

    # hide x-ticks
    ax_acc.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    draw_acc_growth(ax_acc, statistics, by_param=by_param)
    draw_box_plot(ax_bar, statistics, by_param=by_param)

    fig.set_facecolor(facecolor)
    ax_acc.set_facecolor(facecolor)
    ax_bar.set_facecolor(facecolor)

    x_lim, y_lim = _get_border()
    ax_acc.set_xlim(*x_lim)
    ax_acc.set_ylim(*y_lim)

    return fig


def main():
    options = parse_argv()

    statistics = stat_tools.Statistics(path=options.path, params=options.params)
    base_dir = os.path.dirname(options.path)

    for param in statistics.params:
        print(f'param = {param}')

        fig = draw_fig(statistics, by_param=param, facecolor=options.facecolor)

        # showing figure in window
        if not options.quiet:
            plt.show()

        # save image to the same directory as statistics.csv
        image_path = os.path.join(base_dir, f'first_hit_cmpx_{param}.png')
        fig.savefig(image_path)

        # close current figure before drawing again
        plt.close(fig)


if __name__ == '__main__':
    main()
