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
important_coords = np.array([[0, 0]])  # coolection of points MUST be in the fig


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


def _add_2_important(points):
    global important_coords

    points = np.atleast_2d(points)
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


def draw_bg_dots(ax, statistics, hits=False):
    print('draw_bg_dots')
    bg_xs, bg_ys = stat_tools.get_log_dots(statistics)

    path = ax.scatter(bg_xs, bg_ys, c=c.bg_dot_color, alpha=c.bg_dot_alpha,
                      label='all')
    _set_zorder(path)

    if hits:
        _draw_hits(ax, statistics, color=c.bg_dot_color)




def _find_best_param_val(avg_hit_df, param):
    avg_by_val = avg_hit_df.groupby(param).mean().reset_index()
    best = avg_by_val.sort_values(by='end_time').iloc[0]

    return best[param]


def _find_best_params(statistics):
    avg_hit_df = stat_tools.find_first_hits_avg(statistics, acc.acc_requirement)
    best_df = avg_hit_df.copy()  # will get trimmed later

    for param in statistics.params:
        best = _find_best_param_val(avg_hit_df, param)
        best_df = best_df[best_df[param] == best]

    best_params = best_df.iloc[0]  # make it a Series
    best_params = best_params[statistics.params].to_dict()

    return best_params


def draw_best_dots(ax, statistics, hits=False):
    print('draw_best_dots')
    color = next(c.iter_fg_dot_color())

    best_params = _find_best_params(statistics)
    best_stat = statistics.select_by_values(best_params)

    xs, ys = stat_tools.get_log_dots(best_stat)
    path = ax.scatter(xs, ys, c=color, label='best')
    _set_zorder(path)

    if hits:
        _draw_hits(ax, best_stat, color=color)


def _draw_hits(ax, stat, color=None):
    print('\t_draw_hits')
    hits_df = stat_tools.find_first_hits(stat, acc.acc_requirement)
    xs = hits_df['end_time']
    ys = hits_df['val_acc']
    path = ax.scatter(xs, ys, c=color, edgecolor='white', s=72)

    _add_2_important(list(zip(xs, ys)))
    _set_zorder(path)


def draw_acc_req_line(ax):
    print('draw_acc_req_line')
    xs, ys = acc.get_acc_req_xy(x_range=(ax.get_xlim()), y_range=(ax.get_ylim()))

    plot_arg = {'c': c.deep_gray, 'linestyle': '--', 'linewidth': '2'}
    lines = ax.plot(xs, ys, **plot_arg)

    _set_zorder(lines)


def draw_vert_avg(ax, statistics):
    print('draw_vert_avg')
    ymax = ax.get_ylim()[1]

    a_hits_df = stat_tools.find_first_hits(statistics, acc.acc_requirement)
    a_mean = a_hits_df['end_time'].mean()

    a_line = ax.vlines(a_mean, ymin=0, ymax=ymax, colors=c.bg_dot_color,
                       linewidth=3, path_effects=[c.white_outline])
    _set_zorder(a_line)

    best_params = _find_best_params(statistics)
    best_stat = statistics.select_by_values(best_params)
    b_hits_df = stat_tools.find_first_hits(best_stat, acc.acc_requirement)
    b_mean = b_hits_df['end_time'].mean()

    b_color = next(c.iter_fg_dot_color())
    b_line = ax.vlines(b_mean, ymin=0, ymax=ymax, colors=b_color,
                       linewidth=3, path_effects=[c.white_outline])
    _set_zorder(b_line)

    _add_2_important([(a_mean, 0), (b_mean, 0)])


def draw_fig(statistics, facecolor=c.white, focus=False):
    fig, ax = plt.subplots(figsize=figsize)

    draw_bg_dots(ax, statistics, hits=True)
    draw_best_dots(ax, statistics, hits=True)
    draw_acc_req_line(ax)
    draw_vert_avg(ax, statistics)

    ax.set(xlabel='time(s)', ylabel='val_acc')
    ax.legend()

    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    if focus:
        x_lim, y_lim = _get_border()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
    else:
        ax.set_ylim(bottom=0)

    fig.tight_layout()

    return fig


def _main():
    options = parse_argv()

    statistics = stat_tools.Statistics(path=options.path, params=options.params)
    base_dir = os.path.dirname(options.path)

    fig = draw_fig(statistics, facecolor=options.facecolor, focus=options.focus)

    # showing figure in window
    if not options.quiet:
        plt.show()

    # save image to the same directory as statistics.csv
    image_path = os.path.join(base_dir, 'best_param.png')
    fig.savefig(image_path)

    # close current figure before drawing again
    plt.close(fig)


if __name__ == '__main__':
    _main()
