import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections.abc import Iterable
import progressbar

from config import read_config
import statistics as stat_tools
import acc_req_descend as acc
import plot_color as c


figsize = [15, 8]
# figsize = None
cnt_zorder = 1
important_coords = np.array([[0, 0], [0, 1]])  # coolection of points MUST be in the fig

def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default='fig.yaml', help='path to statistics.csv')

    args = parser.parse_args()
    config = read_config(args.path)

    return config


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


def draw_bg_dots(ax, config):
    print('\tdraw_bg_dots')

    # colors = c.iter_fg_dot_off_color()

    for host in config['hosts']:
        statistics = stat_tools.Statistics(**config['hosts'][host])
        xs, ys = stat_tools.get_log_avg_dots(statistics)

        # path = ax.scatter(xs, ys, c=next(colors), alpha=c.bg_dot_alpha)
        path = ax.scatter(xs, ys, c=c.bg_dot_color, alpha=c.bg_dot_alpha)
        _set_zorder(path)


def draw_first_hits(ax, hostname_all, hit_times_all, hit_accs_all, color=None):
    print('\tdraw_first_hit')

    for host, xs, ys in zip(hostname_all, hit_times_all, hit_accs_all):
        path = ax.scatter(xs, ys, label=host,
                          c=color, edgecolor='white', s=72, marker='x')

        _add_2_important((max(xs), max(ys)))
        _set_zorder(path)


def draw_acc_req_line(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xs, ys = acc.get_acc_req_xy(x_range=(ax.get_xlim()), y_range=(0, 1))

    lines = ax.plot(xs, ys, **c.acc_req_line)

    _set_zorder(lines)


def draw_box_plot(ax, hostname_all, hit_times_all):
    print('\tdraw_box_plot')

    colors = c.iter_fg_dot_color()
    box_plot = ax.boxplot(hit_times_all, vert=False, labels=hostname_all,
                          patch_artist=True)

    for box, flier, color in zip(box_plot['boxes'], box_plot['fliers'], colors):
        box.set_facecolor(color)
        flier.set_markerfacecolor(color)


def collect_hits(config):
    hostname_all = []
    hit_times_all = []
    hit_accs_all = []

    for host in config['hosts']:
        print(f'processing host="{host}"')
        statistics = stat_tools.Statistics(**config['hosts'][host])
        hit_df = stat_tools.find_first_hits(statistics, acc.acc_requirement,
                                            reverse=acc.reverse_acc_req)

        hit_times = hit_df['end_time'].values
        hit_accs  = hit_df['val_acc'].values

        hostname_all.append(host)
        hit_times_all.append(hit_times)
        hit_accs_all.append(hit_accs)

    return hostname_all, hit_times_all, hit_accs_all


def draw_winning_annotate(ax, chance):
    x_lim, y_lim = _get_border()
    offset = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.01

    for idx, c_val in zip(range(len(chance)), chance.values()):
        ax.text(x=offset, y=(idx + 1), s=f'~{c_val:.2%} chance', verticalalignment="center")


def calc_winning_chance(hostname_all, hit_times_all, sample=10000):
    grids = np.meshgrid(*hit_times_all)
    sample = min(grids[0].size, sample)

    # [ [host2_times... : 10000], [host2_times... : 10000], [host3_times... : 10000] ]
    times = [np.random.choice(grid.flatten(), size=sample) for grid in grids]
    times = np.array(times)

    # [min_time1, min_time2...]
    times_min = times.min(axis=0)
    # [ [False...], [False...], [True...] ]
    winning = (times == times_min)  # numpy auto-expand

    win_sum = winning.sum(axis=0)
    weighted = winning / win_sum

    win_total = weighted.sum(axis=1)
    chance = win_total / np.sum(win_total)

    chance = dict(zip(hostname_all, chance))

    return chance


def draw_fig(config):
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    grid = gridspec.GridSpec(2, 1, figure=fig)
    grid.update(wspace=0.025, hspace=0.01)

    ax_acc = fig.add_subplot(grid[0])
    ax_bar = fig.add_subplot(grid[1], sharex=ax_acc)

    hostname_all, hit_times_all, hit_accs_all = collect_hits(config)

    draw_bg_dots(ax_acc, config)
    draw_first_hits(ax_acc, hostname_all, hit_times_all, hit_accs_all)
    draw_acc_req_line(ax_acc)
    draw_box_plot(ax_bar, hostname_all, hit_times_all)

    chance = calc_winning_chance(hostname_all, hit_times_all)
    draw_winning_annotate(ax_bar, chance)

    fig.set_facecolor(config['figure']['facecolor'])
    ax_acc.set_facecolor(config['figure']['facecolor'])
    ax_bar.set_facecolor(config['figure']['facecolor'])

    ax_acc.set_title(config['figure']['title'])
    ax_acc.legend(title='host', loc=config['figure']['legend'])

    ax_acc.set(xlabel='time', ylabel='val_acc')
    ax_bar.set(xlabel='end_time(s)', ylabel='hostname')

    x_lim, y_lim = _get_border()
    ax_acc.set_xlim(*x_lim)
    ax_acc.set_ylim(*y_lim)

    return fig


def main():
    config = parse_argv()

    fig = draw_fig(config)

    if config['figure']['preview']:
        plt.show()

    # save figure if path is not None
    if config['figure']['path'] is not None:
        print(f'saving fig to {config["figure"]["path"]}')
        fig.savefig(config['figure']['path'])

    # close current figure before drawing again
    plt.close(fig)


if __name__ == '__main__':
    main()
