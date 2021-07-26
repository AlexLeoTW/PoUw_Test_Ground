import os
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections.abc import Iterable

import statistics as stat_tools
import plot_color as c
import acc_req_descend as acc


n_bins = 20
cnt_zorder = 1


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


def _set_zorder(artist, offset=0):
    global cnt_zorder

    if isinstance(artist, Iterable):
        # assuming ax.plot / matplotlib.lines.Line2D
        artist[0].set_zorder(cnt_zorder + offset)
    else:
        # assuming ax.scatter / matplotlib.collections.PathCollection
        artist.set_zorder(cnt_zorder + offset)

    cnt_zorder = cnt_zorder + 1


def draw_theo_cdf(ax, xs, color=None, z_offset=0):
    xticks = ax.get_xticks()
    xmin, xmax = min(xticks), max(xticks)
    # xmin, xmax = min(xs), max(xs)
    xticks = np.linspace(xmin, xmax, num=n_bins*2)

    mean, stddiv = stats.norm.fit(xs)
    t_cdf = stats.norm.cdf(xticks, loc=mean, scale=stddiv)

    lines = ax.plot(xticks, t_cdf, c=color, linestyle='--')

    _set_zorder(lines, offset=z_offset)

    return lines


def draw_cdf(ax, xs, n_bins=n_bins, color=None, z_offset=0):
    xs = np.sort(xs)
    prob = np.arange(len(xs)) / float(len(xs))

    lines = plt.step(xs, prob, c=color)

    _set_zorder(lines, offset=z_offset)

    return lines


def draw_group(ax, fn, hit_df, groupby, on=True, label=False, **kwargs):
    colors = c.iter_fg_dot_color() if on else c.iter_fg_dot_off_color()

    for param_val, local_df in hit_df.groupby(groupby).__iter__():
        color = next(colors)
        xs = local_df['end_time'].values

        obj = fn(ax, xs, color=color, **kwargs)

        if label:
            _set_label(obj, label=param_val)


def _set_label(artist, label):
    if isinstance(artist, Iterable):
        # assuming ax.plot / matplotlib.lines.Line2D
        artist[0].set_label(label)
    else:
        # assuming ax.scatter / matplotlib.collections.PathCollection
        artist.set_label(label)


def draw_fig(statistics, by_param, facecolor=c.white):
    fig, ax = plt.subplots()

    hit_df = stat_tools.find_first_hits(statistics, acc.acc_requirement)

    draw_group(ax, draw_cdf, hit_df, by_param, label=True,
               n_bins=n_bins, z_offset=10)
    draw_group(ax, draw_theo_cdf, hit_df, by_param, on=False)

    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    ax.set(xlabel="end_time(s)", ylabel="Cumulative Distribution(%)")
    ax.legend(title=by_param, loc='upper left')

    fig.tight_layout()

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
        image_path = os.path.join(base_dir, f'first_hit_cdf_{param}.png')
        fig.savefig(image_path)

        # close current figure before drawing again
        plt.close(fig)


if __name__ == '__main__':
    main()
