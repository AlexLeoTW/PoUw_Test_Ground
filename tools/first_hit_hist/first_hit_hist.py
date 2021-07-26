import os
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
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


def draw_norm_dist(ax, xs, n_bins=n_bins, color=None, z_offset=0):
    xmin, xmax = ax.get_xlim()
    xticks = np.linspace(xmin, xmax, num=n_bins)

    mean, stddiv = stats.norm.fit(xs)
    density = stats.norm.pdf(xticks, mean, stddiv)

    lines = ax.plot(xticks, density,
                    color=color, linewidth=2,
                    path_effects=[path_effects.withStroke(
                        linewidth=4, foreground='white'
                    )])

    _set_zorder(lines, offset=z_offset)

    _norm_dist_anno(ax, mean, stddiv)

    return lines


def _norm_dist_anno(ax, mean, stddiv):
    vline = ax.vlines(mean, 0, ax.get_ylim()[1], colors='red')
    _set_zorder(vline)

    peak = stats.norm.pdf(mean, mean, stddiv)
    anno = ax.annotate(f'mean={mean:.02f}, stddiv={stddiv:.02f}',
                       xy=(mean, peak), xycoords='data',
                       xytext=(15, 15), textcoords='offset points',
                       arrowprops=dict(arrowstyle='->'))

    _set_zorder(anno)


def draw_hist(ax, xs, n_bins=n_bins, color=None, z_offset=0):
    n, bins, patchs = ax.hist(xs, bins=n_bins, density=True, color=color)

    _set_zorder(patchs, offset=z_offset)

    return patchs


def draw_group(axs, fn, hit_df, groupby, on=True, label=False, **kwargs):
    colors = c.iter_fg_dot_color() if on else c.iter_fg_dot_off_color()
    itera_ax = iter(axs)

    for param_val, local_df in hit_df.groupby(groupby).__iter__():
        color = next(colors)
        ax = next(itera_ax)
        xs = local_df['end_time'].values

        obj = fn(ax, xs, color=color, **kwargs)

        if label:
            _set_label(obj, label=param_val)
            # text annotate in legend form
            ax.legend(title=f'{groupby} = {param_val}', labels=[], labelspacing=0.)


def _set_label(artist, label):
    if isinstance(artist, Iterable):
        # assuming ax.plot / matplotlib.lines.Line2D
        artist[0].set_label(label)
    else:
        # assuming ax.scatter / matplotlib.collections.PathCollection
        artist.set_label(label)


def draw_fig(statistics, by_param, facecolor=c.white):
    hit_df = stat_tools.find_first_hits(statistics, acc.acc_requirement)
    num_groups = hit_df.loc[:, by_param].nunique()

    fig = plt.figure(figsize=[10, 4*num_groups], constrained_layout=True)

    grid = gridspec.GridSpec(num_groups, 1, figure=fig)
    grid.update(wspace=0.025, hspace=0.01)

    axs = [fig.add_subplot(spec) for spec in grid]

    draw_group(axs, draw_hist, hit_df, by_param, label=True)
    draw_group(axs, draw_norm_dist, hit_df, by_param)

    fig.set_facecolor(facecolor)
    for ax in axs:
        ax.set_facecolor(facecolor)
        ax.set_ylabel('probabilistic distribution(%)')
    # only on last(bottom) plot
    ax.set_xlabel('end_time(%)')

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
        image_path = os.path.join(base_dir, f'first_hit_hist_{param}.png')
        fig.savefig(image_path)

        # close current figure before drawing again
        plt.close(fig)


if __name__ == '__main__':
    main()
