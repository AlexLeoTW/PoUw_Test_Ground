import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import statistics as stat_tools
import plot_color as c


global_ylim = (None, None)


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


def _get_global_ylim(statistics):
    xs, ys = stat_tools.get_log_dots(statistics, x='acc', y='val_acc')
    return min(xs.min(), ys.min()), max(xs.max(), ys.max())


def draw_logs(ax, logs):
    for idx, log in zip(count(), logs):

        val_acc = np.maximum.accumulate(log['val_acc'])
        ax.plot(range(log.shape[0]), val_acc, label=f'#{idx+1} val_acc')

        ax.scatter(range(log.shape[0]), log['acc'], label=f'#{idx+1} train acc')


def draw_fig(group, logs, facecolor=c.white):
    fig, ax = plt.subplots()

    draw_logs(ax, logs)

    ax.set_ylim(*global_ylim)

    ax.legend(loc='lower right')
    ax.set(xlabel='Epochs', ylabel='Accuracy')
    ax.set_title(' / '.join(
        map(lambda x: f'{x[0]}: {x[1]}', group.items())))

    ax.set_facecolor(facecolor)
    fig.set_facecolor(facecolor)

    return fig


def main():
    options = parse_argv()

    statistics = stat_tools.Statistics(path=options.path, params=options.params)
    base_dir = os.path.dirname(options.path)

    global global_ylim
    global_ylim = _get_global_ylim(statistics)

    # for param in statistics.params:
    for group, logs in statistics.grouped_logs_gen():
        print(f'group = {group}')

        fig = draw_fig(group, logs, facecolor=options.facecolor)

        # showing figure in window
        if not options.quiet:
            plt.show()

        # save image to the same directory as statistics.csv
        image_path = os.path.join(
            base_dir, '_'.join(map(str, group.values())) + '.png')
        fig.savefig(image_path)

        # close current figure before drawing again
        plt.close(fig)


if __name__ == '__main__':
    main()
