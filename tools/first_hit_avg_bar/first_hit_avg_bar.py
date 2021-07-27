import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

import statistics as stat_tools
import acc_req_descend as acc
import plot_color as c


# figsize = [15, 8]
figsize = None


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
    parser.add_argument('--dark', action='store_true',
                        help='use dark mode')

    args = parser.parse_args()

    if args.dark:
        c.dark_mode()

    args.facecolor = c.transparent if args.facecolor else c.facecolor

    return args


def __extract__(first_hit_df, param):
    param_vals = []
    avg_end_times = []
    stddiv_end_times = []

    for param_val, local_df in first_hit_df.groupby(param).__iter__():
        param_vals.append(param_val)
        avg_end_times.append(local_df['end_time'].mean())
        stddiv_end_times.append(local_df['end_time'].std())

    param_vals = [str(val) for val in param_vals]

    # just print table
    __print_table__(param, param_vals, avg_end_times, stddiv_end_times)

    return param_vals, avg_end_times, stddiv_end_times


def __print_table__(param, param_vals, avg_end_times, stddiv_end_times):
    print(tabulate({param: param_vals,
                    'end_time': avg_end_times,
                    'stddiv': stddiv_end_times},
                    headers='keys', tablefmt='github'))


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, 0),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def draw_fig(first_hit_df, param, facecolor=c.white):

    fig, ax = plt.subplots(figsize=figsize)
    param_vals, avg_end_times, stddiv_end_times = __extract__(first_hit_df, param)

    bars = ax.bar(param_vals, avg_end_times, yerr=stddiv_end_times,
                  color=c.fg_dot_color, ecolor=c.ecolor)
    autolabel(ax, bars)

    ax.set_xlabel(param)
    ax.set_ylabel("end_time(s)")

    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    fig.tight_layout()

    return fig


def main():
    options = parse_argv()

    statistics = stat_tools.Statistics(path=options.path, params=options.params)
    base_dir = os.path.dirname(options.path)

    first_hit_df = stat_tools.find_first_hits(statistics, acc.acc_requirement)

    for param in statistics.params:
        print(f'param = {param}')

        fig = draw_fig(first_hit_df, param=param, facecolor=options.facecolor)

        # showing figure in window
        if not options.quiet:
            plt.show()

        # save image to the same directory as statistics.csv
        image_path = os.path.join(base_dir, f'first_hit_avg_bar_{param}_.png')
        fig.savefig(image_path)

        # close current figure before drawing again
        plt.close(fig)


if __name__ == '__main__':
    main()
