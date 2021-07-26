import os
import argparse
import matplotlib.pyplot as plt

import statistics as stat_tools
import plot_color as c


figsize = [15, 8]


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
    parser.add_argument('--style', choices=['line', 'dot'], default='dot',
                        help='plot with "line" or "dot"')
    parser.add_argument('--max', action='store_true',
                        help='plot with accumulated max accuracy')

    args = parser.parse_args()

    args.facecolor = c.transparent if args.facecolor else c.facecolor

    return args


def draw_parm_dots(ax, statistics, param, max=False):
    param_vals = statistics.params_combination[param].unique()
    y_mode = 'avg_max' if max else 'avg'

    for param_val in param_vals:
        print(f'\tparam_val = {param_val}')
        local_stat = statistics.select_by_values({param: param_val})
        xs, ys = stat_tools.get_log_avg_dots(local_stat, y_mode=y_mode)

        ax.scatter(xs, ys, label=param_val)


def draw_parm_lines(ax, statistics, param, max=False):
    param_vals = statistics.params_combination[param].unique()
    y_mode = 'avg_max' if max else 'avg'

    for param_val, color in zip(param_vals, c.iter_fg_dot_color()):
        print(f'\tparam_val = {param_val}')
        local_stat = statistics.select_by_values({param: param_val})

        for avg_x, avg_y in stat_tools.gen_log_avg_dots(local_stat, y_mode=y_mode):
            lines = ax.plot(avg_x, avg_y, c=color)

        lines[0].set_label(param_val)


def draw_fig(statistics, param, style='dot', max=False, facecolor=c.white):
    fig, ax = plt.subplots(figsize=figsize)
    bg_xs, bg_ys = stat_tools.get_log_dots(statistics)

    ax.scatter(bg_xs, bg_ys, c=c.bg_dot_color, alpha=c.bg_dot_alpha)
    if style == 'dot':
        draw_parm_dots(ax, statistics, param, max=max)
    elif style == 'line':
        draw_parm_lines(ax, statistics, param, max=max)
    else:
        raise ValueError('style should be "dot" or "line"')

    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    ax.set_xlabel('end_time(s)')
    ax.set_ylabel('{acc}(%)'.format(acc='val_acc'))

    ax.legend(title=param, loc='lower right')

    fig.tight_layout()

    return fig


def main():
    options = parse_argv()

    statistics = stat_tools.Statistics(path=options.path, params=options.params)
    base_dir = os.path.dirname(options.path)

    for param in statistics.params:
        print(f'param = {param}')

        # create a new blank figure
        fig = draw_fig(statistics, param,
                       style=options.style, max=options.max,
                       facecolor=options.facecolor)

        # showing figure in window
        if not options.quiet:
            plt.show()

        # save image to the same directory as statistics.csv
        image_path = os.path.join(base_dir, f'acc_growth_{param}.png')
        fig.savefig(image_path)

        # close current figure before drawing again
        plt.close(fig)


if __name__ == '__main__':
    main()
