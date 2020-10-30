import os
import matplotlib.pyplot as plt
import statistics as stat_tools
from auto_params import parse_argv
import acc_req_descend
import tf_tricks

tf_tricks.allow_growth()
figsize = [15, 8]


def _find_best_params(first_hits, params):
    best_params = {}
    for param in params:
        first_hit_avg = first_hits.groupby(param).mean().reset_index()
        best = first_hit_avg.sort_values(by='end_time').iloc[0]
        best_params[param] = best[param]
    return best_params
    # lowest_end_time = first_hits.sort_values(by='end_time').iloc[0]
    # best_params = lowest_end_time[params]
    # return best_params.to_dict()


def _main():
    options = parse_argv()
    statistics = stat_tools.Statistics(options.path, options.params, normalize_colname=True)
    collected = statistics.deep_collect(['val_acc', 'end_time'], avg=False)
    first_hits = acc_req_descend.find_first_hit(collected, options.params)

    best_params = _find_best_params(first_hits, options.params)
    best_avg = stat_tools.select_by_values(collected, best_params)
    print(f'best_params = {best_params}')

    first_end_time = first_hits.sort_values(by='end_time').iloc[0]
    first_val_acc, first_end_time = first_end_time[['val_acc', 'end_time']].values

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(collected['end_time'].values, collected['val_acc'].values, label='all')
    ax.scatter(best_avg['end_time'].values, best_avg['val_acc'].values, label='best')
    ax.scatter([first_end_time], [first_val_acc], s=100, c='black', edgecolors='black', marker='x')
    ax.legend()

    acc_req_descend.draw_decending_acc_requirement(ax)
    plt.savefig(os.path.join(
        os.path.dirname(options.path),
        'best_param.jpg'))


if __name__ == '__main__':
    _main()
