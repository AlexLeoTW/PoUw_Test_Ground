import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy import stats
from statistics import Statistics
from auto_params import parse_argv
import acc_req_descend

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
figsize = [15, 8]
n_bins = 20

options = parse_argv()
statistics = Statistics(options.path, options.params, normalize_colname=True)
collected_avg = statistics.deep_collect(['val_acc', 'end_time'], avg=False)
first_hits = acc_req_descend.find_first_hit(collected_avg, options.params)


def _get_perm_values(df, param):
    return (df.loc[:, param]).drop_duplicates().values


def _draw_hist(plt, xs):
    plt.hist(first_hits_categorized, bins=n_bins, density=True, label=parm_values,
             color=colors[:len(parm_values)])


def _draw_norm_dist(plt, xs):
    xticks = plt.xticks()[0]
    xmin, xmax = min(xticks), max(xticks)

    for index, hits in enumerate(xs):
        xticks = np.linspace(xmin, xmax, max(len(hits), n_bins))
        mean, stddiv = stats.norm.fit(hits)
        pdf_g = stats.norm.pdf(xticks, mean, stddiv)
        plt.plot(xticks, pdf_g,
                 color=colors[index], linewidth=2,
                 path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])


for param in options.params:
    parm_values = _get_perm_values(first_hits, param=param)
    first_hits_categorized = []

    print(f'{param}: {parm_values}')

    # collect "first_hits_categorized" of shape(len(parm_value), test_rounds)
    for parm_value in parm_values:
        df_by_parm = first_hits[first_hits[param] == parm_value]
        end_times = df_by_parm.loc[:, 'end_time'].to_list()
        first_hits_categorized.append(end_times)

    plt.figure(figsize=figsize)
    _draw_hist(plt, first_hits_categorized)
    _draw_norm_dist(plt, first_hits_categorized)
    plt.legend(title=param)

    plt.savefig(os.path.join(os.path.dirname(options.path), f'first_hit_hist_{param}.jpg'))
    plt.cla()
