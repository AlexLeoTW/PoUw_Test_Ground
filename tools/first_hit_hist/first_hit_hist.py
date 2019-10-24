import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy import stats
from statistics import Statistics
import acc_req_descend

statistics_path = sys.argv[1]
params = sys.argv[2:]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
figsize = [15, 8]
n_bins = 20

statistics = Statistics(statistics_path, params)
collected_avg = statistics.deep_collect(['val_acc', 'end_time'], avg=False)
first_hits = acc_req_descend.find_first_hit(collected_avg, params)


def _get_perm_values(df, param):
    return (df.loc[:, param]).drop_duplicates().values


def _draw_hist(plt, xs, n_bins):
    plt.hist(first_hits_categorized, bins=n_bins, density=True, label=parm_values,
             color=colors[:len(parm_values)])
    plt.legend(title=param)


def _draw_norm_dist(plt, xs):
    xticks = plt.xticks()[0]
    xmin, xmax = min(xticks), max(xticks)

    for index, hits in enumerate(xs):
        xticks = np.linspace(xmin, xmax, len(hits))
        mean, stddiv = stats.norm.fit(hits)
        pdf_g = stats.norm.pdf(xticks, mean, stddiv)
        plt.plot(xticks, pdf_g,
                 color=colors[index], linewidth=2,
                 path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])


for param in params:
    parm_values = _get_perm_values(first_hits, param=param)
    first_hits_categorized = []

    print('{param}: {parm_values}'.format(param=param, parm_values=parm_values))

    # collect "first_hits_categorized" of shape(round, len(parm_value))
    for parm_value in parm_values:
        df_by_parm = first_hits.query('{param} == {parm_values}'.format(param=param, parm_values=parm_value))
        end_times = df_by_parm.loc[:, 'end_time'].to_list()
        first_hits_categorized.append(end_times)

    plt.figure(figsize=figsize)
    _draw_hist(plt, first_hits_categorized, n_bins)
    _draw_norm_dist(plt, first_hits_categorized)

    # plt.show()
    # break
    plt.savefig(os.path.join(os.path.dirname(statistics_path), 'first_hit_hist_{}.jpg'.format(param)))
    plt.cla()
