import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy import stats
from statistics import Statistics
import acc_req_descend

# statistics_path = 'statistics.csv'
statistics_path = sys.argv[1]
params = ['conv1_filters', 'conv1_kernel_size', 'conv2_filters', 'pool', 'dense']
# params = sys.argv[2:]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
figsize = [15, 8]

statistics = Statistics(statistics_path, params)
collected_avg = statistics.deep_collect(['val_acc', 'end_time'], avg=False)
first_hits = acc_req_descend.find_first_hit(collected_avg, params)

# first_hits[first_hits['conv1_filters'] == 16]
# first_hits.loc[:, ['conv1_filters', 'end_time']].head()
# print(first_hits[first_hits['conv1_filters'] == 16].loc[:, 'conv1_kernel_size'].drop_duplicates())


def _get_perm_values(df, param):
    return (df.loc[:, param]).drop_duplicates().values


for param in params:
    parm_values = _get_perm_values(first_hits, param=param)
    first_hits_categorized = []

    print('{param}: {parm_values}'.format(param=param, parm_values=parm_values))

    # collect "first_hits_categorized" of shape(round, len(parm_value))
    for parm_value in parm_values:
        df_by_parm = first_hits.query('{param} == {parm_values}'.format(param=param, parm_values=parm_value))
        df_by_parm = df_by_parm.loc[:, 'end_time'].to_list()
        first_hits_categorized.append(df_by_parm)

    plt.figure(figsize=figsize)
    plt.hist(first_hits_categorized, bins=20, density=True, label=parm_values, color=colors[:len(parm_values)])
    plt.legend(title=param)

    xticks = plt.xticks()[0]
    xmin, xmax = min(xticks), max(xticks)

    for index, hits in enumerate(first_hits_categorized):
        lnspc = np.linspace(xmin, xmax, len(hits))
        mean, stddiv = stats.norm.fit(hits)
        pdf_g = stats.norm.pdf(lnspc, mean, stddiv)
        plt.plot(lnspc, pdf_g, color=colors[index], linewidth=2,  path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])

    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(statistics_path), 'first_hit_hist_{}.jpg'.format(param)))
    plt.cla()
