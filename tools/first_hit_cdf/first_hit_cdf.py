import os
import matplotlib.pyplot as plt
from statistics import Statistics
from auto_params import parse_argv
import acc_req_descend

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
figsize = None
n_bins = 20

options = parse_argv()
statistics = Statistics(options.path, options.params, normalize_colname=True)
collected_avg = statistics.deep_collect(['val_acc', 'end_time'], avg=False)
first_hits = acc_req_descend.find_first_hit(collected_avg, options.params)


def _get_perm_values(df, param):
    return (df.loc[:, param]).drop_duplicates().values


def _draw_cdf(xs, n_bins):
    plt.figure(figsize=figsize)
    plt.hist(xs, bins=n_bins, density=True, cumulative=True, label=parm_values,
             histtype='step', color=colors[:len(parm_values)])
    plt.legend(title=param)


for param in options.params:
    parm_values = _get_perm_values(first_hits, param=param)
    first_hits_categorized = []

    print('{param}: {parm_values}'.format(param=param, parm_values=parm_values))

    # collect "first_hits_categorized" of shape(len(parm_value), round)
    for parm_value in parm_values:
        df_by_parm = first_hits[first_hits[param] == parm_value]
        end_times = df_by_parm.loc[:, 'end_time'].to_list()
        first_hits_categorized.append(end_times)

    _draw_cdf(first_hits_categorized, n_bins)

    plt.savefig(os.path.join(os.path.dirname(options.path), f'first_hit_cdf_{param}.jpg'))
    plt.cla()
