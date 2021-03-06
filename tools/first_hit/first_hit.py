import os
import matplotlib.pyplot as plt
from statistics import Statistics
from auto_params import parse_argv
import acc_req_descend

highlighted_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
dimmed_colors = ['#3B5F77', '#AF7C4E', '#467546', '#803A3A', '#776587', '#67514D']

options = parse_argv()
statistics = Statistics(options.path, options.params, normalize_colname=True)
collected_avg = statistics.deep_collect(['val_acc', 'end_time'], avg=False)
first_hits = acc_req_descend.find_first_hit(collected_avg, options.params)

x_lim, y_lim = acc_req_descend.x_y_lim(first_hits, expand=True, margin=0.1, fit_curve=True)
plt.figure()

for param in options.params:
    print('param = {param}'.format(param=param))
    highlighted_colors_iter = iter(highlighted_colors)
    dimmed_colors_iter = iter(dimmed_colors)
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    for param_value in collected_avg[param].drop_duplicates().values:

        selected = collected_avg[collected_avg[param] == param_value]
        sel_hit = first_hits[first_hits[param] == param_value]

        plt.scatter(selected['end_time'], selected['val_acc'],
            alpha=0.5, edgecolors='none', c=next(dimmed_colors_iter))

        plt.scatter(sel_hit['end_time'], sel_hit['val_acc'], [120]*first_hits.shape[0],
            label=str(param_value), c=next(highlighted_colors_iter), edgecolors='black', marker='x')

    acc_req_descend.draw_decending_acc_requirement(plt)

    plt.xlabel("end_time(s)")
    plt.ylabel("val_acc(%)")
    plt.legend(title=param)
    plt.savefig(os.path.join(os.path.dirname(options.path), 'first_hit_{}.jpg'.format(param)))
    plt.cla()
