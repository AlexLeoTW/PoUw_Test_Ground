import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import Statistics

# statistics_path = 'statistics.csv'
params = ['conv1_filters', 'conv1_kernel_size', 'conv2_filters', 'pool', 'dense']
statistics_path = sys.argv[1]
# params = sys.argv[2:]

statistics = Statistics(statistics_path, params)
plt.figure(figsize=[30, 16])

# fix log_path
statistics.statistics['log_path'] = statistics.statistics['log_path'].apply(lambda log_path: os.path.join(os.path.dirname(statistics_path), os.path.basename(log_path)))

# collect avg-ed data
collected_avg = statistics.deep_collect(['val_acc', 'end_time'], avg=True)


def acc_requirement(x):
    return (-0.00003) * x + 1


def draw_decending_acc_requirement(f):
    xs = np.arange(0.0, 1000, 5)
    ys = f(xs)
    plt.plot(xs, ys, c='#303030', linestyle='--', linewidth='2')


def find_first_hit(df, params):
    df.insert(loc=len(params), column='pass', value=df['val_acc'] >= acc_requirement(df['end_time']))

    # first_hit = pd.DataFrame([], columns=pd.columns)
    #
    # for index, row in df.iterrows():
    #     if row['pass']:
    #         first_hit.append(row)
    first_hit = df[df['pass']]
    first_hit = first_hit.drop_duplicates(subset=params)
    return first_hit


def colors():
    highlighted_colors = iter(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    dimmed_colors = iter(['#3B5F77', '#AF7C4E', '#467546', '#803A3A', '#776587', '#67514D'])
    return highlighted_colors, dimmed_colors


for param in params:

    print('param = {param}'.format(param=param))
    draw_decending_acc_requirement(f=acc_requirement)
    highlighted_colors, dimmed_colors = colors()

    for param_value in collected_avg[param].drop_duplicates().values:

        selected = collected_avg[collected_avg[param] == param_value]
        plt.scatter(selected['end_time'], selected['val_acc'],
            alpha=0.5, edgecolors='none', c=next(dimmed_colors))

        first_hit = find_first_hit(selected, params)
        plt.scatter(first_hit['end_time'], first_hit['val_acc'], label=str(param_value),
            c=next(highlighted_colors))

    plt.xlabel("end_time(s)")
    plt.ylabel("val_acc(%)")
    plt.legend(title=param)
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(statistics_path), 'first_hit_{}.jpg'.format(param)))
    # break
    plt.cla()
