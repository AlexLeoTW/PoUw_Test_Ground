import os
import sys
import matplotlib.pyplot as plt
from statistics import Statistics
import acc_req_descend

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
statistics_path = sys.argv[1]
params = sys.argv[2:]

statistics = Statistics(statistics_path, params)
collected_avg = statistics.deep_collect(['val_acc', 'end_time'], avg=False)
first_hits = acc_req_descend.find_first_hit(collected_avg, params)

# =========== end setting up ===========

fig, ax = plt.subplots()


def first_hit_avg(df, param):
    selected_cols = df.loc[:, (param, 'end_time')]
    avgs = selected_cols.groupby([param]).mean()
    avgs['stddiv'] = selected_cols.groupby([param]).std()['end_time']
    avgs.reset_index(inplace=True)
    return avgs


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, 0),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


for param in params:
    print('param = {param}'.format(param=param))

    avgs = first_hit_avg(first_hits, param)
    avgs[param] = avgs[param].astype('str')

    print(avgs)

    rects = plt.bar(avgs.loc[:, param], avgs.loc[:, 'end_time'], yerr=avgs.loc[:, 'stddiv'], color=colors)
    autolabel(rects)
    plt.xlabel(param)
    plt.ylabel("end_time(s)")
    plt.title('first hit avg')
    # plt.show()
    # break
    plt.savefig(os.path.join(os.path.dirname(statistics_path), 'first_hit_avg_bar_{}.jpg'.format(param)))
    plt.cla()
