import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from auto_params import parse_argv

threshold = 10


def _add_annotate(bp, ax):
    """ This actually adds the numbers to the various points of the boxplots"""
    for element in ['whiskers', 'medians', 'caps']:
        for line in bp[element]:
            # Get the position of the element. y is the label you want
            (x, y_l), (_, y_u) = line.get_xydata()
            # array([[686.20377672,   0.925     ],
            #        [686.20377672,   1.075     ]])
            # Make sure datapoints exist
            # (I've been working with intervals, should not be problem for this case)
            if not np.isnan(x):
                x_line_center = x
                y_line_center = y_l + (y_u - y_l) / 2
                ax.annotate('{:.2f}'.format(x),
                            xy=(x_line_center, y_line_center))


def _main():
    options = parse_argv()
    statistics = pd.read_csv(options.path)

    train_time = statistics['train_time'].values.tolist()
    val_time = statistics['val_time'].values.tolist()

    # TODO: print Markdown table, and include 1.5*lr
    # df = pd.DataFrame(data={
    #     'Min': [min(train_time), min(val_time)],
    #     'Q1': [np.quantile(train_time, [0.25]), np.quantile(val_time, [0.25])],
    #     'Q2': [np.quantile(train_time, [0.5]), np.quantile(val_time, [0.5])],
    #     'Q3': [np.quantile(train_time, [0.75]), np.quantile(val_time, [0.75])],
    #     'Max': [max(train_time), max(val_time)]
    # }, index=['train_time', 'val_time'])
    # print(df)

    data = [train_time, val_time]

    plot_diff = np.ptp(np.concatenate((
        statistics['train_time'].values,
        statistics['val_time'].values)))
    train_time_diff = np.ptp(statistics['train_time'].values)
    val_time_diff = np.ptp(statistics['val_time'].values)

    # if the smaller boxplot is smaller then 1/10 the plot
    if threshold * min([train_time_diff, val_time_diff]) < plot_diff:
        # draw separately
        fig, ax = plt.subplots(nrows=2, ncols=1)
        bp0 = ax[0].boxplot(train_time, vert=False, labels=['train_time'])
        bp1 = ax[1].boxplot(val_time, vert=False, labels=['val_time'])

        _add_annotate(bp0, ax[0])
        _add_annotate(bp1, ax[1])
    else:
        # draw as single plot
        fig, ax = plt.subplots()
        bp = ax.boxplot(data, vert=False, labels=['train_time', 'val_time'])
        _add_annotate(bp, ax)

    plt.show()


if __name__ == '__main__':
    _main()
