import numpy as np
from auto_params import parse_argv
from statistics import avg_statistics
import matplotlib.pyplot as plt
import os


def _main():
    options = parse_argv()
    avg = avg_statistics(options.path, options.params)
    plot_config = read_config()

    targets = [plot['col_name'] for plot in plot_config]
    values = _collect_values(avg, options.params, targets)
    labels = _collect_labels(avg, options.params)

    for param in values:
        print(f'drawing fig "pref_{param}.jpg"')

        fig = _draw_plot(values[param], plot_config, labels[param], title=param)
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(
            os.path.join(os.path.dirname(options.path), f'pref_{param}.jpg'))
        plt.close('all')


def _collect_values(avg, params, targets):
    values = {
        # 'conv1_filters': {
        #     'acc_score(%)': {16: [], 32: [], 64: []},
        #     'train_time': {16: [], 32: [], 64: []},
        #     ...
        # },
        # 'conv1_kernel_size': {
        #     'acc_score(%)': {2: [], 3: [], 4: []},
        #     'train_time': {2: [], 3: [], 4: []},
        #     ...
        # }
    }

    # init values
    for param in params:
        values[param] = {}
        for target in targets:
            values[param][target] = {}

    for param in params:
        param_avg = avg.set_index(param)
        param_avg = param_avg[targets]
        p_values = list(set(param_avg.index.sort_values()))

        for p_value in p_values:
            temp = param_avg.loc[p_value].reset_index(drop=True).to_dict(orient='list')

            for tnl in targets:
                values[param][tnl][p_value] = temp[tnl]

    return values


def _collect_labels(avg, params):
    labels = {
        # 'conv1_filters': [...],
        # 'conv1_kernel_size': [...]
    }

    def _labeler(row):
        out = ''
        for key, value in row[params_list].items():
            out += f'{key}_{int(value)}\n'

        return out

    for param in params:
        hide_param = param
        params_list = params.copy()
        params_list.remove(hide_param)
        labels[param] = avg[params_list].drop_duplicates().apply(_labeler, axis=1)

    return labels


def _draw_plot(values, plot_config, labels=[], title=None):
    num_targets = len(values)
    num_group = max(map(    # dual-loop, find longest list
        lambda k1: max(map(
            lambda k2: len(values[k1][k2]), values[k1]))
        , values))
    figsize = (2 * num_group, 5 * num_targets)

    fig, axes = plt.subplots(nrows=num_targets, ncols=1, figsize=figsize)
    if num_targets == 1:
        axes = [axes]

    for ax, target, config in zip(axes, values, plot_config):
        _draw_subplot(ax, values[target], labels, annotate=config['annotate'], scale=config['scale'])
        ax.set(ylabel=target)
        ax.set_xticks([])

    ax.set_xticks(range(num_group))
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

    if title:
        fig.suptitle(title)

    return fig


def _draw_subplot(ax, values, labels=[], annotate=False, scale=False):
    # values = { 16: [...], 32: [...], 64: [...] }
    num_categories = len(values.keys())
    num_bar_group = max([len(values[key]) for key in values])
    bar_width = 1.0 / (num_categories + 1)
    x_loc_iter = _x_loc_iter(bar_width, num_categories, num_bar_group)

    for label, value in values.items():
        rects = ax.bar(next(x_loc_iter), value, width=bar_width, label=label)

        if annotate:
            _draw_annotate(rects, ax, value, annotate)

    # ax.set_xticks(range(num_bar_group))
    # ax.set_xticklabels(labels)
    # plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
    #      rotation_mode="anchor")

    ax.set_xlim(-0.5, num_bar_group)
    if scale:
        ax.set_ylim(*_ylim(values))


def _draw_annotate(rects, ax, value, annotate='{.2f}'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(annotate.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', annotation_clip=True)


def _ylim(values):
    values = list(map(lambda key: values[key], values))
    values = np.array(values)
    v_min = values.min()
    v_max = values.max()
    range = v_max - v_min

    x_min = max((v_min - range * 0.1), 0)
    x_max = v_max + range * 0.1

    return (x_min, x_max)


def _x_loc_iter(bar_width, num_categories, num_bar_group):
    x_loc_base = np.arange(num_bar_group)

    for idx_bar in range(num_categories):
        yield x_loc_base + (idx_bar * bar_width)


def read_config():
    from conf import plots

    for plot in plots:
        # ============================== col_name ==============================
        assert 'col_name' in plot, 'must specify "col_name"'

        # ============================= omit_syle ==============================
        if 'title' not in plot or plot['title'] is None:
            plot['title'] = plot['col_name']

        # ============================== annotate ==============================
        if 'scale' not in plot:
            plot['omit_syle'] = False

        # =============================== title ================================
        if 'annotate' not in plot:
            plot['annotate'] = None

    return plots


if __name__ == '__main__':
    _main()
