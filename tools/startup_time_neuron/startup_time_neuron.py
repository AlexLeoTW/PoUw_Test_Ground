## Warning: this is TF2.0+ Only ###

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tf_tricks
from functools import reduce
from auto_params import parse_argv


def _sort_df(df, by):
    df = df.sort_values(by=by).reset_index(drop=True)
    return df


def _group_by_params(df, params, sorted=False):
    local_df = df if sorted else _sort_df(df, by=params)

    params_combination = local_df[params]
    params_combination.insert(len(params), column='count', value=[1] * params_combination.shape[0])
    params_combination = params_combination.groupby(params).count()
    params_combination = params_combination.reset_index()

    end_index = np.add.accumulate(params_combination['count'].values)
    start_index = end_index - params_combination['count'].values
    end_index = end_index - 1   # count vs. index

    return params_combination[params], start_index, end_index


def _num_trainable_variables(trainable_variables):
    is_model = isinstance(trainable_variables, tf.keras.models.Model)
    is_layer = isinstance(trainable_variables, tf.python.keras.engine.base_layer.Layer)

    if is_model or is_layer:
        trainable_variables = trainable_variables.trainable_variables

    if not type(trainable_variables) == list:
        trainable_variables = [trainable_variables]

    shapes = [variable.shape.as_list() for variable in trainable_variables]
    trainables = map(lambda layer: reduce(lambda x, y: x*y, layer), shapes)

    return sum(trainables)


def _num_trainable_variables_categorized(model):
    result = {'Conv': 0, 'RNN': 0, 'Other': 0}

    for layer in model.layers:
        if isinstance(layer, tf.python.keras.layers.convolutional.Conv):
            result['Conv'] += _num_trainable_variables(layer)
        elif isinstance(layer, tf.python.keras.layers.recurrent.RNN):
            result['RNN'] += _num_trainable_variables(layer)
        else:
            result['Other'] += _num_trainable_variables(layer)

    return result


def _load_model(dirname, filename):
    model_path = os.path.join(dirname, filename)
    return tf.keras.models.load_model(model_path)


def _drop_all_zero(df):
    zeros = (df == 0).all(axis='index')
    zero_col = zeros[zeros].index.tolist()
    return df.drop(zero_col, axis='columns')


def _split_df(df, by='startup_time_avg', sections=2):
    num_rows = df.shape[0]
    idx_s = np.array_split(np.arange(0, num_rows), sections)
    sorted_df = df.sort_values(by=['startup_time_avg'])
    return [sorted_df.iloc[idx] for idx in idx_s]


def _main():
    tf_tricks.allow_growth()

    options = parse_argv()

    print(f'options.path = {options.path}')
    statistics = pd.read_csv(options.path)
    print(f'options.params = {options.params}')
    sorted_stat = _sort_df(statistics, by=options.params)
    params_combination, start_index, end_index = _group_by_params(statistics, options.params)

    columns = None
    values = []

    for start, end in zip(start_index, end_index):
        df_slice = sorted_stat.loc[start:end]
        startup_time_avg = np.average(df_slice['startup_time'].values)

        model_name = df_slice['log_path'].values[0].rsplit('.', 1)[0] + '.h5'
        print(f'resding {model_name}    ...')
        model = _load_model(os.path.dirname(options.path), model_name)

        num_variables = _num_trainable_variables_categorized(model)
        num_variables['startup_time_avg'] = startup_time_avg

        columns = list(num_variables.keys()) if columns is None else columns
        values.append(list(num_variables.values()))

    startup_time_df = pd.DataFrame(np.array(values), columns=columns)
    print(startup_time_df)

    startup_time_df = _drop_all_zero(startup_time_df)
    dfs = _split_df(startup_time_df, by='startup_time_avg', sections=2)
    x_label = startup_time_df.columns[0]
    y_label = startup_time_df.columns[1]
    z_label = startup_time_df.columns[2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for df in dfs:
        ax.scatter(df[x_label].values, df[y_label].values, df[z_label].values)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    plt.show()


if __name__ == '__main__':
    _main()
