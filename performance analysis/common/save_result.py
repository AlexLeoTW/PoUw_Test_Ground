import os
import pandas as pd
from keras.models import Sequential


def dir_create_if_not_exist(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def save_model(path, model):
    if not isinstance(model, Sequential):
        raise ValueError('"model" must be an instance of keras.model.Sequential')
    if not isinstance(path, str):
        raise ValueError('"path" must be an instance of string')

    dir_create_if_not_exist(path)
    model.save(path)


def save_log(path, fit_history):
    if not isinstance(path, str):
        raise ValueError('"path" must be an instance of string')

    dir_create_if_not_exist(path)
    log_csv = open(path, "w")
    log_csv.write(pd.DataFrame(fit_history).to_csv())


def save_statistics(path, entries, drop_duplicates=False):
    if not isinstance(path, str):
        raise ValueError('"path" must be an string')

    if os.path.isfile(path):
        history_statistics = pd.read_csv(path)
    else:
        history_statistics = pd.DataFrame(columns=list(entries.keys()))

    statistics = pd.DataFrame([list(entries.values())], columns=list(entries.keys()))
    history_statistics = history_statistics.append(statistics, sort=False, ignore_index=True)

    if drop_duplicates:
        history_statistics.drop_duplicates(subset=drop_duplicates, inplace=True, keep='last')

    dir_create_if_not_exist(path)
    statistics_csv = open(path, "w")
    statistics_csv.write(history_statistics.to_csv(index=False))
