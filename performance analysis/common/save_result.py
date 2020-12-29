import os
import pandas as pd
from keras.models import Sequential


def dir_create_if_not_exist(file_path):
    dirname = os.path.dirname(file_path)
    if len(dirname) > 0 and not os.path.exists(os.path.dirname(file_path)):
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
        new_columns = _join_lists(list(entries.keys()), list(history_statistics.columns))
    else:
        new_columns = list(entries.keys())
        history_statistics = pd.DataFrame(columns=new_columns)

    statistics = pd.DataFrame([list(entries.values())], columns=list(entries.keys()))
    history_statistics = history_statistics.append(statistics, sort=False, ignore_index=True)
    history_statistics = history_statistics[new_columns]

    if drop_duplicates:
        history_statistics.drop_duplicates(subset=drop_duplicates, inplace=True, keep='last')

    dir_create_if_not_exist(path)
    statistics_csv = open(path, "w")
    statistics_csv.write(history_statistics.to_csv(index=False))


def _join_lists(list1, list2):
    if list1 == list2:
        return list1

    list1 = list1.copy()
    list2 = list2.copy()
    front = _max_matching_list(list1, list2)

    list1.reverse()
    list2.reverse()
    back = _max_matching_list(list1, list2)
    back.reverse()

    middle = list(set(list1).union(set(list2)) - set(front) - set(back))

    return front + middle + back


def _max_matching_list(list1, list2):
    match = []

    for x, y in zip(list1, list2):
        if x != y:
            break
        match.append(x)

    return match
