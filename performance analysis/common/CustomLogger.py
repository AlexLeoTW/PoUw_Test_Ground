import time
import keras
import pandas as pd


def dict_to_df_row(input):
    # if input has only 1 dimension
    if not isinstance(list(input.values())[0], list):
        return pd.DataFrame([list(input.values())], columns=input.keys())
    else:
        return pd.DataFrame(input)


class CSVManager():
    def __init__(self, path):
        self.file = open(path, 'w', newline='')
        self.header_written = False

    def insert_row(self, df):
        if self.header_written:
            self.file.write(df.to_csv(header=False, index=False))
        else:
            self.file.write(df.to_csv(header=True, index=False))
            self.header_written = True

        self.file.flush()

    def close(self):
        self.file.close()


class CustomLogger(keras.callbacks.Callback):

    def __init__(self, path=None, metrics=None):
        self.metrics = metrics
        self.csv = CSVManager(path)

        self.train_begin = None
        self.epoch_time_start = None
        self.epoch_time_end = None

    def on_train_begin(self, logs={}):
        self.train_begin = time.time()

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.epoch_time_end = time.time()

        logs['start_time'] = self.epoch_time_start
        logs['end_time'] = self.epoch_time_end

        new_row = dict_to_df_row(logs)
        # new_row = self.__dict_to_df_row(logs)
        new_row = new_row if self.metrics is None else new_row[self.metrics]
        self.csv.insert_row(new_row)

    def on_train_end(self, logs={}):
        self.csv.close()
        self.train_end = time.time()
        self.train_time = self.train_end - self.train_begin
