import csv
import time
import keras

metrics = ['val_loss', 'val_acc', 'loss', 'acc', 'start_time', 'end_time']


class CustomLogger(keras.callbacks.Callback):

    def __init__(self, path=None, metrics=metrics):
        super(CustomLogger, self).__init__()

        self.logfile = open(path, 'w', newline='')
        self.csv = csv.DictWriter(self.logfile, fieldnames=metrics)
        self.csv.writeheader()

        self.metrics = metrics
        self.result = {}
        for metrix in self.metrics:
            self.result[metrix] = []

    def on_train_begin(self, logs={}):
        self.train_begin = time.time()

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time() - self.train_begin

    def on_epoch_end(self, batch, logs={}):
        self.epoch_time_end = time.time() - self.train_begin

        logs['start_time'] = self.epoch_time_start
        logs['end_time'] = self.epoch_time_end

        new_row = {}
        for metrix in self.metrics:
            new_row[metrix] = logs[metrix] if metrix in logs else float('nan')
            self.result[metrix].append(new_row[metrix])

        self.csv.writerow(new_row)

    def on_train_end(self, logs={}):
        self.logfile.close()
        self.train_end = time.time()
        self.train_time = self.train_end - self.train_begin
