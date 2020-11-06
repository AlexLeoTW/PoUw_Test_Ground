'''Loosely Based on Fashion_Mnist CNN by tensorflow team
https://www.tensorflow.org/tutorials/keras/classification
'''

import os
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import fashion_mnist
from keras.losses import SparseCategoricalCrossentropy
from sklearn import metrics

import cmdargv
import save_result
from CustomLogger import CustomLogger
import tf_tricks

batch_size = 128
epochs = 12

# read commandline arguments
options = cmdargv.parse_argv()

# TensorFlow wizardry
if options.allow_growth:
    tf_tricks.allow_growth()

start_time = time.time()    # -------------------------------------------------┐
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
preprocess_time = time.time() - start_time   # --------------------------------┘

start_time = time.time()    # -------------------------------------------------┐
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(options.hidden, activation='relu'))
model.add(Dense(10))

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
startup_time = time.time() - start_time   # -----------------------------------┘

custom_logger = CustomLogger(options.log_path)

start_time = time.time()    # -------------------------------------------------┐
model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_images, test_labels),
          callbacks=[custom_logger])
train_time = time.time() - start_time   # -------------------------------------┘

start_time = time.time()    # -------------------------------------------------┐
pred = model.predict(test_images)
val_time = time.time() - start_time   # ---------------------------------------┘

pred = np.argmax(pred, axis=1)
acc_score = metrics.accuracy_score(test_labels, pred)
print('Accuracy =', acc_score*100)

# write files
print('saving model "{}"...'.format(os.path.basename(options.model_path)))
save_result.save_model(options.model_path, model)

print('saving statistics "{}"...'.format(os.path.basename(options.statistics_path)))
save_result.save_statistics(options.statistics_path, entries={
    'hidden:': options.hidden,
    'acc_score(%)': acc_score,
    'preprocess_time': preprocess_time,
    'startup_time': startup_time,
    'train_time': train_time,
    'val_time': val_time,
    'log_path': options.log_path
}, drop_duplicates=False)
