'''Loosely Based on Mnist CNN by keras-team
https://keras.io/examples/mnist_cnn/
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
'''

import os
import time
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn import metrics
import cmdargv
import save_result
from CustomLogger import CustomLogger

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# read commandline arguments
options = cmdargv.parse_argv()
print('options = {}'.format(options))

# TensorFlow wizardry
if options.allow_growth:
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    import keras.backend as k_backend
    k_backend.tensorflow_backend.set_session(tf.Session(config=config))

start_time = time.time()    # -------------------------------------------------┐
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
preprocess_time = time.time() - start_time   # --------------------------------┘

custom_logger = CustomLogger(options.log_path)

start_time = time.time()    # -------------------------------------------------┐
model = Sequential()
model.add(Conv2D(options.conv1[0], kernel_size=(options.conv1[1], options.conv1[1]),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(options.conv2, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(options.pool, options.pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(options.dense, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
startup_time = time.time() - start_time   # -----------------------------------┘

start_time = time.time()    # -------------------------------------------------┐
result = model.fit(x_train, y_train,     #                                     |
          batch_size=batch_size,    #          train_time                      |
          epochs=epochs,    #                                                  |
          verbose=1,        #                                                  |
          validation_data=(x_test, y_test), #                                  |
          callbacks=[custom_logger])   #                                       |
score = model.evaluate(x_test, y_test, verbose=0)   #                          |
train_time = time.time() - start_time   # -------------------------------------┘
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('custom_logger.train_begin = {}'.format(custom_logger.train_begin))
print('custom_logger.train_end = {}'.format(custom_logger.train_end))
print('custom_logger.train_time = {}'.format(custom_logger.train_time))

# measure accuracy
start_time = time.time()    # -------------------------------------------------┐
pred = model.predict(x_test) #                       val_time                  |
val_time = time.time() - start_time   # ---------------------------------------┘

pred = np.argmax(pred, axis=1)
print('pred.shape = {}'.format(pred.shape))
y_eval = np.argmax(y_test, axis=1)
print('y_eval.shape = {}'.format(y_eval.shape))

acc_score = metrics.accuracy_score(y_eval, pred)
print('Accuracy =', acc_score*100)

# write files
print('saving model "{}"...'.format(os.path.basename(options.model_path)))
save_result.save_model(options.model_path, model)

print('saving statistics "{}"...'.format(os.path.basename(options.statistics_path)))
save_result.save_statistics(options.statistics_path, entries = {
    'conv1_filters': options.conv1[0],
    'conv1_kernel_size': options.conv1[1],
    'conv2_filters': options.conv2,
    'pool': options.pool,
    'dense': options.dense,
    'acc_score(%)': acc_score,
    'preprocess_time': preprocess_time,
    'startup_time': startup_time,
    'train_time': train_time,
    'val_time': val_time,
    'log_path': options.log_path
}, drop_duplicates=False)
