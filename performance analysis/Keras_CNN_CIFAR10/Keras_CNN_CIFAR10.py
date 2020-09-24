'''Loosely Based on CIFAR10 CNN by keras-team
https://keras.io/examples/cifar10_cnn/
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
'''

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import math
import time
import numpy as np
from sklearn import metrics
import cmdargv
import save_result
from CustomLogger import CustomLogger
import tf_tricks

batch_size = 32
num_classes = 10
epochs = 100
num_predictions = 20

# read commandline arguments
options = cmdargv.parse_argv()
print('options = {}'.format(options))

# TensorFlow wizardry
if options.allow_growth:
    tf_tricks.allow_growth()

start_time = time.time()    # -------------------------------------------------┐
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

preprocess_time = time.time() - start_time   # --------------------------------┘

custom_logger = CustomLogger(options.log_path)

start_time = time.time()    # -------------------------------------------------┐
model = Sequential()

if options.stack == 'independent':
    model.add(Conv2D(filters=options.conv[0], kernel_size=(options.conv[1], options.conv[1]), padding='same',
                     activation='relu',
                     input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(options.pool, options.pool)))
    model.add(Dropout(rate=0.25))

    for num in range(2, options.conv_num+1):
        model.add(Conv2D(filters=options.conv[0], kernel_size=(options.conv[1], options.conv[1]),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(options.pool, options.pool)))
        model.add(Dropout(rate=0.25))

elif options.stack == '2_in_a_row':
    model.add(Conv2D(filters=options.conv[0], kernel_size=(options.conv[1], options.conv[1]), padding='same',
                     activation='relu',
                     input_shape=x_train.shape[1:]))
    model.add(Conv2D(filters=options.conv[0], kernel_size=(options.conv[1], options.conv[1]),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(options.pool, options.pool)))
    model.add(Dropout(rate=0.25))

    for num in range(3, options.conv_num+1, 2):
        model.add(Conv2D(filters=options.conv[0], kernel_size=(options.conv[1], options.conv[1]),
                         activation='relu'))
        model.add(Conv2D(filters=options.conv[0], kernel_size=(options.conv[1], options.conv[1]),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(options.pool, options.pool)))
        model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=num_classes, activation='softmax'))

optimizers = keras.optimizers.Adam(lr=0.0002)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers,
              metrics=['accuracy'])
startup_time = time.time() - start_time   # -----------------------------------┘

model.summary()

if not options.aug:
    start_time = time.time()    # ---------------------------------------------┐
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[custom_logger])
    train_time = time.time() - start_time   # ---------------------------------┘
else:
    start_time = time.time()    # ---------------------------------------------┐
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.6, 1.4],
        shear_range=0.1,
        zoom_range=0.1,
        channel_shift_range=10,
        fill_mode='nearest',
        horizontal_flip=True,
        data_format='channels_last'
    )
    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        steps_per_epoch=math.ceil(x_train.shape[0] / batch_size),
                        validation_data=(x_test, y_test),
                        callbacks=[custom_logger])
    train_time = time.time() - start_time   # ---------------------------------┘

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

start_time = time.time()    # -------------------------------------------------┐
pred = model.predict(x_test)
val_time = time.time() - start_time   # ---------------------------------------┘

pred = np.argmax(pred, axis=1)
y_eval = np.argmax(y_test, axis=1)

acc_score = metrics.accuracy_score(y_eval, pred)
print('Accuracy =', acc_score*100)

# write files
print('saving model "{}"...'.format(os.path.basename(options.model_path)))
save_result.save_model(options.model_path, model)

print('saving statistics "{}"...'.format(os.path.basename(options.statistics_path)))
save_result.save_statistics(options.statistics_path, entries={
    'aug': options.aug,
    'conv_filters': options.conv[0],
    'conv_kernel_size': options.conv[1],
    'conv_num': options.conv_num,
    'pool': options.pool,
    'stack': options.stack,
    'acc_score(%)': acc_score,
    'preprocess_time': preprocess_time,
    'startup_time': startup_time,
    'train_time': train_time,
    'val_time': val_time,
    'log_path': options.log_path
}, drop_duplicates=False)
