'''Loosely Based on IMDB LSTM by keras-team
https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
'''

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.datasets import imdb
import time
import os
from sklearn import metrics
import cmdargv
from CustomLogger import CustomLogger
import save_result

maxlen = 80
batch_size = 32
num_epochs = 50

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
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=options.max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

preprocess_time = time.time() - start_time   # --------------------------------┘

# load whhatever options.type says (checked in cmdargv)
K_RNN = __import__('keras.layers', fromlist=[options.type])
K_RNN = K_RNN.__dict__[options.type]

start_time = time.time()    # -------------------------------------------------┐
model = Sequential()
model.add(Embedding(input_dim=options.max_features, output_dim=options.embd_size))
model.add(K_RNN(units=options.embd_size))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

startup_time = time.time() - start_time   # -----------------------------------┘

model.summary()
custom_logger = CustomLogger(options.log_path)

start_time = time.time()    # -------------------------------------------------┐
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_data=(x_test, y_test),
          callbacks=[custom_logger])

train_time = time.time() - start_time   # -------------------------------------┘

score, acc = model.evaluate(x_test, y_test)
print('Test score:', score)
print('Test accuracy:', acc)

start_time = time.time()    # -------------------------------------------------┐
pred = model.predict(x_test)
val_time = time.time() - start_time   # ---------------------------------------┘

pred = pred.reshape(len(pred))
pred = (pred > 0.5).astype('int')
print(pred[:10])
print(y_test[:10])

acc_score = metrics.accuracy_score(y_test, pred)
print('Accuracy =', acc_score*100)

# write files
print('saving model "{}"...'.format(os.path.basename(options.model_path)))
save_result.save_model(options.model_path, model)

print('saving statistics "{}"...'.format(os.path.basename(options.statistics_path)))
save_result.save_statistics(options.statistics_path, entries={
    'max_features': options.max_features,
    'embd_size': options.embd_size,
    'type': options.type,
    'acc_score(%)': acc_score,
    'preprocess_time': preprocess_time,
    'startup_time': startup_time,
    'train_time': train_time,
    'val_time': val_time,
    'log_path': options.log_path
}, drop_duplicates=False)
