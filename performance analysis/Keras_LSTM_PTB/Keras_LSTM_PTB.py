import os
import sys
import time
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Activation
from sklearn import metrics
import cmdargv
import save_result
import dataset
from KerasBatchGenerator import KerasBatchGenerator
from CustomLogger import CustomLogger

# options
num_epochs = 50
options = cmdargv.parse_argv(sys.argv)

# preprocess
start_time = time.time()    # -------------------------------------------------┐
# load PTB dataset
train_data, valid_data, test_data, labelEnc = dataset.load_ptb()

num_vocabulary = len(labelEnc.classes_)
train_data_generator = KerasBatchGenerator(train_data, options.num_steps, options.batch_size,
                                           num_vocabulary, skip_step=options.num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, options.num_steps, options.batch_size,
                                           num_vocabulary, skip_step=options.num_steps)
preprocess_time = time.time() - start_time   # --------------------------------┘

start_time = time.time()    # -------------------------------------------------┐
model = Sequential()
model.add(Embedding(num_vocabulary, options.embedding_size, input_length=options.num_steps))
if options.cudnn:
    model.add(CuDNNLSTM(options.embedding_size, return_sequences=True))
    model.add(CuDNNLSTM(options.embedding_size, return_sequences=True))
else:
    model.add(LSTM(options.embedding_size, return_sequences=True))
    model.add(LSTM(options.embedding_size, return_sequences=True))
model.add(Dropout(0.25))
model.add(TimeDistributed(Dense(num_vocabulary)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
startup_time = time.time() - start_time   # -----------------------------------┘

if options.train:
    custom_logger = CustomLogger(options.log_path)

    start_time = time.time()    # ---------------------------------------------┐
    result = model.fit_generator(
        train_data_generator, len(train_data) // (options.batch_size * options.num_steps), num_epochs,
        validation_data=valid_data_generator,
        validation_steps=len(valid_data) // (options.batch_size * options.num_steps),
        callbacks=[custom_logger])
    score = model.evaluate_generator(valid_data_generator,
        steps=len(valid_data) // (options.batch_size * options.num_steps))
    train_time = time.time() - start_time   # ---------------------------------┘
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('custom_logger.train_begin = {}'.format(custom_logger.train_begin))
    print('custom_logger.train_end = {}'.format(custom_logger.train_end))
    print('custom_logger.train_time = {}'.format(custom_logger.train_time))
    print('custom_logger.result = {}'.format(custom_logger.result))

    # measure accuracy
    start_time = time.time()    # ---------------------------------------------┐
    def valid_data_transform():
        steps = len(valid_data) // (options.batch_size * options.num_steps)
        x = np.zeros((steps, options.batch_size, options.num_steps))
        y = np.zeros((steps, options.batch_size, options.num_steps, num_vocabulary))

        for index in range(steps):
            step_x, step_y = next(valid_data_generator)
            x[index, :, :] = step_x
            y[index, :, :, :] = step_y

        return x, y

    x_eval, y_eval = valid_data_transform()

    pred = model.predict_generator(valid_data_generator,
        steps=len(valid_data) // (options.batch_size * options.num_steps))
    val_time = time.time() - start_time   # -----------------------------------┘

    acc_score = metrics.accuracy_score(y_eval, pred)
    print('Accuracy =', acc_score*100)

    # write files
    print('saving model "{}"...'.format(os.path.basename(options.model_path)))
    save_result.save_model(options.model_path, model)

    print('saving statistics "{}"...'.format(os.path.basename(options.statistics_path)))
    save_result.save_statistics(options.statistics_path, entries={
        'num_steps': options.num_steps,
        'batch_size': options.batch_size,
        'embedding_size': options.embedding_size,
        'acc_score(%)': acc_score,
        'preprocess_time': preprocess_time,
        'startup_time': startup_time,
        'train_time': train_time,
        'val_time': val_time,
        'log_path': options.log_path
    }, drop_duplicates=False)

if options.eval:
    model = load_model(options.model_path)

    def predict_with_data(data, num_predict=10, dummy_iters=40):
        generator = KerasBatchGenerator(data, options.num_steps, 1, num_vocabulary, skip_step=1)
        actual_words = []
        predicted_words = []

        for index in range(dummy_iters):
            next(generator)

        for index in range(num_predict):
            actual_words.append(data[dummy_iters + options.num_steps + index])

            input, _ = next(generator)
            prediction = model.predict(input)
            prediction = np.argmax(prediction[:, options.num_steps-1, :])
            predicted_words.append(prediction)

        return actual_words, predicted_words

    actual_words, predicted_words = predict_with_data(train_data)
    actual_words = labelEnc.inverse_transform(actual_words)
    predicted_words = labelEnc.inverse_transform(predicted_words)
    print("Training data:")
    print('  Actual words: ' + ' '.join(actual_words))
    print('  Predicted words: ' + ' '.join(predicted_words))

    actual_words, predicted_words = predict_with_data(test_data)
    actual_words = labelEnc.inverse_transform(actual_words)
    predicted_words = labelEnc.inverse_transform(predicted_words)
    print("Training data:")
    print('  Actual words: ' + ' '.join(actual_words))
    print('  Predicted words: ' + ' '.join(predicted_words))