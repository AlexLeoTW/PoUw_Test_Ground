import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
import os
import save_result
import tf_tricks
import cmdargv
from GAN import GAN
from SaveFig import SaveFig

# Tensorflow GPU Trick
tf_tricks.allow_growth()

# pre-defined consstants
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50

# read commandline arguments
options = cmdargv.parse_argv()

# preprocess
start_time = time.time()    # -------------------------------------------------┐
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(train_images, train_labels) = np.append(x_train, x_test, axis=0), np.append(y_train, y_test)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalize the images to [-1, 1]

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
preprocess_time = time.time() - start_time   # --------------------------------┘

# startup
start_time = time.time()    # -------------------------------------------------┐
gan = GAN()

gan.add_gen(Dense(7 * 7 * 256, input_shape=(options.noise_dim,)))
gan.add_gen(BatchNormalization())
gan.add_gen(LeakyReLU())
gan.add_gen(Reshape((7, 7, 256)))
# output_shape = (None, 7, 7, 256)

gan.add_gen(Conv2DTranspose(options.gen_l1, kernel_size=5, strides=1, padding='same'))
gan.add_gen(BatchNormalization())
gan.add_gen(LeakyReLU())
# output_shape = (None, 7, 7, 128)

gan.add_gen(Conv2DTranspose(options.gen_l2, kernel_size=5, strides=2, padding='same'))
gan.add_gen(BatchNormalization())
gan.add_gen(LeakyReLU())
# output_shape = (None, 14, 14, 64)

gan.add_gen(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh'))
# output_shape = (None, 28, 28, 1)

gan.add_disc(Conv2D(options.disc_l1, kernel_size=5, strides=2, padding='same',
             input_shape=[28, 28, 1]))
gan.add_disc(LeakyReLU())
gan.add_disc(Dropout(0.3))

gan.add_disc(Conv2D(options.disc_l2, kernel_size=5, strides=2, padding='same'))
gan.add_disc(LeakyReLU())
gan.add_disc(Dropout(0.3))

gan.add_disc(Flatten())
gan.add_disc(Dense(1))

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

gan.compile(optimizer_gen=Adam(1e-4), optimizer_disc=Adam(1e-4),
            loss_gen=cross_entropy, loss_disc=cross_entropy,
            metrics_gen='binary_accuracy', metrics_disc='binary_accuracy')
startup_time = time.time() - start_time   # -----------------------------------┘

# train
start_time = time.time()    # -------------------------------------------------┐
gan.train(train_dataset, EPOCHS,
          callbacks=[SaveFig(model=gan.gen_model, path=options.img_dir)])
train_time = time.time() - start_time   # -------------------------------------┘

print('saving generator model "{}"...'.format(os.path.basename(options.gen_model_path)))
save_result.save_model(options.gen_model_path, gan.gen_model)

print('saving generator model "{}"...'.format(os.path.basename(options.disc_model_path)))
save_result.save_model(options.disc_model_path, gan.disc_model)

print('saving statistics "{}"...'.format(os.path.basename(options.statistics_path)))
save_result.save_statistics(options.statistics_path, entries={
    'noise_dim': options.noise_dim,
    'gen_l1': options.gen_l1,
    'gen_l2': options.gen_l2,
    'disc_l1': options.disc_l1,
    'disc_l2': options.disc_l2,
    'preprocess_time': preprocess_time,
    'startup_time': startup_time,
    'train_time': train_time,
    'log_path': os.basename(options.log_path)
}, drop_duplicates=False)
