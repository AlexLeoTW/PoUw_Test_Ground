import sys
import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

figsize = [10, 10]
seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print('seed = {}'.format(seed))
print('x_train.shape = {}'.format(x_train.shape))
print('y_train.shape = {}'.format(y_train.shape))
print('x_test.shape = {}'.format(x_test.shape))
print('y_test.shape = {}'.format(y_test.shape))

train_datagen = ImageDataGenerator(
    # featurewise_center=False,  # set input mean to 0 over the dataset (Default: False)
    # samplewise_center=False,  # set each sample mean to 0 (Default: False)
    # featurewise_std_normalization=False,  # divide inputs by std of the dataset (Default: False)
    # samplewise_std_normalization=False,  # divide each input by its std (Default: False)
    # zca_whitening=False,  # apply ZCA whitening (Default: False)
    # zca_epsilon=1e-06,  # epsilon for ZCA whitening (Default: 1e-06)
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180) (Default: 0.0)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width) (Default: 0.0)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height) (Default: 0.0)
    brightness_range=[0.6, 1.4],  # Tuple or list of two floats. Range for picking a brightness shift value from. (Default: None)
    shear_range=0.1,  # set range for random shear (Default: 0.0)
    zoom_range=0.2,  # set range for random zoom (Default: 0.0)
    channel_shift_range=10,  # set range for random channel shifts (Default: 0.0)
    fill_mode='nearest',  # set mode for filling points outside the input boundaries (Default: 'nearest')
    # cval=0.0,  # value used for fill_mode = "constant" (Default: 0.0)
    horizontal_flip=True,  # randomly flip images (Default: False)
    # vertical_flip=False,  # randomly flip images (Default: False)
    # rescale=None,  # set rescaling factor (applied before any other transformation) (Default: None)
    # preprocessing_function=None,  # set function that will be applied on each input (Default: None)
    data_format='channels_last',  # image data format, either "channels_first" or "channels_last" (Default: 'channels_last')
    # validation_split=0.0  # fraction of images reserved for validation (strictly between 0 and 1) (Default: 0.0)
)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
train_datagen.fit(x_train, seed=seed)

train_datagen = train_datagen.flow(x_train, y_train, batch_size=25)
x_batch, y_batch = next(train_datagen)

plt.figure(figsize=figsize)

for num in range(25):
    plt.subplot(5, 5, num + 1)
    plt.imshow(x_batch[num].astype(np.uint8))

plt.show()
