import sys
import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180) (Default: 0.0)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width) (Default: 0.0)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height) (Default: 0.0)
    brightness_range=[0.6, 1.4],  # Tuple or list of two floats. Range for picking a brightness shift value from. (Default: None)
    shear_range=0.5,  # set range for random shear (Default: 0.0)
    zoom_range=0.2,  # set range for random zoom (Default: 0.0)
    channel_shift_range=10,  # set range for random channel shifts (Default: 0.0)
    fill_mode='constant',  # set mode for filling points outside the input boundaries (Default: 'nearest')
    horizontal_flip=True,  # randomly flip images (Default: False)
    data_format='channels_last',  # image data format, either "channels_first" or "channels_last" (Default: 'channels_last')
)
# train_datagen = ImageDataGenerator()

train_datagen.fit(x_train, seed=seed)

train_datagen = train_datagen.flow(x_train, y_train, batch_size=16)
x_batch, y_batch = next(train_datagen)

for num in range(16):
    plt.subplot(4, 4, num + 1)
    plt.imshow(x_batch[num].astype(np.uint8))

plt.show()
