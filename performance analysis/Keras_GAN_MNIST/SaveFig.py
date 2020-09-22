import tensorflow as tf
import matplotlib.pyplot as plt
import functools
import os


class SaveFig(tf.keras.callbacks.Callback):
    def __init__(self, model, path='.', plot_size=(4, 4)):
        self.model = model

        self.path = path if path.endswith('/') else f'{path}/'
        _mkdir_if_not_exist(self.path)

        self.plot_size = plot_size

        num_fig = functools.reduce(lambda x, y: x*y, plot_size)
        self.noise = tf.random.normal([num_fig, *self.model.input_shape[1:]])

    def on_epoch_end(self, epoch, logs={}):
        self.generate_and_save_images(epoch+1)

    def generate_and_save_images(self, epoch):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.model(self.noise, training=False)

        plt.figure(figsize=self.plot_size)
        fig, axs = plt.subplots(*self.plot_size)

        for ax, pred in zip(axs.flatten(), predictions):
            ax.imshow(pred * 127.5 + 127.5, cmap='gray')
            ax.axis('off')

        plt.savefig(f'{self.path}image_at_epoch_{epoch:04d}.png')
        plt.close()


def _mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.mkdir(path)
