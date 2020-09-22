# try re-create Sequential like API for GAN

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.engine import compile_utils


class Metrics_Formatter():
    def __init__(self, gen_loss_name, disc_loss_name,
                 gen_metric_name=None, disc_metric_name=None,
                 num_format='f'):
        self.gen_loss_name = gen_loss_name
        self.disc_loss_name = disc_loss_name
        self.gen_metric_name = gen_metric_name
        self.disc_metric_name = disc_metric_name
        self.num_format = num_format

    def progress_bar_append(self, gen_loss, disc_loss, gen_metric=None, disc_metric=None):
        types = ['gen', 'disc'] * 2
        tags = [self.gen_loss_name, self.disc_loss_name,
                self.gen_metric_name, self.disc_metric_name]
        values = filter(lambda x: x is not None,
                       [gen_loss, disc_loss, gen_metric, disc_metric])
        bar_str = ''

        for type, tag, value in zip(types, tags, values):
            #  - accuracy: 0.3345
            m_str = ' - {{type}}_{{tag}}: {{value:{format}}}'.format(
                format=self.num_format)
            m_str = m_str.format(type=type, tag=tag, value=value)

            bar_str += m_str

        return bar_str

    def log_dict(self, gen_loss, disc_loss, gen_metric=None, disc_metric=None):
        types = ['gen', 'disc'] * 2
        tags = [self.gen_loss_name, self.disc_loss_name,
                self.gen_metric_name, self.disc_metric_name]
        values = filter(lambda x: x is not None,
                       [gen_loss, disc_loss, gen_metric, disc_metric])
        logs = {}

        for type, tag, value in zip(types, tags, values):
            key = f'{type}_{tag}'
            logs[key] = value

        return logs


class GAN():

    def __init__(self):
        self.gen_model = Sequential()
        self.disc_model = Sequential()

        self.gen_optimizer = None  # Optimizer
        self.disc_optimizer = None  # Optimizer
        self.gen_losses_container = None  # LossesContainer
        self.disc_losses_container = None  # LossesContainer
        self.gen_metrics_container = None  # MetricsContainer
        self.disc_metrics_container = None  # MetricsContainer

    def add_gen(self, layer):
        self.gen_model.add(layer)

    def add_disc(self, layer):
        self.disc_model.add(layer)

    def compile(self,
                optimizer_gen='rmsprop', optimizer_disc='rmsprop',
                loss_gen=None, loss_disc=None,
                metrics_gen=None, metrics_disc=None):

        self.gen_optimizer = optimizers.get(optimizer_gen)
        self.disc_optimizer = optimizers.get(optimizer_disc)

        self.gen_losses_container = compile_utils.LossesContainer(loss_gen) if loss_gen else None
        self.disc_losses_container = compile_utils.LossesContainer(loss_disc) if loss_disc else None

        self.gen_metrics_container = compile_utils.MetricsContainer(metrics_gen) if metrics_gen else None
        self.disc_metrics_container = compile_utils.MetricsContainer(metrics_disc) if metrics_disc else None

        self.m_formatter = Metrics_Formatter(
            gen_loss_name=_get_tag_name(self.gen_losses_container),
            disc_loss_name=_get_tag_name(self.disc_losses_container),
            gen_metric_name=_get_tag_name(self.gen_metrics_container),
            disc_metric_name=_get_tag_name(self.disc_metrics_container),
            num_format='.03f'
        )

    # TODO: add support for batch_size
    def train(self, dataset, epochs=1, callbacks=[]):
        self.callbacks = callbacks
        self._on_train_begin()  # ==============================================

        @tf.function
        def _train_step(real_data):
            # noise_dim = (batch_size, input_shape[1], input_shape[2]...)
            noise_dim = (real_data.shape[0], *self.gen_model.input_shape[1:])
            noise = tf.random.normal(noise_dim)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                fake_data = self.gen_model(noise, training=True)

                real_output = self.disc_model(real_data, training=True)
                fake_output = self.disc_model(fake_data, training=True)

                gen_loss = _generator_loss(
                    self.gen_losses_container, fake_output)
                disc_loss = _discriminator_loss(
                    self.disc_losses_container, real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.gen_model.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.disc_model.trainable_variables)

            self.gen_optimizer.apply_gradients(
                zip(gradients_of_generator, self.gen_model.trainable_variables))
            self.disc_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.disc_model.trainable_variables))

            step_metrics = {'gen_loss': gen_loss, 'disc_loss': disc_loss}

            if self.gen_metrics_container:
                gen_metric = _generator_metric(
                    self.gen_metrics_container, fake_output)
                step_metrics['gen_metric'] = gen_metric

            if self.disc_metrics_container:
                disc_metric = _discriminator_metric(
                    self.disc_metrics_container, real_output, fake_output)
                step_metrics['disc_metric'] = disc_metric

            return step_metrics

        for epoch in range(epochs):
            # TODO: replace this with ProgbarLogger(Callback)
            step_cnt = len(list(dataset.as_numpy_iterator())) if isinstance(dataset, tf.data.Dataset) else len(dataset)
            progress_bar = tf.keras.utils.Progbar(step_cnt)
            print(f'\nEpoch {epoch+1}/{epochs}')

            self._on_epoch_begin(epoch)  # -------------------------------------

            for real_data in dataset:
                step_metrics = _train_step(real_data)
                progress_bar.add(1)
                print(self.m_formatter.progress_bar_append(**step_metrics), end='')

            self._on_epoch_end(epoch, step_metrics)  # -------------------------

        self._on_train_end(step_metrics)  # ====================================
        print()

    def _on_train_begin(self, logs={}):
        _broadcast_callbacks(self.callbacks, 'on_train_begin', logs=logs)

    def _on_train_end(self, step_metrics):
        logs = self.m_formatter.log_dict(**step_metrics)
        _broadcast_callbacks(self.callbacks, 'on_train_end', logs=logs)

    def _on_epoch_begin(self, epoch, logs={}):
        _broadcast_callbacks(self.callbacks, 'on_epoch_begin', epoch=epoch, logs=logs)

    def _on_epoch_end(self, epoch, step_metrics):
        logs = self.m_formatter.log_dict(**step_metrics)
        _broadcast_callbacks(self.callbacks, 'on_epoch_end', epoch=epoch, logs=logs)


def _broadcast_callbacks(callbacks, fn, **kwargs):
    for callback in callbacks:
        getattr(callback, fn)(**kwargs)


def _reset_states(container):
    if not isinstance(container, compile_utils.Container):
        return
    for metric in container.metrics:
        metric.reset_states()


def _generator_loss(loss_container, fake_output):
    loss = loss_container(tf.ones(fake_output.shape), fake_output)
    _reset_states(loss_container)
    return loss


def _discriminator_loss(loss_container, real_output, fake_output):
    real_loss = loss_container(tf.ones(real_output.shape), real_output)
    _reset_states(loss_container)
    fake_loss = loss_container(tf.zeros(fake_output.shape), fake_output)
    _reset_states(loss_container)
    total_loss = real_loss + fake_loss
    return total_loss


def _generator_metric(metric_container, fake_output):
    metric_container.update_state(tf.ones(fake_output.shape), fake_output)
    metric = metric_container.metrics[0].result()
    _reset_states(metric_container)
    return metric


def _discriminator_metric(metric_container, real_output, fake_output):
    metric_container.update_state(tf.ones(real_output.shape), real_output)
    real_metric = metric_container.metrics[0].result()
    _reset_states(metric_container)
    metric_container.update_state(tf.zeros(fake_output.shape), fake_output)
    fake_metric = metric_container.metrics[0].result()
    _reset_states(metric_container)
    metric = (real_metric + fake_metric) / 2
    return metric


def _get_tag_name(container):
    assert isinstance(container, compile_utils.Container)

    if isinstance(container, compile_utils.LossesContainer):
        user_input = container._user_losses
    elif isinstance(container, compile_utils.MetricsContainer):
        user_input = container._user_metrics
    else:
        raise AssertionError('should be LossesContainer or MetricsContainer')

    if type(user_input) == str:
        return user_input
    else:
        return user_input.name
