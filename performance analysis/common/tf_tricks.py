import keras
import sys
import importlib
from packaging import version


def is_backend_tf():
    if keras.backend.backend() == 'tensorflow':
        return True
    return False


def _is_tf_v2(tf=None):
    if not tf:
        tf = importlib.import_module('tensorflow')

    return version.parse(tf.__version__) > version.parse('2.0.0a0')


def allow_growth():
    if not is_backend_tf():
        print('backend not support "allow_growth" function.', file=sys.stderr)
        return

    tf = importlib.import_module('tensorflow')

    if _is_tf_v2(tf):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        import keras.backend as k_backend
        k_backend.tensorflow_backend.set_session(tf.Session(config=config))


def is_cudnn(layer):
    if type(layer) == str:
        name = layer
    else:
        assert isinstance(layer, keras.layers.Layer)

        class_name = str(layer.__class__)[8:-2]
        name = class_name.split('.')[-1]

    return name.lower().startswith('CuDNN'.lower())


# HACK: this is for making CuDNN specific-layers to work with legacy project
#   * Keras_LSTM_PTB
#   * Keras_RNN_IMDB
#   TF 2.0+ has merged CuDNN layers into usual RNNs
#   https://keras.io/api/layers/recurrent_layers/lstm/
def import_layer(name):
    if is_cudnn(name) and not is_backend_tf():
        print('backend not support "CuDNN" layers.', file=sys.stderr)
        return None

    tf = importlib.import_module('tensorflow')

    if not is_backend_tf():
        return keras.layers.__dict__[name]

    if _is_tf_v2(tf) and is_cudnn(name):
        return tf.compat.v1.keras.layers.__dict__[name]
    else:
        return tf.keras.layers.__dict__[name]
