import tensorflow as tf
import functools

# ==============================================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# ==============================================================================

model = tf.keras.models.load_model('model.h5')
model.summary()


def trainable_variables(model):
    shapes = [variable.shape.as_list() for variable in model.trainable_variables]
    trainables = map(lambda layer: functools.reduce(lambda x, y: x*y, layer), shapes)
    return sum(trainables)


trainable_variables(model)


def trainable_variables_cnt(trainable_variables):
    if isinstance(trainable_variables, tf.keras.models.Model) or isinstance(trainable_variables, tf.python.keras.engine.base_layer.Layer):
        trainable_variables = trainable_variables.trainable_variables

    if not type(trainable_variables) == list:
        trainable_variables = [trainable_variables]

    shapes = [variable.shape.as_list() for variable in trainable_variables]
    trainables = map(lambda layer: functools.reduce(lambda x, y: x*y, layer), shapes)
    return sum(trainables)


trainable_variables_cnt(model)
trainable_variables_cnt(model.trainable_variables)
trainable_variables_cnt(model.layers[0])
trainable_variables_cnt(model.layers[0].trainable_variables)
trainable_variables_cnt(model.layers[5])

dir(model.layers[0])
dir(model.layers[0].__class__)
model.layers[0].__class__.__name__


def trainable_variables_cat(model):
    result = {'Conv': 0, 'RNN': 0, 'Other': 0}

    for layer in model.layers:
        if isinstance(layer, tf.python.keras.layers.convolutional.Conv):
            result['Conv'] += trainable_variables_cnt(layer)
        elif isinstance(layer, tf.python.keras.layers.recurrent.RNN):
            result['RNN'] += trainable_variables_cnt(layer)
        else:
            result['Other'] += trainable_variables_cnt(layer)

    return result


trainable_variables_cat(model)
