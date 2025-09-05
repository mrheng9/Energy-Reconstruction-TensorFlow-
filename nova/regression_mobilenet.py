from __future__ import absolute_import
import os
import argparse
import datetime
import numpy
import keras
from keras.models import Model
from keras.regularizers import *
from keras.optimizers import *
from keras.layers import *
import keras.backend as K
from keras.callbacks import *
#from coord import CoordinateChannel2D
from .config import MODEL_DIR, LOG_DIR, PLOT_DIR, TENSORBOARD_DIR, DATA_DIR, NUE_DATA_DIR, DATA_FORMAT
import tensorflow as tf


def regression(hparams, input_dims, vertex=False):
    """
    Creates the regression CNN as a compiled Keras model.

    # Arguments
        hparams (dict): dictionary of hyper# Arguments, can be empty
        input_dims (tuple(int)): the dimensionality of the input

    # Returns
        (keras.Model)
    """
    optimizer = hparams.pop('optimizer')
    lr = hparams.pop('lr')
    lrdecay = hparams.pop('lrdecay')
    dropout = hparams.pop('dropout')
    momentum = hparams.pop('momentum')
    filter_number = hparams.pop('filter_number')
    num_layers = hparams.pop('num_layers')
    l2norm = hparams.pop('l2')
    fcl2norm = hparams.pop('fc_l2')
    input_scaling = hparams.pop('input_scaling')

    assert len(hparams) == 0, "Unused parameter left"


    conv_args = lambda: dict(kernel_regularizer=l2(l2norm),
                             bias_regularizer=l2(l2norm),
                             data_format=DATA_FORMAT)
    pool_args = lambda: dict(data_format=DATA_FORMAT)

    img_shape = input_dims + (1,) if DATA_FORMAT=='channels_last' else (1,) + input_dims

    input_x = Input(shape=img_shape)
    input_y = Input(shape=img_shape)

    def bottleneck(x, expansion, stride=1, out_channels=64, res=False):
        expand = out_channels * expansion
        m = Conv2D(expand, (1,1), padding='same')(x)
        m = BatchNormalization()(m)
        m = Activation(tf.nn.relu6)(m)
        m = DepthwiseConv2D((3,3), strides=stride, padding='same')(m)
        m = BatchNormalization()(m)
        m = Activation(tf.nn.relu6)(m)
        m = Conv2D(out_channels, (1,1), padding='same')(m)
        m = BatchNormalization()(m)
        if res:
            m = Add()([m, x])
        return m

    def repetitive_bottleneck(x, n, expansion, stride=1, out_channels=64):
        if n == 1:
            return bottleneck(x, expansion, stride=stride, out_channels=out_channels)
        else:
            m = bottleneck(x, expansion, stride=stride, out_channels=out_channels)
            for _ in range(n - 1):
                m = bottleneck(m, expansion, out_channels=out_channels, res=True)
            return m

    def subnet(x, name):

        x = Conv2D(filter_number, (5,5), padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.relu6)(x)

        x = repetitive_bottleneck(x, 1, expansion=1, stride=1, out_channels=16)
        x = AveragePooling2D(pool_size=2, padding='same')(x)
        x = repetitive_bottleneck(x, 2, expansion=6, stride=1, out_channels=24)

        return x

    if input_scaling != 1.:
        x = Lambda(lambda x_: x_ * input_scaling)(input_x)
        y = Lambda(lambda x_: x_ * input_scaling)(input_y)
    else:
        x = input_x
        y = input_y

    x = subnet(x, name='x')
    y = subnet(y, name='y')

    top = Maximum()([x, y])

    top = AveragePooling2D(pool_size=2, padding='same')(top)
    top = repetitive_bottleneck(top, 3, expansion=6, stride=1, out_channels=32)
    top = AveragePooling2D(pool_size=2, padding='same')(top)
    top = repetitive_bottleneck(top, 4, expansion=6, stride=1, out_channels=48)
    top = repetitive_bottleneck(top, 3, expansion=6, stride=1, out_channels=64)
    top = AveragePooling2D(pool_size=2, padding='same')(top)
    top = repetitive_bottleneck(top, 3, expansion=6, stride=1, out_channels=96)
    top = repetitive_bottleneck(top, 1, expansion=6, stride=1, out_channels=160)

    if not vertex:
        out = GlobalAveragePooling2D()(top)
        out = Dense(1024)(out)
        out = Activation(tf.nn.relu6)(out)
        out = Dropout(dropout)(out)

        out = Dense(1, name='out')(out)

        model = Model([input_x, input_y], out)

    else:
        input_vtx = Input(shape=(3,))
        out = Concatenate(axis=1)([Conv2D(1, (1,1), padding='same', strides=1)(top), input_vtx])
        out = Dense(256, kernel_regularizer=l2(fcl2norm), activation='relu')(out)
        out = Dense(3, kernel_regularizer=l2(fcl2norm))(out)
        model = Model([input_x, input_y, input_vtx], out)

    if optimizer=='adam':
        opt = Adam(lr=lr)
    elif optimizer=='sgd':
        opt = SGD(lr=lr, momentum=momentum, decay=lrdecay)
    else:
        raise("No optimizer found")

    if vertex:
        model.compile(loss='mae',
                      optimizer=opt, metrics=['mean_absolute_error', 'mse'])
    else:
        model.compile(loss='mean_absolute_percentage_error',
                      optimizer=opt, metrics=['mean_absolute_percentage_error'])
    print(model.summary())

    return model

def get_smooth_l1(change_point):
    sq_const = 1./(change_point*2.)
    abs_const = sq_const * (change_point)**2
    def smooth_l1(y_true, y_pred):
        diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                                K.epsilon(),
                                                None))
        bigger = K.cast(K.greater(diff, change_point), 'float32')
        abs_err = (diff - abs_const) * bigger
        sq_err = sq_const*K.square(diff)*(1.-bigger)
        return K.mean(abs_err + sq_err, axis=-1)*100.
    return smooth_l1




def validate(model, name, generator, steps):
    """
    Run validation pipeline.

    # Arguments
        model (keras.model): model to make predictions from
        generator (generator): yields (input, output) pairs
        steps (int): number of times that generator will be called for

    # Returns
        (tuple(numpy.array)) targets and predictions
    """
    y, yhat = [], []
    for i in range(steps):
        print("Batch {}/{}".format(i, steps))
        x_batch, y_batch = next(generator)
        yhat_batch = model.predict(x_batch)
        y.append(y_batch)
        yhat.append(yhat_batch)
    targets, predictions = numpy.concatenate(y), numpy.concatenate(yhat)

    return targets, predictions

def load(name, group):
    return keras.models.load_model(os.path.join(MODEL_DIR, group, name))


def callbacks(name, group='', tensorboard=False, reduce_lr=True, early_stopping=False, model_checkpoint=True):
    """
    Creates callbacks to be used during training.

    # Arguments
        name (str): name for the model that's being trained
        group (str): group of experiments that this model is part of
        tensorboard (bool): whether to run tensorboard or not

    # Returns
        (list) callbacks

    """
    if group:
        for path in [MODEL_DIR, LOG_DIR, TENSORBOARD_DIR]:
            try:
                os.mkdir(os.path.join(path, group))
            except OSError:
                pass

    weights_path = os.path.join(MODEL_DIR, group, name)
    logs_path = os.path.join(LOG_DIR, group, '{}.log'.format(name))
    tensorboard_path = os.path.join(TENSORBOARD_DIR, group, name)

    model_callbacks = []
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1,
    #                                patience=5, verbose=0, mode='auto')
    if model_checkpoint:
        model_saver = ModelCheckpoint(weights_path,
                                      monitor='val_loss',
                                      verbose=0,
                                      save_best_only=True,
                                      mode='auto')
        model_callbacks.append(model_saver)
    if reduce_lr:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4,
                                      min_lr=0.000001, min_delta=0.4)
        model_callbacks.append(reduce_lr)

    if early_stopping:
        early_stopping = EarlyStopping(monitor='loss', min_delta=0.1, patience=5, verbose=1)
        model_callbacks.append(early_stopping)
    # csv_logger = CSVLogger(logs_path)
    # model_callbacks = [early_stopping, model_saver, reduce_lr, csv_logger]

#     if K.backend() == 'tensorflow' and tensorboard:
#         model_callbacks.append(TensorBoard(log_dir=tensorboard_path,
#                                            write_graph=False,
#                                            write_images=False))

    return model_callbacks


def parse_args(presets={}):
    """
    Parse training arguments.

    # Returns:
        l2 (float, default=0.): l2 norm multiplier
        fc_l2 (float, default=0.): l2 norm multiplier for fully connected layers
        dropout (float, default=0.): dropout applied to fully connected layers
        optimizer (str, default='sgd'): 'adam' or 'sgd'
        lr (float, default=0.001): learning rate
        batch_size (int, default=64): training batch size
        filter_number (int, default=32)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2", default=0., type=float)
    parser.add_argument("--fc_l2", default=0., type=float)
    parser.add_argument("--dropout", default=0., type=float)
    parser.add_argument("--optimizer", default='adam', type=str)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--lrdecay", default=0.00001, type=float)
    parser.add_argument("--momentum", default=0.7, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--filter_number", default=32, type=int,
                        help="Number of Conv filters for each layer.")
    parser.add_argument("--mode", default='nue', type=str)
    parser.add_argument("--dataset", default='flat', type=str)
    args = parser.parse_args()
    params = vars(args)
    for key, value in presets.items():
        params[key] = value
    return params


def generate_name(basename, args):
    """
    Generates a unique file name for a model that includes argument settings.

    # Returns
        (str) file name
    """
    name = basename
    for key in args:
        if key == 'continue':
            name += '_{}'.format(key)
        else:
            name += '_{}_{}'.format(key, args.get(key))
    name += '_'
    name += str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return name
