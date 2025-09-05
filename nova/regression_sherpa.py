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
from .config import MODEL_DIR, LOG_DIR, PLOT_DIR, TENSORBOARD_DIR, DATA_DIR, NUE_DATA_DIR, DATA_FORMAT


def regression(hparams, input_dims, vertex=False):
    """
    Creates the regression CNN as a compiled Keras model.

    # Arguments
        outdim (int): the dimensionality of the regression output
        hparams (dict): dictionary of hyper# Arguments, can be empty
        input_dims (tuple(int)): the dimensionality of the input

    # Returns
        (keras.Model)
    """
    optimizer = hparams.pop('optimizer')
    lr = hparams.pop('lr', None)
    lrdecay = hparams.pop('lrdecay', None)
    momentum = hparams.pop('momentum', None)
    filter_number = hparams.pop('filter_number')
    num_blocks = hparams.pop('num_blocks')
    num_top_blocks = hparams.pop('num_top_blocks')
    num_dense = hparams.pop('num_dense')
    l2norm = hparams.pop('l2')
    fcl2norm = hparams.pop('fc_l2')
    kernel_init = hparams.pop('kernel_init')
    skipconnection = (hparams.pop('skipconnection') == 'yes')
    activation = hparams.pop('activation')
    activation_function = {'prelu': PReLU, 'elu': ELU, 'leakyrelu': LeakyReLU, 'relu': lambda: Activation('relu')}[activation]
    pooling = hparams.pop('pooling')
    locallyconnected = (hparams.pop('locallyconnected', 'no') == 'yes')
    input_scaling = hparams.pop('input_scaling')
    loss = hparams.pop('loss', 'mean_absolute_percentage_error')
    
#     assert len(hparams) == 0, "Unused parameter left: {}".format(hparams)
    
    
    conv_args = lambda: dict(kernel_regularizer=l2(l2norm),
                             bias_regularizer=l2(l2norm),
                             data_format=DATA_FORMAT,
                             kernel_initializer=kernel_init)
    pool_args = lambda: dict(data_format=DATA_FORMAT)

    img_shape = input_dims + (1,) if DATA_FORMAT=='channels_last' else (1,) + input_dims
    
    input_x = Input(shape=img_shape)
    input_y = Input(shape=img_shape)
    
    def convblock(x, num_filters, level, block):
        if locallyconnected and level==1 and block==0:
            y = LocallyConnected2D(num_filters, (3, 3), padding='valid', **conv_args())(x)
        else:
            y = Conv2D(num_filters, (3, 3), padding='same', **conv_args())(x)
        y = activation_function()(y)
        y = Conv2D(num_filters, (3, 3), padding='same', **conv_args())(y)
        if skipconnection:
            y = keras.layers.Add()([x, y])
        y = activation_function()(y)
        return y

    def subnet(x, name):
        if pooling == 'average':
            pool = lambda: AveragePooling2D(pool_size=(2, 2))
        elif pooling == 'fullyconv':
            pool = lambda: Conv2D(filter_number, (3, 3), padding='valid', strides=(2,2), **conv_args())
        else:
            pool = lambda: MaxPooling2D(pool_size=(2, 2))
        
        for j in range(num_blocks):
            x = convblock(x, filter_number, 1, j)
        x = pool()(x)
        
        for j in range(num_blocks):
            x = convblock(x, filter_number, 2, j)
        x = pool()(x)
        
        for j in range(num_blocks):
            x = convblock(x, filter_number, 3, j)
        x = pool()(x)
        return x
    
    x = Lambda(lambda x_: x_ * input_scaling)(input_x)
    y = Lambda(lambda x_: x_ * input_scaling)(input_y)

    x = subnet(x, name='x')
    y = subnet(y, name='y')

    top = Concatenate(axis=-1 if DATA_FORMAT=='channels_last' else 1)([x, y])

    for j in range(num_top_blocks):
        top = convblock(top, filter_number*2, 4, j)

    top = AveragePooling2D(pool_size=(4, 4))(top)

    out = Flatten()(top)
    if vertex:
        input_vtx = Input(shape=(2,))
        out = Concatenate(axis=1)([out, input_vtx])
        model_inputs = [input_x, input_y, input_vtx]
    else:
        model_inputs = [input_x, input_y]
        
    for _ in range(num_dense):
        out = Dense(512, kernel_regularizer=l2(fcl2norm), activation='relu')(out)

    out = Dense(1, kernel_regularizer=l2(fcl2norm))(out)

    model = Model(model_inputs, out)

    if optimizer=='adam':
        opt = Adam(lr=lr)
    elif optimizer=='sgd':
        opt = SGD(lr=lr, momentum=momentum, decay=lrdecay)
    else:
        raise("No optimizer found")

    model.compile(loss=get_loss(loss),
                  optimizer=opt, metrics=['mean_absolute_percentage_error'])
    print(model.summary())

    return model

def get_loss(loss):
    if loss.startswith('double'):
        change_point = float(loss.split(',')[1])
        return double_loss(change_point)
    elif loss.startswith('triple'):
        change_point = float(loss.split(',')[1])
        change_point_2 = float(loss.split(',')[2])
        return triple_loss(change_point, change_point_2)
    elif loss.startswith('sqrt'):
        change_point = float(loss.split(',')[1])
        factor = float(loss.split(',')[2])
        return sqrt_loss(change_point, factor)
    elif loss == 'log_resolution':
        return log_resolution
    elif loss == 'jianming_loss':
        return jianming_loss
    elif loss == 'mean_absolute_scaled_error':
        return mean_absolute_scaled_error
    elif loss == 'mean_squared_scaled_error':
        return mean_squared_scaled_error
    else:
        return loss

def mean_absolute_scaled_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return K.mean(diff, axis=-1)

def mean_squared_scaled_error(y_true, y_pred):
    diff = K.square((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return K.mean(diff, axis=-1)

def double_loss(change_point):
    sq_const = 1./(2.*change_point)
    abs_const = change_point/2.
    def loss(y_true, y_pred):
        diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                                K.epsilon(),
                                                None))
        bigger = K.cast(K.greater(diff, change_point), 'float32')
        abs_err = (diff - abs_const) * bigger
        sq_err = sq_const*K.square(diff)*(1.-bigger)
        return K.mean(abs_err + sq_err, axis=-1)
    return loss

def triple_loss(change_point, change_point_2):
    sq_const = 1./(2.*change_point)
    abs_const = change_point/2.
    sqrt_factor = 2*np.sqrt(change_point_2)
    sqrt_offset = change_point_2 + abs_const
    def loss(y_true, y_pred):
        diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                                K.epsilon(),
                                                None))
        
        in_zone_1 = K.cast(K.greater(diff, change_point), 'float32')
        in_zone_2 = K.cast(K.greater(diff, change_point_2), 'float32')
        
        sq_err = sq_const*K.square(diff)*(1.-in_zone_1)*(1.-in_zone_2)
        
        abs_err = (diff - abs_const) * in_zone_1 * (1.-in_zone_2)
        
        sqrt_err = (K.sqrt(diff) * sqrt_factor - sqrt_offset) * in_zone_2
        
        return K.mean(abs_err + sq_err + sqrt_err, axis=-1)
    return loss

def sqrt_loss(change_point, factor):
    sq_factor = factor
    sqrt_factor = 4.*sq_factor*change_point**(3/2)
    sqrt_offset = -3.*sq_factor*change_point**2
    print(sq_factor, sqrt_factor, sqrt_offset)
    def loss(y_true, y_pred):
        diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                                K.epsilon(),
                                                None))
        sqrt_zone = K.cast(K.greater(diff, change_point), 'float32')
        sqrt_err = (K.sqrt(K.clip(diff, change_point, None))*sqrt_factor + sqrt_offset) * sqrt_zone
        sq_err = sq_factor*K.square(diff) * (1.-sqrt_zone)
        return K.mean(sq_err + sqrt_err, axis=-1)
    return loss

def log_resolution(y_true, y_pred):
    resolution = K.clip(y_pred, K.epsilon(), None) / K.clip(K.abs(y_true), K.epsilon(), None)
    return K.mean(-1. * K.abs(K.log(resolution)), axis=-1)

def jianming_loss(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return K.mean(diff * K.sqrt(y_true), axis=-1)


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


def callbacks(name, group='', tensorboard=False, reduce_lr=True, early_stopping=False, model_checkpoint=True, lr_scheduler=False):
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
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                                      min_lr=0.000001, min_delta=0.4)
        model_callbacks.append(reduce_lr)
        
    if lr_scheduler:
        def schedule(epoch, lr):
            if epoch < 15:
                return 1e-3
            elif epoch < 30:
                return 1e-4
            else:
                return 1e-5
            
        lr_scheduler = LearningRateScheduler(schedule, verbose=0)
        model_callbacks.append(lr_scheduler)
        
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
    parser.add_argument("--optimizer", default='adam', type=str)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--lrdecay", default=0.00001, type=float)
    parser.add_argument("--momentum", default=0.7, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
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