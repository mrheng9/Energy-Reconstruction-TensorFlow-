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
    pooling = hparams.pop('pooling')
    
    if pooling == 'average':
        pool = lambda name, strides: AveragePooling2D(pool_size=(3, 3), strides=strides, padding='same', name=name, **pool_args())
    else:
        pool = lambda name, strides: MaxPooling2D(pool_size=(3, 3), strides=strides, padding='same', name=name, **pool_args())

    assert len(hparams) == 0, "Unused parameter left"
    
    
    conv_args = lambda: dict(kernel_regularizer=l2(l2norm),
                             bias_regularizer=l2(l2norm),
                             data_format=DATA_FORMAT)
    pool_args = lambda: dict(data_format=DATA_FORMAT)

    img_shape = input_dims + (1,) if DATA_FORMAT=='channels_last' else (1,) + input_dims
    
    input_x = Input(shape=img_shape)
    input_y = Input(shape=img_shape)

    def inception(x, name="inception"):
        branch11_x = Conv2D(filter_number, (1, 1), activation='relu', padding='same', name='branch11_{}'.format(name), **conv_args())(x)
        branch11_x = Conv2D(filter_number, (5, 5), activation='relu', padding='same', name='branch12_{}'.format(name), **conv_args())(branch11_x)
        branch12_x = Conv2D(filter_number, (1, 1), activation='relu', padding='same', name='branch21_{}'.format(name), **conv_args())(x)
        branch12_x = Conv2D(filter_number, (3, 3), activation='relu', padding='same', name='branch22_{}'.format(name), **conv_args())(branch12_x)
        branch13_x = pool(strides=(1,1), name='branch3mp_{}'.format(name))(x)
        branch13_x = Conv2D(filter_number, (1, 1), padding='same', activation='relu', name='branch31_{}'.format(name), **conv_args())(branch13_x)
        branch14_x = Conv2D(filter_number, (1, 1), padding='same', activation='relu', name='branch32_{}'.format(name), **conv_args())(x)

        x = Concatenate(axis=-1 if DATA_FORMAT=='channels_last' else 1, name='Concatenate_{}'.format(name))([branch11_x, branch12_x, branch13_x, branch14_x])
        return x

    def subnet(x, name):
        def input(x, name='Input'):
            x = Conv2D(filter_number, (7, 7), activation='relu', strides=(2, 2), name=name + 'Conv1', **conv_args())(x)
            x = pool(strides=(2, 2), name=name + 'Pool1')(x)
            x = Conv2D(filter_number, (1, 1), activation='relu', name=name + 'Conv2', **conv_args())(x)
            x = Conv2D(filter_number, (3, 3), activation='relu', name=name + 'Conv3', **conv_args())(x)
            x = pool(strides=(2, 2), name=name + 'Pool2')(x)
            return x

        x = input(x, name=name + 'Input')
        for k in range(num_layers):
            x = inception(x, name=name + 'Inception' + str(k))
        x = pool(strides=(2, 2), name=name + 'MaxPool')(x)
        return x


    if input_scaling != 1.:
        x = Lambda(lambda x_: x_ * input_scaling)(input_x)
        y = Lambda(lambda x_: x_ * input_scaling)(input_y)
    else:
        x = input_x
        y = input_y
   
    x = subnet(x, name='x')
    y = subnet(y, name='y')

    top = Concatenate(axis=-1 if DATA_FORMAT=='channels_last' else 1)([x, y])

    top = inception(top, name='topInception')
    top = AveragePooling2D(pool_size=(3,3))(top)

    top = Flatten()(top)
    
    if not vertex:
#         input_vtx = Input(shape=(2,))
#         out = Concatenate(axis=1)([top, input_vtx])
#         out = Dense(1, kernel_regularizer=l2(fcl2norm))(out)
        out = Dense(256, kernel_regularizer=l2(fcl2norm), activation='relu')(top)
        out = Dense(1, kernel_regularizer=l2(fcl2norm))(out)
        if dropout:
            out = Dropout(dropout)(out)
        model = Model([input_x, input_y], out)
        
    else:
        input_vtx = Input(shape=(3,))
        out = Concatenate(axis=1)([top, input_vtx])
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
