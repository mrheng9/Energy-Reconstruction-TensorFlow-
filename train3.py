from __future__ import print_function
import os
import pickle

if True:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    CONFIG = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
    CONFIG.gpu_options.allow_growth = True  # Prevents tf from grabbing all gpu memory.
    # sess = tf.Session(config=CONFIG)
    sess = tf.Session(config=CONFIG)
    from keras import backend as K

    K.set_session(sess)
else:
    print("Using Theano")
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda0,gpuarray.preallocate=1'
import argparse
import random
import h5py
import nova
import nova.regression
import nova.regression_sherpa
import nova.regression_mobilenet
import keras
import time
import datetime
from keras.models import load_model
import keras.backend as K
from config import *
import multiprocessing

# Parse command line arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='nue', type=str, help="nue or electron")
parser.add_argument("--name", type=str, help="a descriptive name")
parser.add_argument("--path", type=str, help="the data path")
parser.add_argument("--pooling", type=str, default='average', help="the type of pooling to use everywhere")
parser.add_argument("--input_scaling", type=float, default=1.0, help="constant to multiply inputs by")
parser.add_argument("--arch", default='resnet', type=str,
                    help="architecture to train (resnet, googlenet, or mobilenet)")
parser.add_argument("--weighted", default=False, action="store_true", help="Whether to weight the training")
parser.add_argument("--calibration_shift", default=None, type=str, help="whether to apply a calibration shift")
args = parser.parse_args()
args = vars(args)

# The group of experiments that this is part of
group = 'debug'
mode = args.pop('mode')
name = args.pop('name')
path = args.pop('path')
epochs = 100
# epochs = 3
batch_size = 32
weighted = args.pop("weighted")
calibration_shift = args.pop("calibration_shift")
reweigh = False
random_flip = False
input_dims = (100, 80)

filenames = sorted(os.listdir(path))
print("number of total files ====================", len(filenames))
train_files = filenames[0:4 * len(filenames) // 5]
valid_files = filenames[4 * len(filenames) // 5:]
print("Getting sample size count...")

train_files, train_count = sample_size_cvn(train_files, path, mode=mode)
name += "_num_train_samples_{}_".format(train_count)
print("Number of training examples:\t{}".format(train_count))

valid_files, valid_count = sample_size_cvn(valid_files, path, mode=mode)
name += "_num_valid_samples_{}_".format(valid_count)
print("Number of validation examples:\t{}".format(valid_count))

train_gen = nova.cvngenerator(path, train_files, batch_size=batch_size, weighted=weighted, mode=mode)
valid_gen = nova.cvngenerator(path, valid_files, batch_size=batch_size, mode=mode)

n_steps = train_count // batch_size // 4
val_steps = valid_count // batch_size // 4

print(args)
print(name)
print(group)
# Model

arch = args.pop('arch')
assert arch in ["googlenet", "mobilenet", "resnet"], "Architecture not known"  # prompt empty arch if not in the list
print("Using architecture", arch)

if arch == "googlenet":
    hparams = {'optimizer': 'adam',
               'lr': 1e-3,
               'lrdecay': 0.00001,
               'dropout': 0.,
               'momentum': 0.7,
               'filter_number': 32,
               'num_layers': 3,
               'l2': 0.,
               'fc_l2': 0.,
               'input_scaling': args['input_scaling'],
               'pooling': args['pooling']}
    model = nova.regression.regression(hparams=hparams, input_dims=input_dims, vertex=False)
elif arch == "mobilenet":
    hparams = {'optimizer': 'adam',
               'lr': 1e-3,
               'lrdecay': 0.00001,
               'dropout': 0.3,
               'momentum': 0.7,
               'filter_number': 32,
               'num_layers': 3,
               'l2': 0.,
               'fc_l2': 0.,
               'input_scaling': args['input_scaling']}
    model = nova.regression_mobilenet.regression(hparams=hparams, input_dims=input_dims, vertex=False)
else:
    hparams = {'pooling': 'average',
               'activation': 'relu',
               'skipconnection': 'yes',
               'kernel_init': 'he_normal',
               'num_blocks': 1,
               'num_top_blocks': 0,
               'num_dense': 1,
               'filter_number': 64,
               'input_scaling': 0.5,
               'optimizer': 'adam',
               'l2': 0.,
               'fc_l2': 0.,
               'lr': 1e-3}
    # this regression_sherpa is in nova/nova directory
    model = nova.regression_sherpa.regression(hparams=hparams, input_dims=input_dims, vertex=False)

# Callbacks
# c = nova.regression.callbacks(name=name, group=group, tensorboard=True, reduce_lr=True, model_checkpoint=True,
#                               early_stopping=False)
c = nova.regression.callbacks(name=name, group=group, tensorboard=True, reduce_lr=True, model_checkpoint=True,
                              early_stopping=False)

# Training
workers = multiprocessing.cpu_count()

history = model.fit_generator(generator=train_gen,
                    steps_per_epoch=n_steps,
                    epochs=epochs,
                    validation_data=valid_gen,
                    validation_steps=val_steps,
                    callbacks=c,
                    workers=workers,
                    max_queue_size=128,
                    use_multiprocessing=True,
                    verbose=1)

# change this folder path for your loss records
with open('/home/danglicao/nova/logs/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
model.save(os.path.join(nova.config.MODEL_DIR, group, name))
