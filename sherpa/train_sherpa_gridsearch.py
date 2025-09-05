"""

Important: This file assumes that you have updated sherpa_regression.py. 
This was done to allow the function "regression" to take a new argument: dense_size (number of nodes per layer). around line 131. 
Please update regression_sherpa.py or use this file to test different parameters.

The bottom of this file also contains commented code that would write the results of every trial into a text file for later analysis. 

"""


from __future__ import print_function
import os

if True:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
    CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
    sess = tf.Session(config=CONFIG)
    from keras import backend as K
    K.set_session(sess)
else:
    print("Using Theano")
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda0,gpuarray.preallocate=1'


import sherpa
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
batch_size = 32
weighted = args.pop("weighted")
calibration_shift = args.pop("calibration_shift")
reweigh = False
random_flip = False
input_dims = (100, 80)

filenames = sorted(os.listdir(path))
train_files = filenames[0:4*len(filenames)//5]
valid_files = filenames[4*len(filenames)//5:]
print("Getting sample size count...")

train_files, train_count = sample_size_cvn(train_files, path, mode=mode)
name += "_num_train_samples_{}_".format(train_count)
print("Number of training examples:\t{}".format(train_count))

valid_files, valid_count = sample_size_cvn(valid_files, path, mode=mode)
name += "_num_valid_samples_{}_".format(valid_count)
print("Number of validation examples:\t{}".format(valid_count))

train_gen = nova.cvngenerator(path, train_files, batch_size=batch_size, weighted=weighted, mode=mode)
valid_gen = nova.cvngenerator(path, valid_files, batch_size=batch_size, mode=mode)

n_steps = train_count//batch_size//4
val_steps = valid_count//batch_size//4

print(args)
print(name)
print(group)


# To get multiple trials of the same hyperparameters
for attempt in range(1):    
    
    parameters = [sherpa.Discrete('num_dense', [1, 3]), 
                  sherpa.Ordinal('dense_size', [32, 64, 128, 256, 512, 1024])]
    algorithm = sherpa.algorithms.GridSearch(num_grid_points=3)  # 3 points to get discrete num_dense to be 1,2, and 3. 

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True)
    
    
    # Testing Trials
    for trial in study:
        # Retrieving hyperparameters
        num_dense = trial.parameters['num_dense']
        dense_size = trial.parameters['dense_size']

    
        # defining hyperparameters space
        hparams={'pooling': 'average',
            'activation': 'relu',             
            'skipconnection': 'yes',
            'kernel_init': 'he_normal',
            'num_blocks': 1,       
            'num_top_blocks': 0,   
            'num_dense': num_dense,        # hparam
            'filter_number': 64,   
            'input_scaling': 0.5,
            'optimizer': 'adam',
            'l2': 0.,
            'fc_l2': 0.,
            'lr': 0.0005}                     
        print(hparams)
        print(f"Number of Nodes: {dense_size}")
        
    
        # Creating Model
        model = nova.regression_sherpa.regression(hparams=hparams, input_dims=input_dims, vertex=False, dense_size=dense_size)
        # Callbacks
        c = nova.regression.callbacks(name=name, group=group, tensorboard=False, reduce_lr=True, model_checkpoint=True, early_stopping=True)
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


        # Updating Study    
        val_loss = history.history['val_loss']
        accuracy = history.history['mean_absolute_percentage_error']
        for i, stat in enumerate(val_loss):
            study.add_observation(trial=trial, iteration=i,
                                  objective=accuracy[i],
                                  context={'val_loss': stat})
        study.finalize(trial=trial)


        # write to gridsearch_results when this is finished
        # destination = "/home/asmet/nova/my_scripts/gridsearch_results.txt"
        # with open(destination, "a") as file:
        #     file.write(f"Attempt {attempt+1} of {num_dense} layers and {dense_size} nodes per layer\n")
        #     file.write(f"Lowest Validation MAPE: {min(accuracy)}\n\n")
    
    # destination = "/home/asmet/nova/my_scripts/gridsearch_results.txt"
    # with open(destination, "a") as file:
    #    file.write(study.get_best_result())
