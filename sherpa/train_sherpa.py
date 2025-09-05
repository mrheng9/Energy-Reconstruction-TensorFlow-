from __future__ import print_function
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)
from keras import backend as K
K.set_session(sess)

import random
import h5py
import nova
import nova.regression_sherpa
import nova.regression
import keras
from keras.models import load_model
import keras.backend as K
from config import dataset_paths, sample_size

import sherpa
client = sherpa.Client()
trial = client.get_trial()

# Parse command line arguments:
# l2 (float, default=0.): l2 norm multiplier
# fc_l2 (float, default=0.): l2 norm multiplier for fully connected layers
# dropout (float, default=0.): dropout applied to fully connected layers
# optimizer (str, default='adam'): 'adam' or 'sgd'
# lr (float, default=0.001): learning rate
# batch_size (int, default=64): training batch size
# filter_number (int, default=32)
args = nova.regression_sherpa.parse_args(trial.parameters)

# The group of experiments that this is part of
group = 'debug'

mode, dataset = args.pop('mode'), args.pop('dataset')
x_path, y_path, vtx_path = dataset_paths(mode=mode, dataset=dataset, size='small')
    
# Appends [argument-name]_[argument-value] pairs and date/time to basename
name = nova.regression.generate_name(basename='_'.join([mode, dataset]), args=args)

# Set constants
epochs = 20
batch_size = args.pop('batch_size')
reweigh = False
random_flip = False

# filenames = [str(i) + ".h5" for i in range(1300)]
filenames = list(set(n for n in os.listdir(x_path) if n.endswith(".h5")) & set(n for n in os.listdir(y_path) if n.endswith(".h5")))
train_files = filenames[0:4*len(filenames)//5]
valid_files = filenames[4*len(filenames)//5:]
print("Number of training files:\t{}\nNumber of validation files:\t{}".format(len(train_files), len(valid_files)))

# Print number of training examples
train_count = sample_size(train_files, y_path)
name += "_num_train_samples_{}_".format(train_count)
print("Number of training examples:\t{}".format(train_count))
valid_count = sample_size(valid_files, y_path)
name += "_num_valid_samples_{}_".format(valid_count)
print("Number of validation examples:\t{}".format(valid_count))

# Generators
train_gen = nova.generator_mmap(input_path=x_path,
                           vtx_path = vtx_path,
                           output_path=y_path,
                           batch_size=batch_size,
                           filenames=train_files,
                           sample=False,
                           check_ids=True,
                           random_flip=random_flip)
valid_gen = nova.generator_mmap(input_path=x_path,
                           vtx_path = vtx_path,
                           output_path=y_path,
                           batch_size=batch_size,
                           filenames=valid_files,
                           sample=False,
                           check_ids=True,
                           random_flip=False)

n_steps = train_count//batch_size//4
val_steps = valid_count//batch_size//4
# n_steps = 100
# val_steps = 100

# import time

# t = []

# for _ in range(2000):
#     t1 = time.time()
#     next(train_gen)
#     t2 = time.time()
#     t.append(t2-t1)
#     print("Average time per batch {} s".format(sum(t)/float(len(t))))


print(args)
print(name)
print(group)
# Model

model = nova.regression_sherpa.regression(hparams=args, input_dims=(76, 141))
c = nova.regression_sherpa.callbacks(name=name, group=group, tensorboard=False, reduce_lr=True, model_checkpoint=False, early_stopping=False)
c.append(client.keras_send_metrics(trial=trial, objective_name='val_loss', context_names=['loss']))

# Training
model.fit_generator(generator=train_gen,
                    steps_per_epoch=n_steps,
                    epochs=epochs,
                    validation_data=valid_gen,
                    validation_steps=val_steps,
                    callbacks=c,
                    verbose=2)
# model.save(os.path.join(nova.config.MODEL_DIR, group, name))
