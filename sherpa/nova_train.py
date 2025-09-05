from __future__ import print_function
import os
import random
import time
import gpu_lock

GPUIDX = gpu_lock.obtain_lock_id() # Return gpuid, or -1 if there was a problem.
assert GPUIDX >= 0, '\nNo gpu available.'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUIDX)

import h5py
import nova
import keras
import sherpa
client = sherpa.Client()
trial = client.get_trial()
# except:
#     print("Using default params")
    # trial = sherpa.Trial(id=0, parameters={'lr': 1e-3, 'lrdecay':1e-4,
    #                                        'momentum':0.9, 'batch_size':16})



# Parse command line arguments:
# l2 (float, default=0.): l2 norm multiplier
# fc_l2 (float, default=0.): l2 norm multiplier for fully connected layers
# dropout (float, default=0.): dropout applied to fully connected layers
# optimizer (str, default='sgd'): 'adam' or 'sgd'
# lr (float, default=0.001): learning rate
# batch_size (int, default=64): training batch size
# filter_number (int, default=32)
args = nova.parse_args(trial.parameters)

# Appends [argument-name]_[argument-value] pairs and date/time to basename
name = nova.generate_name(basename='nue', args=args)

# The group of experiments that this is part of
group = 'debug'

# Data generation         
x_path = '/baldig/physicsprojects/nova/data/lars/electron_event_energy/electron_event_images_unchunked_0_2033'
y_path = '/baldig/physicsprojects/nova/data/lars/electron_event_energy/electron_event_energies_unchunked_0_2033'
vtx_path = '/baldig/physicsprojects/nova/data/lars/electron_event_energy/electron_event_vertices_unchunked_0_2033'

# Set constants
epochs = 100
batch_size = args.pop('batch_size')
reweigh = True
random_flip = True

# filenames = [str(i) + ".h5" for i in range(1300)]
filenames = [n for n in os.listdir(x_path) if n.endswith(".h5")]


train_gen = nova.sample_unchunked_generator(x_path, y_path, vtx_path, batch_size, filenames=filenames[0:1800], reweigh=reweigh, random_flip=random_flip)
valid_gen = nova.sample_unchunked_generator(x_path, y_path, vtx_path, batch_size, filenames=filenames[1800:], reweigh=False, random_flip=False)

n_steps = 27000*64//batch_size//3
val_steps = 3500*64//batch_size//3

# n_steps = 10
# val_steps = 10


print(args)
print(name)
print(group)
# Model
model = nova.regression(hparams=args, input_dims=(151, 141))

send_call = lambda epoch, logs: client.send_metrics(trial=trial,
                                                   iteration=epoch, objective=logs['val_mean_absolute_percentage_error'], context={'val_loss': logs['val_loss'], 'loss': logs['loss'], 'mean_absolute_percentage_error': logs['mean_absolute_percentage_error']})


c = [keras.callbacks.LambdaCallback(on_epoch_end=send_call)]

# Training
try:
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=n_steps,
                        epochs=epochs,
                        validation_data=valid_gen,
                        validation_steps=val_steps,
                        callbacks=c,
                        verbose=2)
finally:
    print("Releasing GPU {}".format(GPUIDX))
    gpu_lock.free_lock(GPUIDX)