from __future__ import print_function
import os
import random
try:
    # gpu_lock module located at /home/pjsadows/libs
    import gpu_lock
    GPUIDX = gpu_lock.obtain_lock_id() # Return gpuid, or -1 if there was a problem.
except:
    print('Could not import gpu_lock. Prepend /extra/pjsadows0/libs/shared/gpu_lock/ to PYTHONPATH.')
    GPUIDX = 0
assert GPUIDX >= 0, '\nNo gpu available.'
print('Running from GPU %s' % str(GPUIDX))
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUIDX)

import nova

# Parse command line arguments: l2, fc_l2, dropout, optimizer, lr, batch_size,
# filter_number
args = nova.parse_args()

# Appends [argument-name]_[argument-value] pairs and date/time to basename
name = nova.generate_name(basename='nue', args=args)

# The group of experiments that this is part of
group = 'debug'

# Set constants
batch_size = args.get('batch_size')
# batch_number = 1000//batch_size
epochs = 100

# Data generation
data_files = nova.list_files(nova.config.NUE_DATA_DIR, filter='.txt')
num_train_files = 1800
num_valid_files = len(data_files) - num_train_files
pm_generator = nova.PixelmapGenerator.quick_build('nue', pixel_map_dim=(args.get('pixelmap_width'), 141))

train_gen = pm_generator.flow(data_files[:num_train_files],
                               batch_size=batch_size,
                               reweigh=args.get('reweigh'))
valid_gen = pm_generator.flow(data_files[num_train_files:],
                               batch_size=batch_size,
                               reweigh=False)

print(args)
print(name)
print(group)
# Model
model = nova.regression(outdim=1, hparams=args, input_dims=(args.get('pixelmap_width'), 141))
# c = nova.callbacks(name=name, group=group, tensorboard=True)
c = []

# Training
model.fit_generator(generator=train_gen,
                    steps_per_epoch=1728000//batch_size,
                    epochs=epochs,
                    validation_data=valid_gen,
                    validation_steps=192000//batch_size,
                    callbacks=c)
