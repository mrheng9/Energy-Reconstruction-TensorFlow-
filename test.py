from __future__ import print_function
import h5py
import numpy as np
import pickle
import os
import sys
import argparse
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用 GPU 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)
from keras import backend as K
K.set_session(sess)


from keras.models import load_model
import nova
import nova.regression_sherpa
import nova.regression_mobilenet

# from tensorflow.nn import relu6
def relu6(x):
    return K.relu(x, max_value=6)
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", help="The model path", type=str)
parser.add_argument("--steps", help="The number of test steps", type=int, default=0)
parser.add_argument("--batch_size", help="The batchsize", type=int, default=128)
parser.add_argument("--mode", default='nue', type=str)
parser.add_argument("--name", default='', type=str)
parser.add_argument("--path", default='', type=str)
parser.add_argument("--all_files", default=False, action="store_true")
parser.add_argument("--calibration_shift", default=None, type=str)

args = parser.parse_args()

# Set constants
mode = args.mode
name = args.name
path = args.path
batch_size = args.batch_size
valid_steps = args.steps
calibration_shift = args.calibration_shift
model = load_model(args.modelpath, custom_objects={'loss': nova.regression_sherpa.get_loss('sqrt,0.05,10.'), 'relu6': relu6})
save_batches = False

filenames = sorted(os.listdir(path))
if args.all_files:
    valid_files = filenames
else:
    valid_files = filenames[4*len(filenames)//5:]
print("Getting sample size count...")
valid_files, valid_count = sample_size_cvn(valid_files, path, mode=mode)
# valid_count = 1000
name += "_num_samples_{}_".format(valid_steps*args.batch_size)
print("Number of validation examples:\t{}".format(valid_count))

valid_gen =  nova.cvngenerator(path, valid_files, batch_size=batch_size, mode=mode, weighted=False, calibration_shift=calibration_shift)

if valid_steps == 0:
    valid_steps = valid_count//batch_size

y_arr, yhat_arr = [], []
for i in range(valid_steps):
    sys.stdout.write("Step {}/{}\n".format(i, valid_steps))
    sys.stdout.flush()
    x, y = next(valid_gen)
    yhat = model.predict(x)
    y_arr.append(y)
    yhat_arr.append(yhat)
    if save_batches:
        with open('example_batch_{}.pkl'.format(model_name), 'wb') as f:
            pickle.dump({'x': x, 'yhat': yhat}, f)
        break
    
y_arr = np.concatenate(y_arr)
y_arr = y_arr.reshape(len(y_arr),1)    
yhat_arr = np.concatenate(yhat_arr)
resolution = yhat_arr/y_arr
print("Prediction IQR=%s" % (np.percentile(resolution, 75)-np.percentile(resolution, 25)))
save_path = "/home/houyh/nova/nova/predictions/{}_{}.pkl".format(name, time.strftime("%Y-%m-%d--%H-%M-%S"))
print("Saving to: {}".format(save_path))
with open(save_path, 'wb') as f:
    pickle.dump({'y': y_arr, 'yhat':yhat_arr, 'resolution': resolution}, f, protocol=2)
