import tensorflow as tf
CONFIG = tf.ConfigProto(device_count = {'GPU': 0}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)

from keras import backend as K
K.set_session(sess)
from keras.models import load_model

import nova
import nova.regression_sherpa
import nova.regression_mobilenet

def relu6(x):
 return K.relu(x, max_value=6)
from config import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", help="The model path", type=str)
args = parser.parse_args()

model = load_model(args.modelpath,custom_objects={'loss': nova.regression_sherpa.get_loss('sqrt,0.05,10.'), 'relu6': relu6})

pred_node_names = ['dense_2_1']
num_output = len(pred_node_names)

pred = [tf.identity(model.outputs[i], name = pred_node_names[i]) for i in range(num_output)]

from tensorflow.python.framework import graph_util
od_graph_def = graph_util.convert_variables_to_constants(sess,sess.graph.as_graph_def(),pred_node_names)

frozen_graph_path=args.modelpath+'.pb'
with tf.gfile.GFile(frozen_graph_path, 'wb') as f: 
    f.write(od_graph_def.SerializeToString())