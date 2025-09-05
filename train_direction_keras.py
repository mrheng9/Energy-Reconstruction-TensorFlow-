import numpy as np
import argparse
import pickle
import datetime
import os
import time
import nova
import sherpa
parser = argparse.ArgumentParser()
# parser.add_argument("--load", default='', type=str)
parser.add_argument("--notsherpa", default=False, action='store_true')
parser.add_argument("--init", default='glorot_uniform', type=str)
parser.add_argument("--name", default='', type=str)
parser.add_argument("--test", default='', type=str, help="name of model to test (from checkpoint dir)")
parser.add_argument("--steps", default=4000, type=int)
parser.add_argument("--valsteps", default=800, type=int)
parser.add_argument("--nue", default=False, action='store_true')
args = parser.parse_args()                 

if args.notsherpa:
#     loadpath = args.load
#     savepath = "dir_model_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    lr = 0.00021
    reduction = 0.9
    init = args.init
    model_name = args.name
else:
    client = sherpa.Client()
    trial = client.get_trial()
    lr = trial.parameters['lr']
    reduction = trial.parameters['reduction']
    model_name = 'lr-{}-reduction-{}'.format(lr, reduction)
    init = 'glorot_uniform'
    gpu = os.environ.get("SHERPA_RESOURCE", '')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

nfilters = 32
stride = 1
kernel = 3
nlayers = 4
import h5py
import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.backend as K
# # from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model
import keras
import keras.backend as K
# from tensorflow.keras.layers import *
from keras.models import Model
from keras.models import load_model

                    

X = keras.layers.Input(shape=(76, 141, 1))
activations = [X]

for i in range(nlayers):
    Z1 = keras.layers.Conv2D(nfilters, activation='relu', kernel_size=kernel, padding="SAME", kernel_initializer=init)(activations[-1])
    Z2 = keras.layers.Conv2D(nfilters, activation='relu', kernel_size=kernel, padding="SAME", kernel_initializer=init)(Z1)
    Zpooled = keras.layers.MaxPooling2D(pool_size=(2, 2))(Z2)
    activations.append(Zpooled)

convout = keras.layers.Flatten()(activations[-1])
fc = keras.layers.Dense(1024, activation='relu', kernel_initializer=init)(convout)
output = keras.layers.Dense(1, activation='linear', kernel_initializer=init)(fc)
predicted_direction = keras.layers.Lambda(lambda x: K.squeeze(K.stack([K.ones_like(x), x], axis=1), axis=2))(output)
# predicted_direction = output

def generator(input_path, reco_path, output_path, batch_size, filenames, check_ids=False, shuffle=False, view=0, yieldreco=False, cut=False, oneloop=False, skip_nan=False):
    """
    Args:
        input_path (str): path to pixel map HDF5 files
        reco_path (str): path to reco HDF5 files
        output_path (str): path to energy HDF5 files
        batch_size (int): the batch size
        filenames (list[str]): the names of the HDF5 files (files are expected to have the same names for pixelmap/energy/vtx but different directories)
    """
    while True:
        if shuffle:
            random.shuffle(filenames)
        for fname in filenames:
            with h5py.File(os.path.join(input_path, fname), 'r') as xfile, \
                 h5py.File(os.path.join(reco_path, fname), 'r') as recofile, \
                 h5py.File(os.path.join(output_path, fname), 'r') as yfile:

                try:
                    x = xfile['data']
                    reco = recofile['data']
                    y = yfile['data']
                except KeyError:
                    print("skipping {}".format(fname))
                    continue
                    
                ### test that ids match
                if check_ids:
                    assert (xfile['id'][...] == yfile['id'][...]).sum() == x.shape[0]
                    
                if skip_nan:
                    valid_idxs = np.where(~np.isnan(y[:, 0]))[0]
                    num_samples = len(valid_idxs)
                else:
                    num_samples = x.shape[0]
                
                num_batches = num_samples//batch_size

                for j in range(num_batches):
                    if skip_nan:   
                        idxs = valid_idxs[j*batch_size:(j+1)*batch_size]
                    else:
                        idxs = list(range(j*batch_size, (j+1)*batch_size))
                    ax0 = x[idxs, 0, :, :]
                    targets = y[idxs, :]
                    reco_out = reco[idxs, :]
                    
                    if cut:
                        include = []
                        for sampleidx in range(batch_size):
                            xvals, yvals = np.nonzero(ax0[sampleidx])
                            if len(xvals) < 10 or len(np.unique(xvals)) < 4:
                                include.append(False)
                            else:
                                include.append(True)
                        include = np.array(include)
                    else:
                        include = np.array([True]*batch_size)

                    if yieldreco:
                        yield np.expand_dims(ax0, -1)[include], targets[:, [0, 1]][include], reco_out[:, [0, 1]][include]
                    else:
                        yield np.expand_dims(ax0, -1)[include], targets[:, [0, 1]][include]
        if oneloop:
            break


def angle_loss(y_true, y_pred):
    true_direction_normed = tf.nn.l2_normalize(y_true, axis=1)
    predicted_direction_normed = tf.nn.l2_normalize(y_pred, axis=1)

    cos_loss = tf.losses.cosine_distance(true_direction_normed, predicted_direction_normed, axis=1, reduction=tf.losses.Reduction.NONE)
    cos_loss_adj = tf.minimum(cos_loss, 2.-cos_loss)
    loss = tf.reduce_mean(cos_loss_adj)
    return loss
        
def angle_errs(y_true, y_pred):
    true_normed = tf.nn.l2_normalize(y_true, axis=1)
    pred_normed = tf.nn.l2_normalize(y_pred, axis=1)
    
    dot_prod = tf.reduce_sum(true_normed * pred_normed, axis=1)
    dot_prod_clipped = tf.clip_by_value(dot_prod, clip_value_min=-1., clip_value_max=1.)
    angle_err = tf.acos(dot_prod_clipped)
    angle_err_adj = tf.minimum(np.pi-angle_err, angle_err)
    return angle_err_adj

def mean_angle_diff(y_true, y_pred):
    angle_err_adj = angle_errs(y_true, y_pred)
    return tf.reduce_mean(angle_err_adj)




if __name__ == '__main__':

    x_path = "/baldig/physicsprojects/nova/data/flat/small_{}_images".format("event" if args.nue else "electron")
    reco_path = "/baldig/physicsprojects/nova/data/flat/{}_reco_direction".format("event" if args.nue else "electron")
    y_path = "/baldig/physicsprojects/nova/data/flat/{}_direction".format("event" if args.nue else "electron")
    batch_size=16
    
    filenames = ['{}.h5'.format(i) for i in range(100)]
    assert all(os.path.isfile(os.path.join(x_path, fname)) for fname in filenames)
    if not all(os.path.isfile(os.path.join(reco_path, fname)) for fname in filenames):
        print("Reco files missing: ",
              [fname for fname in filenames if not os.path.isfile(os.path.join(reco_path, fname))])
        assert False
    if not all(os.path.isfile(os.path.join(y_path, fname)) for fname in filenames):
        print("Dir files missing: ",
              [fname for fname in filenames if not os.path.isfile(os.path.join(y_path, fname))])
        assert False
    train_files = filenames[0:300]
    valid_files = filenames[300:350]
    test_files = filenames[350:394]
#     test_files = filenames[80:100]
    print("Number of training files:\t{}\nNumber of validation files:\t{}\nNumber of test files:\t{}".format(
        len(train_files), len(valid_files), len(test_files)))

    if args.test == '':
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9,
                                          beta_2=0.999, epsilon=None,
                                          decay=0.0, amsgrad=False)
        model = Model(inputs=X, outputs=predicted_direction)
        model.compile(optimizer=optimizer,
                      loss=angle_loss,
                      metrics=[mean_angle_diff])
        print(model.summary())

        gen = generator(input_path=x_path,
                        output_path=y_path,
                        reco_path=reco_path,
                        batch_size=batch_size,
                        filenames=train_files,
                        cut=True,
                        skip_nan=args.nue)
        validgen = generator(input_path=x_path,
                             output_path=y_path,
                             reco_path=reco_path,
                             batch_size=batch_size,
                             filenames=valid_files,
                             cut=True,
                             skip_nan=args.nue)

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor='mean_angle_diff',
                                              factor=reduction, patience=4,
                                              verbose=1, mode='auto',
                                              min_delta=0.001,
                                              cooldown=0,
                                              min_lr=0),
            keras.callbacks.EarlyStopping(monitor='mean_angle_diff',
                                          min_delta=0,
                                          patience=15,
                                          verbose=1,
                                          mode='auto',
                                          baseline=None),
            tf.keras.callbacks.TensorBoard(log_dir='./direction_logs/{}-{}/'.format(
                model_name,
                time.strftime("%Y-%m-%d--%H-%M-%S")),
                                           histogram_freq=0,
                                           write_graph=False,
                                           write_grads=True,
                                           write_images=False,
                                           embeddings_freq=0,
                                           embeddings_layer_names=None,
                                           embeddings_metadata=None,
                                           embeddings_data=None),
            keras.callbacks.ModelCheckpoint('./direction_checkpoints/{}-{}'.format(
                model_name, time.strftime("%Y-%m-%d--%H-%M-%S")),
                                            monitor='val_mean_angle_diff',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)]
        if not args.notsherpa:
            callbacks.append(
                client.keras_send_metrics(trial,
                                          objective_name='mean_angle_diff',
                                          context_names=['loss', 'val_loss',
                                                         'val_mean_angle_diff']))
            
        print("Training...")
        model.fit_generator(gen,
                            validation_data=validgen,
                            steps_per_epoch=args.steps,
                            validation_steps=args.valsteps,
                            epochs=1000,
                            callbacks=callbacks,
                            verbose=1)
        
    else:
        model = load_model('./direction_checkpoints/{}'.format(args.test),
                           custom_objects={'angle_loss': angle_loss,
                                           'mean_angle_diff': mean_angle_diff})
        testgen = generator(input_path=x_path,
                            output_path=y_path,
                            reco_path=reco_path,
                            batch_size=batch_size,
                            filenames=test_files,
                            cut=True,
                            oneloop=True,
                            yieldreco=True)
        
        y_arr, yhat_arr, recoy_arr = [], [], []
        for step, (x, y, recoy) in enumerate(testgen):
            print("Step {}".format(step), end='\r')
            yhat = model.predict(x)
            y_arr.append(y)
            yhat_arr.append(yhat)
            recoy_arr.append(recoy)
            if step == args.steps:
                break

        y_arr = np.concatenate(y_arr)
        yhat_arr = np.concatenate(yhat_arr)
        recoy_arr = np.concatenate(recoy_arr)

        save_path = "./direction_predictions/{}-test-time-{}-steps-{}.pkl".format(
            args.test,
            time.strftime("%Y-%m-%d--%H-%M-%S"),
            args.steps)
        print("Saving to: {}".format(save_path))
        with open(save_path, 'wb') as f:
            pickle.dump({'y': y_arr,
                         'yhat':yhat_arr,
                         'recoy': recoy_arr}, f, protocol=2)

