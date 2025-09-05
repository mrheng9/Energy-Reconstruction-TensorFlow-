from __future__ import print_function
from itertools import chain
import random
import numpy as np
import h5py
import os
import time
from .dataprocessing import *
from contextlib import contextmanager
import contextlib
from .config import DATA_FORMAT
from .utils import *
import threading

CHANNEL_AXIS = -1 if DATA_FORMAT == 'channels_last' else 1

@contextmanager
def filelist(pathlist):
    openfiles = [h5py.File(p) for p in pathlist]
    yield openfiles
    for f in openfiles:
        f.close()

def lmdbgenerator(path, batch_size, start_event, end_event):
    with lmdb.open(path) as env:
        with env.begin() as txn:
            idx = start_event
            while True:
                data = {}
                for data_type in ['event_images', 'event_energy', 'event_vertices']:
                    dat = []
                    t = [[] ,[], []]
                    for i in range(idx, idx+batch_size):
                        key = data_type + "_" + str(i)
                        t0 = time.time()
                        buf = txn.get(key.encode('ascii'))
                        t[0].append(time.time()-t0)

                        t0 = time.time()
                        string = np.fromstring(buf)
                        t[1].append(time.time()-t0)

                        shape = eval(txn.get((key + "_shape").encode('ascii')))

                        t0 = time.time()
                        dat.append(string.reshape(shape))
                        t[2].append(time.time()-t0)

                    print("Loading batch: get {}s\tfromstring {}s\treshape {}s".format(sum(t[0]), sum(t[1]), sum(t[2])))
                    t0 = time.time()
                    data[data_type] = np.stack(dat)
                    t1 = time.time()
                    print("Stacking batch {}s".format(t1-t0))

                idx += batch_size
                if idx > end_event:
                    idx = start_event

                yield data['event_images'], data['event_vertices'], data['event_energy']

def lmdbcursor(path, batch_size, start_event, end_event):
    with lmdb.open(path) as env:
        with env.begin() as txn:
            idx = start_event
            while True:
                counter = 0
                cursor = txn.cursor()
                t0 = time.time()
                v = []
                for key, value in cursor:
                    counter += 1
                    v.append(value)
                    if counter % 32 == 0:
                        print("Time: {}s".format(time.time()-t0))
                        yield v
                        t0 = time.time()




def generatorfixedcache(input_path, vtx_path, output_path, batch_size, filenames, check_ids=False, sample=False, random_flip=False):
    random.shuffle(filenames)
    while True:
        for fname in filenames:
            with h5py.File(os.path.join(input_path, fname)) as xfile, \
                 h5py.File(os.path.join(vtx_path, fname)) as vtxfile, \
                 h5py.File(os.path.join(output_path, fname)) as yfile:

                try:
                    x = xfile['data']
                    vtx = vtxfile['data']
                    y = yfile['data']
                except KeyError:
                    continue

                num_samples = x.shape[0]

                ### test that ids match
                if check_ids:
                    assert (xfile['id'][...] == yfile['id'][...]).sum() == num_samples

#                 load_size = max(64, batch_size)
                load_size = batch_size
                num_batches = num_samples//load_size  # always load 64 samples at a time for best performance

                for j in range(num_batches):
                    ### sampling
                    if sample:
                        idxs = np.random.choice(list(range(num_samples)), size=load_size, replace=False)
                        idxs = sorted(idxs)
                    else:
                        idxs = list(range(j*load_size, (j+1)*load_size))

                    ax0 = x[idxs, 0, :, :]
                    ax1 = x[idxs, 1, :, :]

                    if random_flip:
                        ax0 = x[idxs, 0, :, :]
                        ax1 = x[idxs, 1, :, :]
                        for k in range(ax0.shape[0]):
                            if np.random.random() < 0.5:
                                ax0[k] = flip_axis(ax0[k], -1)
                            if np.random.random() < 0.5:
                                ax1[k] = flip_axis(ax1[k], -1)

                    targets = y[idxs]
#                     if batch_size <= 64:
#                         for fr, to in zip(range(0, 64, batch_size), range(batch_size, 64+batch_size, batch_size)):
#                             yield [np.expand_dims(ax0[fr:to], -1), np.expand_dims(ax1[fr:to], -1), vtx[idxs[fr:to]]], targets[fr:to]
#                     else:
                    yield [np.expand_dims(ax0, -1), np.expand_dims(ax1, -1), vtx[idxs]], targets


def generator(input_path, vtx_path, output_path, batch_size, filenames, check_ids=False, sample=False, random_flip=False, shuffle=False, calibration_shift=None):
    """
    Args:
        input_path (str): path to pixel map HDF5 files
        vtx_path (str): path to pixel map HDF5 files
        output_path (str): path to energy HDF5 files
        batch_size (int): the batch size
        filenames (list[str]): the names of the HDF5 files (files are expected to have the same names for pixelmap/energy/vtx but different directories)
    """
    while True:
        if shuffle:
            random.shuffle(filenames)
        for fname in filenames:

#             propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
#             settings = list(propfaid.get_cache())
#             settings[2] *= 10
#             propfaid.set_cache(*settings)

#             with contextlib.closing(h5py.h5f.open(os.path.join(input_path, fname).encode('utf-8'), fapl=propfaid)) as xfid, \
#                  h5py.File(os.path.join(vtx_path, fname), 'r') as vtxfile, \
#                  h5py.File(os.path.join(output_path, fname), 'r') as yfile:

            with h5py.File(os.path.join(input_path, fname), 'r') as xfile, \
                 h5py.File(os.path.join(vtx_path, fname), 'r') as vtxfile, \
                 h5py.File(os.path.join(output_path, fname), 'r') as yfile:

                try:
#                     xfile = h5py.File(xfid, 'r')
                    x = xfile['data']
                    vtx = vtxfile['data']
                    y = yfile['data']
                except KeyError:
                    print("skipping {}".format(fname))
                    continue

                num_samples = x.shape[0]

                ### test that ids match
                if check_ids:
                    assert (xfile['id'][...] == yfile['id'][...]).sum() == num_samples

                load_size = batch_size
                num_batches = num_samples//load_size

                for j in range(num_batches):
                    ### sampling
                    if sample:
                        idxs = np.random.choice(list(range(num_samples)), size=load_size, replace=False)
                        idxs = sorted(idxs)
                    else:
                        idxs = list(range(j*load_size, (j+1)*load_size))

                    ax0 = x[idxs, 0, :, :]
                    ax1 = x[idxs, 1, :, :]

                    if random_flip:
                        ax0 = x[idxs, 0, :, :]
                        ax1 = x[idxs, 1, :, :]
                        for k in range(ax0.shape[0]):
                            if np.random.random() < 0.5:
                                ax0[k] = flip_axis(ax0[k], -1)
                            if np.random.random() < 0.5:
                                ax1[k] = flip_axis(ax1[k], -1)

                    if calibration_shift:
                        ax0 *= calibration_shift
                        ax1 *= calibration_shift

                    targets = y[idxs]

                    yield [np.expand_dims(ax0, CHANNEL_AXIS), np.expand_dims(ax1, CHANNEL_AXIS), vtx[idxs]], targets

class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

@threadsafe_generator
def cvngenerator(path, filenames, batch_size, mode='nue', shuffle=False, weighted=False, calibration_shift=None):
    """
    Args:
        path (str): path to HDF5 files
        batch_size (int): the batch size
        filenames (list[str]): the names of the HDF5 files
    """
    if mode == 'nue':
        ffix = 'event'
    elif mode == 'electron':
        ffix = 'prong'
    else:
        print ('wrong mode... check your mode')
        os.abort()
    while True:
        if shuffle:
            random.shuffle(filenames)
        for fname in filenames:
            try:
                with h5py.File(os.path.join(path, fname), 'r') as f:
                    num_samples = len(f[ffix+'trueE/df'])
                    num_batches = num_samples//batch_size

                    for j in range(num_batches):
                        idxs = list(range(j*batch_size,(j+1)*batch_size))
                        ax0 = f[ffix+'map/df'][idxs].reshape(len(idxs),2,100,80)[:,0,:,:]
                        ax1 = f[ffix+'map/df'][idxs].reshape(len(idxs),2,100,80)[:,1,:,:]
                        targets = f[ffix+'trueE/df'][idxs].flatten()

                        if calibration_shift:
                            if calibration_shift == 'discrete':
                                shift = np.random.choice([1., 1., 0.95, 1.05])
                            else:
                                shift = float(calibration_shift)
                            ax0 = ax0*shift
                            ax1 = ax1*shift

                        if weighted:
                            w = get_weights(targets)
                            w /= np.sum(w)
                            yield [np.expand_dims(ax0, CHANNEL_AXIS), np.expand_dims(ax1, CHANNEL_AXIS)], targets, w
                        else:
                            yield [np.expand_dims(ax0, CHANNEL_AXIS), np.expand_dims(ax1, CHANNEL_AXIS)], targets
            except OSError:
                pass


def rawcvngenerator(path, filenames, batch_size, shuffle=False, weighted=False):
    """
    Args:
        path (str): path to HDF5 files
        batch_size (int): the batch size
        filenames (list[str]): the names of the HDF5 files
    """
    while True:
        if shuffle:
            random.shuffle(filenames)
        for fname in filenames:
            try:
                with h5py.File(os.path.join(path, fname), 'r') as f:
                    # find indices of Nue CC samples
                    # QE:4, Res:5, DIS:6, Others: 7
                    nuecc_idx = list(np.where( (f['rec.training.trainingdata/interaction'][:,0]>= 4) &
                        (f['rec.training.trainingdata/interaction'][:,0]<= 7)
                        )[0])

                   # count number of samples in a file
                    num_samples = len(nuecc_idx)
                    num_batches = num_samples//batch_size

                    for j in range(num_batches):
                        idxs = nuecc_idx[j*batch_size:(j+1)*batch_size]
                        ax0 = f['rec.training.cvnmaps/cvnmap'][idxs].reshape(len(idxs),2,100,80)[:,0,:,:]
                        ax1 = f['rec.training.cvnmaps/cvnmap'][idxs].reshape(len(idxs),2,100,80)[:,1,:,:]
                        targets = f['rec.training.trainingdata/nuenergy'][idxs]
                        # set to zeros unless we need reco vertices
#                         reco_vertices = np.zeros((len(idxs),2))

                        if weighted:
                            w = get_weights(targets)
                            w /= np.sum(w)
                            yield [np.expand_dims(ax0, CHANNEL_AXIS), np.expand_dims(ax1, CHANNEL_AXIS)], targets, w
                        else:
                            yield [np.expand_dims(ax0, CHANNEL_AXIS), np.expand_dims(ax1, CHANNEL_AXIS)], targets
            except OSError:
                pass

def rawcvnpronggenerator(path, filenames, batch_size, shuffle=False, weighted=False):
    """
    Args:
        path (str): path to HDF5 files
        batch_size (int): the batch size
        filenames (list[str]): the names of the HDF5 files
    """
    while True:
        if shuffle:
            random.shuffle(filenames)
        for fname in filenames:
            try:
                with h5py.File(os.path.join(path, fname), 'r') as f:
                    df_png = get_prongdata_index(f)
                    # count number of samples in a file
                    num_samples = len(df_png)
                    num_batches = num_samples//batch_size
                    #print (fname)
                    for j in range(num_batches):
                        idxs = df_png['pngtrainidx'][j*batch_size:(j+1)*batch_size]
                        ax0 = f['rec.vtx.elastic.fuzzyk.png.cvnmaps/cvnmap'][idxs].reshape(len(idxs),2,100,80)[:,0,:,:]
                        ax1 = f['rec.vtx.elastic.fuzzyk.png.cvnmaps/cvnmap'][idxs].reshape(len(idxs),2,100,80)[:,1,:,:]
                        targets = df_png['p.E'][j*batch_size:(j+1)*batch_size].values

                        if weighted:
                            w = get_weights(targets)
                            w /= np.sum(w)
                            yield [np.expand_dims(ax0, CHANNEL_AXIS), np.expand_dims(ax1, CHANNEL_AXIS)], targets, w
                        else:
                            yield [np.expand_dims(ax0, CHANNEL_AXIS), np.expand_dims(ax1, CHANNEL_AXIS)], targets
            except OSError:
                pass


def vtxgenerator(input_path, output_path, batch_size, filenames, check_ids=False, sample=False, random_flip=False, shuffle=True):
    """
    Args:
        input_path (str): path to pixel map HDF5 files
        vtx_path (str): path to pixel map HDF5 files
        output_path (str): path to energy HDF5 files
        batch_size (int): the batch size
        filenames (list[str]): the names of the HDF5 files (files are expected to have the same names for pixelmap/energy/vtx but different directories)
    """
    while True:
        if shuffle:
            random.shuffle(filenames)
        for fname in filenames:

            propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            settings = list(propfaid.get_cache())
            settings[2] *= 10
            propfaid.set_cache(*settings)

            with contextlib.closing(h5py.h5f.open(os.path.join(input_path, fname).encode('utf-8'), fapl=propfaid)) as xfid, \
                 h5py.File(os.path.join(output_path, fname)) as yfile:

                try:
                    xfile = h5py.File(xfid)
                    x = xfile['data']
                    y = yfile['data']
                except KeyError:
                    print("skipping {}".format(fname))
                    continue

                num_samples = x.shape[0]

                ### test that ids match
                if check_ids:
                    assert (xfile['id'][...] == yfile['id'][...]).sum() == num_samples

                load_size = batch_size
                num_batches = num_samples//load_size

                for j in range(num_batches):
                    ### sampling
                    if sample:
                        idxs_ = np.random.choice(list(range(num_samples)), size=load_size, replace=False)
                        idxs_ = np.sort(idxs_)
                    else:
                        idxs_ = np.arange(j*load_size, (j+1)*load_size)

                    targets = y[list(idxs_)]
                    idxs = list(idxs_[np.isfinite(np.sum(targets, axis=1))])
                    targets = y[idxs]

                    ax0 = x[idxs, 0, :, :]
                    ax1 = x[idxs, 1, :, :]

                    if random_flip:
                        ax0 = x[idxs, 0, :, :]
                        ax1 = x[idxs, 1, :, :]
                        for k in range(ax0.shape[0]):
                            if np.random.random() < 0.5:
                                ax0[k] = flip_axis(ax0[k], -1)
                            if np.random.random() < 0.5:
                                ax1[k] = flip_axis(ax1[k], -1)



                    # , (np.isnan(np.sum(targets, axis=1))==False).astype('int')

                    yield [np.expand_dims(ax0, CHANNEL_AXIS), np.expand_dims(ax1, CHANNEL_AXIS)], targets



def  _mmap_h5(path, h5path):
    with h5py.File(path) as f:
        ds = f[h5path]
        # We get the dataset address in the HDF5 fiel.
        offset = ds.id.get_offset()
        # We ensure we have a non-compressed contiguous array.
        assert ds.chunks is None
        assert ds.compression is None
        assert offset > 0
        dtype = ds.dtype
        shape = ds.shape
    arr = np.memmap(path, mode='r', shape=shape, offset=offset, dtype=dtype)
    return arr

def generator_mmap(input_path, vtx_path, output_path, batch_size, filenames, check_ids=False, sample=False, random_flip=False, shuffle=True):
    print("Using MMap")
    while True:
        if shuffle:
            random.shuffle(filenames)
        for fname in filenames:
            with h5py.File(os.path.join(vtx_path, fname)) as vtxfile, \
                 h5py.File(os.path.join(output_path, fname)) as yfile:


                try:
                    vtx = vtxfile['data']
                except KeyError:
                    print("Vertex File {} does not have a data object".format(fname))
                    continue
                try:
                    y = yfile['data']
                except KeyError:
                    print("Energy File {} does not have a data object".format(fname))
                    continue
                try:
                    x = _mmap_h5(os.path.join(input_path, fname), 'data')
                except KeyError:
                    print("Image File {} does not have a data object".format(fname))
                    continue


                num_samples = y.shape[0]

                load_size = batch_size
                num_batches = num_samples//load_size  # always load 64 samples at a time for best performance

                for j in range(num_batches):
                    ### sampling
                    if sample:
                        idxs = np.random.choice(list(range(num_samples)), size=load_size, replace=False)
                        idxs = sorted(idxs)
                    else:
                        idxs = list(range(j*load_size, (j+1)*load_size))

                    ax0 = x[idxs, 0, :, :]
                    ax1 = x[idxs, 1, :, :]

                    if random_flip:
                        ax0 = x[idxs, 0, :, :]
                        ax1 = x[idxs, 1, :, :]
                        for k in range(ax0.shape[0]):
                            if np.random.random() < 0.5:
                                ax0[k] = flip_axis(ax0[k], -1)
                            if np.random.random() < 0.5:
                                ax1[k] = flip_axis(ax1[k], -1)

                    targets = y[idxs]

                    yield [np.expand_dims(ax0, CHANNEL_AXIS), np.expand_dims(ax1, CHANNEL_AXIS), vtx[idxs]], targets





def get_weights(y):
    w = np.zeros_like(y)
    w += np.where(np.logical_and(0<=y, y<0.5), np.ones_like(y), np.zeros_like(y))*42.5586924
    w += np.where(np.logical_and(0.5<=y, y<5), np.ones_like(y), np.zeros_like(y))*(83.0923-97.7887*y+35.1566*y**2-3.42726*y**3)
    w += np.where(np.logical_and(5<=y, y<15), np.ones_like(y), np.zeros_like(y))*(-137.471+68.8724*y-7.69719*y**2+0.26117*y**3)
    w += np.where(15<=y, np.ones_like(y), np.zeros_like(y))*45.196
    return w.reshape(-1)


def samplegenerator(x, y, vtx, batch_size, reweigh=False, random_flip=False):
    n_batches = x.shape[0]//batch_size
    while True:
        batch_idxs = range(n_batches)
        n = y.shape[0]
        scope = 500
        num_batches = scope//batch_size
        for l, u in zip(range(0, n, scope), range(scope, n+scope, scope)):
            if reweigh:
                p = get_weights(y[l:u])
                p = np.sqrt(p)
            else:
                p = np.ones_like(y[l:u]).reshape(-1)
            p /= np.sum(p)
            for _ in range(num_batches):
                idxs = np.random.choice(list(range(l, l+p.shape[0])), size=batch_size, replace=False, p=p)
                idxs = sorted(idxs)
                if reweigh:
                    w = np.zeros((batch_size,))
                    w = get_weights(y[idxs])
                    w = np.sqrt(w)
                else:
                    w = np.ones((batch_size,))
                w /= np.sum(w)
                w *= batch_size
                if random_flip:
                    # ax0 = np.copy(x[idxs, 0, :, :])
                    # ax1 = np.copy(x[idxs, 1, :, :])
                    ax0 = x[idxs, 0, :, :]
                    ax1 = x[idxs, 1, :, :]
                    for k in range(ax0.shape[0]):
                        if np.random.random() < 0.5:
                            ax0[k] = flip_axis(ax0[k], -1)
                        if np.random.random() < 0.5:
                            ax1[k] = flip_axis(ax1[k], -1)
                    yield [np.expand_dims(ax0, -1), np.expand_dims(ax1, -1), vtx[idxs]], y[idxs], w

                else:
                    yield [np.expand_dims(x[idxs, 0, :, :], -1), np.expand_dims(x[idxs, 1, :, :], -1), vtx[idxs]], y[idxs], w


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def sample_unchunked_generator(xdir, ydir, vtxdir, batch_size, filenames, reweigh=False, random_flip=False, test=False):
    random.shuffle(filenames)
    while True:
        for fname in filenames:
            with h5py.File(os.path.join(xdir, fname)) as xdata, \
                h5py.File(os.path.join(ydir, fname)) as ydata, \
                h5py.File(os.path.join(vtxdir, fname)) as vtxdata:

                x = xdata['data']
                y = ydata['data']
                vtx = vtxdata['data']

                num_samples = len(y)

                ### test that ids match
                if test:
                    idset = 'id_' + str(dset[-1])
                    assert (xdata['id'][...] == ydata['id'][...]).sum() == num_samples
                    assert (xdata['id'][...] == vtxdata['id'][...]).sum() == num_samples

                num_batches = num_samples//batch_size
                ### reweigh
                if reweigh:
                    w = get_weights(y[...])
                    p = np.sqrt(w)
                else:
                    p = np.ones_like(y[...]).reshape(-1)
                    w = np.ones_like(y[...]).reshape(-1)
                p /= np.sum(p)


                for _ in range(num_batches):
                    ### sampling
                    idxs = np.random.choice(list(range(num_samples)), size=batch_size, replace=False, p=p)
                    idxs = sorted(idxs)
                    batch_weights = w[idxs]
                    batch_weights = np.sqrt(batch_weights)
                    batch_weights /= np.sum(batch_weights)
                    batch_weights *= batch_size

                    if random_flip:
                        ax0 = x[idxs, 0, :, :]
                        ax1 = x[idxs, 1, :, :]
                        for k in range(ax0.shape[0]):
                            if np.random.random() < 0.5:
                                ax0[k] = flip_axis(ax0[k], -1)
                            if np.random.random() < 0.5:
                                ax1[k] = flip_axis(ax1[k], -1)
                        yield [np.expand_dims(ax0, -1), np.expand_dims(ax1, -1), vtx[idxs]], y[idxs], batch_weights

                    else:
                        yield [np.expand_dims(x[idxs, 0, :, :], -1), np.expand_dims(x[idxs, 1, :, :], -1), vtx[idxs]], y[idxs], batch_weights
