import h5py
import os
import numpy as np
import pandas as pd
from nova.utils import *


def sample_size(files, *paths):
    count = 0
    for i in reversed(range(len(files))):
        fname = files[i]
        samples_in_file = []
        rm = False
        for path in paths:
            with h5py.File(os.path.join(path, fname), 'r') as f:
                try:
                    thiscount = f['data'].shape[0]
                except KeyError:
                    print("File {} corrupted. Removing it.".format(os.path.join(path, fname)))
                    os.remove(os.path.join(path, fname))
                    rm = True
                    thiscount = 0
            samples_in_file.append(thiscount)
        if len(set(samples_in_file)) != 1:
            assert rm, print(samples_in_file)
        else:
            count += samples_in_file[0]
        if rm:
            files.pop(i)
    return files, count

def sample_size_cvn(files, *paths, mode):
    count = 0
    new_files = []
    for i in reversed(range(len(files))):
        fname = files[i]
        for path in paths:
            try:
                with h5py.File(os.path.join(path, fname), 'r') as f:
                    if mode == 'nue':
                        c = len(f['eventtrueE/df'])
                    elif mode == 'electron':
                        c = len(f['prongtrueE/df'])
                    else:
                        print ("wrong mode... check your mode")
                        os.abort()
                    count += c
                new_files.append(fname)
            except OSError:
                print ("Error on ", fname)
                #os.abort()
                pass

    return new_files, count



def sample_size_caf(files, *paths):
    count = 0
    new_files = []
    for i in reversed(range(len(files))):
        fname = files[i]
        for path in paths:
            try:
                with h5py.File(os.path.join(path, fname), 'r') as f:
                    # find indices of Nue CC samples
                    # QE:4, Res:5, DIS:6, Others: 7
                    nuecc_idx = list(np.where( (f['rec.training.trainingdata/interaction'][:,0]>= 4) &
                        (f['rec.training.trainingdata/interaction'][:,0]<= 7)
                        )[0])
                    count += len(nuecc_idx)
                new_files.append(fname)
            except OSError:
                print ("Error on ", fname)
                #os.abort()
                pass

    return new_files, count

def sample_size_caf_prong(files, *paths):
    count = 0
    new_files = []
    for i in reversed(range(len(files))):
        fname = files[i]
        for path in paths:
            try:
                with h5py.File(os.path.join(path, fname), 'r') as f:
                    df_png = get_prongdata_index(f)
                    count += len(df_png)
                new_files.append(fname)
            except OSError:
                print ("Error on ", fname)
                #os.abort()
                pass

    return new_files, count
