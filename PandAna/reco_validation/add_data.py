import os
import time
import sys

sys.path.append('../')

import h5py
import numpy as np

from keras.models import load_model
from PandAna import *


kVeto = Cut(lambda tables: tables['rec.sel.veto']['keep'] == 1)

def kCVNVar(model):
    classes = ['numuid','nueid','nutauid','ncid','cosmicid']

    def kVar(tables):
        pms = tables['rec.training.cvnmaps']['cvnmap']
        df = pms.apply(lambda x: model.predict(np.array([x]))[0])
        return pd.DataFrame(df.values.tolist(), columns=classes, index=df.index)
    return Var(kVar)

def dfToDict(df):
    withid = df.reset_index()
    ret = {}
    for col in list(withid):
        ret[col] = withid[col].values[..., np.newaxis].astype(np.float32)
    return ret

if __name__ == '__main__':
    # Miniprod 5 h5s
    d = sys.argv[1]
    stride = int(sys.argv[2])
    offset = int(sys.argv[3])
    print('Adding new cvns to files in '+d)
    print('Stride: '+str(stride)+'; Offset: '+str(offset))
    files = [f for f in os.listdir(d) if 'h5caf.h5' in f][offset::stride]
    print('There are '+str(len(files))+' files.')

    # Make Models
    # Base Model
    modelBase  = load_model('models/model_mynet_cos_best.h5')
    # pTp Cut
    modelPTP  = load_model('models/model_mynet_ptp_best.h5')

    modellist = [modelBase, modelPTP]
    namelist  = ['veto', 'ptpcut']

    t0 = time.time()
    # One file at a time to avoid problems with loading a bunch of pixel maps in memory
    for f in files:
        # Make a loader and the spectra to fill
        tables = loader([os.path.join(d,f)])
        specs = []
        for m in modellist:
            specs.append(spectrum(tables, kVeto, kCVNVar(m)))

        # GO GO GO
        tables.Go()

        # Append the results to an existing file
        #h5 = h5py.File(os.path.join(d,f), 'a')
        # Or make a friend
        h5 = h5py.File(os.path.join(outdir,f), 'a')

        for i,s in enumerate(specs):
            thedict = dfToDict(s.df())
            for dataset, vals in thedict.items():
                datastr = 'rec.sel.cvn2020'+namelist[i]+'/'+dataset
                
                if datastr in h5:
                    del h5[datastr]
                h5.create_dataset(datastr, data=vals)

        h5.close()
        print('File '+f+' processed at '+str(time.time()-t0))

    print('Finished in '+str(time.time()-t0))
