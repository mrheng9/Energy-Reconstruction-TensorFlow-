import h5py
import os

# CHANGE THESE ONLY
d = '/lfstev/nnet/R19-02-23-miniprod5/FD-Nonswap-FHC-Eval/'
var = ['rec.sel.cvn2020veto','rec.sel.cvn2020ptpcut', \
           'rec.sel.cvn2020taucut','rec.sel.cvn2020allcut']

#######################################################################

files = [os.path.join(d,f) for f in os.listdir(d) if 'h5caf.h5' in f]

for f in files:
    h5 = h5py.File(f, 'r')

    for v in var:
        if v not in h5:
            print(f+' has no group named '+v)

