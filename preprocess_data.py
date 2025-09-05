import numpy as np
import os
import h5py
import pandas as pd

from PandAna.core.core import *
from PandAna.cut.analysis_cuts import kNueProngContainment, kVeto
from PandAna.cut.regcvn_training_cuts import *
from PandAna.var.regcvn_training_vars import *
from PandAna.var.analysis_vars import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Path to data files", type=str)
parser.add_argument("--nfiles", help="Number of files to process", type=int, default=10)
parser.add_argument("--all", help="Process all files", action="store_true", default=False)
parser.add_argument("--concat_size", help="Number of files to run over for each preprocessed file", type=int, default=25)
parser.add_argument("--output_path", help="Provide output folder to save preprocessed files", type=str)

args = parser.parse_args()

filenames = [os.path.join(args.path, f) for f in sorted(os.listdir(args.path))]
if not args.all:
    filenames = filenames[:args.nfiles]

noutput = len(filenames)//args.concat_size + 1

slc_cut = kSigPresel
prong_cut = kElectronProng
slc_variables = {
        'eventmap': kSliceMap,
        'eventint': kSliceInter,
        'eventtrueE': kSliceNuEnergy,
        'eventcalE': kCaloE,
        'eventrecE': kNueEnergy,
        'eventvtx': kRecoVtx,
        'eventCVNE': kNuERegCVN
        }
prong_variables = {
        'prongmap': kProngMap,
        'prongtrueE': kProngEnergy,
        'prong3mom': kProng3Momentum,
        'prongrecE': kProngRecE,
        'prongCVNE': kElectronRegCVN
            }

import time 

start = time.time()
for fout in range(noutput):
    t0 = time.time()    
    print("Running over {}/{} batch".format(fout,noutput))
    slc_tables = loader(filenames, offset=fout, stride=noutput)
    prong_tables = loader(filenames, offset=fout, stride=noutput, index=KLP)
    specs = {}
    for key in slc_variables.keys():
        specs[key] = spectrum(slc_tables, slc_cut, slc_variables[key])
    for key in prong_variables.keys():
        specs[key] = spectrum(prong_tables, prong_cut, prong_variables[key])
    slc_tables.Go()
    prong_tables.Go()
    concat_dfs = [specs[key].df().reset_index().set_index(KL) for key in prong_variables.keys()]
    png_concat_dfs = pd.concat(concat_dfs, axis = 1, join='inner')
    png_concat_dfs = png_concat_dfs.loc[:,~png_concat_dfs.columns.duplicated()]
    
    # sort prongs by recoE
    index_df = png_concat_dfs[['rece','rec.vtx.mlvertex.fuzzyk.png_idx']].groupby(level=KL, group_keys=False)
    index_df = index_df.apply(lambda x: x.loc[x['rece'] == x['rece'].max(), ['rece','rec.vtx.mlvertex.fuzzyk.png_idx']])

    # locate prong and reset to KL
    slc_index = index_df.reset_index().set_index(KLP).index
    png_concat_dfs = png_concat_dfs.reset_index().set_index(KLP).loc[slc_index].reset_index().set_index(KL)

    # remake dataframes 
    specs['prongmap']._df = png_concat_dfs['cvnmap']
    specs['prong3mom']._df = png_concat_dfs[['p.px','p.py', 'p.pz']]
    specs['prongtrueE']._df = png_concat_dfs['p.E']
    specs['prongrecE']._df = png_concat_dfs['rece']
    specs['prongCVNE']._df = png_concat_dfs['prongE']
    for key in prong_variables.keys():
        png_df = specs[key]._df
        specs[key]._df = pd.concat([png_df, specs['eventcalE'].df()], axis=1, join='inner').drop('calE', axis=1)

    save_tree(os.path.join(args.output_path, 'preprocessed_{}.h5'.format(fout)),
                 list(specs.values()),
                 list(specs.keys()), attrs=False)
    print("=========================")
    print("Batch Processing Time : {}".format(time.time()-t0))
    print("=========================")

print("*************************")
print("Overall running time : {}".format(time.time()-start))
print("*************************")
