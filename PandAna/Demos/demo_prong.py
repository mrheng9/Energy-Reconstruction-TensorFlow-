import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import PandAna.core as pa
from PandAna.core.core import KL, KLN, KLS
from PandAna.cut.analysis_cuts import *
from PandAna.var.analysis_vars import *

# Demo script for analysing neutron prongs on ND RHC data, similar to Miranda's caf level analysis

KLP = KL + ['rec.vtx.mlvertex.fuzzyk.png_idx']    

# demo slice cuts 
def kNDRockFilter(tables):
    df = tables['rec.vtx.mlvertex']
    ret = (df['vtx.x'] > -180) & \
            (df['vtx.x'] < 180) & \
            (df['vtx.y'] > -180) & \
            (df['vtx.y'] < 180) & \
            (df['vtx.z'] > 20)
    ret.name = 'rock'
    return ret
kNDRockFilter = Cut(kNDRockFilter)

# Prong Cuts that need slice indices
def kProngDispl(tables):
    dfvtx = tables['rec.vtx.mlvertex'][['vtx.x', 'vtx.y', 'vtx.z']]
    dfpng = tables['rec.vtx.mlvertex.fuzzyk.png'][['start.x', 'start.y', 'start.z', 'rec.vtx.mlvertex.fuzzyk.png_idx']]

    xdiff = dfvtx['vtx.x'] - dfpng['start.x']
    ydiff = dfvtx['vtx.y'] - dfpng['start.y']
    zdiff = dfvtx['vtx.z'] - dfpng['start.z']
    length = np.sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff)
    length.name = 'disp'
    idx = dfpng['rec.vtx.mlvertex.fuzzyk.png_idx']
    ret = pd.concat([length, idx], axis=1)
    ret = ret.reset_index().set_index(KLP)

    return (ret['disp'] > 20).where(ret['disp'] > 20)


# prong cuts that don't need slice indices
kProngNhitsCut = Cut(lambda tables: tables['rec.vtx.mlvertex.fuzzyk.png']['nhit'] < 6)
kPhotonID2018Cut = Cut(lambda tables: tables['rec.vtx.mlvertex.fuzzyk.png.cvnpart']['photonid'] < 0.6)
kIsNeutron = Cut(lambda tables: abs(tables['rec.vtx.mlvertex.fuzzyk.png.truth']['motherpdg']) == 2112)

def kIgnoreMuon2018(tables):
    dfpng = tables['rec.vtx.mlvertex.fuzzyk.png.cvnpart']['muonid']
    dflen = tables['rec.vtx.mlvertex.fuzzyk.png']['len']
    if not dfpng.empty:
      dflen = dflen.reset_index().set_index(KL)
      dfpng = dfpng.reset_index().set_index(KL)
      maxid = dfpng['muonid'].groupby(level=KL).agg(np.max)
      ret = pd.concat([dfpng['muonid'] - maxid, dfpng['rec.vtx.mlvertex.fuzzyk.png_idx'], dflen['len']], axis=1)
      ret = ret.reset_index().set_index(KLP)
      return ((ret['muonid'] < 0) & (ret['len'] <= 500))
    else:
      return dfpng
kIgnoreMuon2018 = Cut(kIgnoreMuon2018)

# kIgnoreMuon2018 has to come first here before other cuts. 
# Otherwise the maximum muon id won't be taken over all prongs in the slice but just on prongs that pass some cuts which would have removed muons 
kProngCuts = kIgnoreMuon2018 & kIsNeutron & kProngNhitsCut & kPhotonID2018Cut 

# prong level var
kProtonID = Var(lambda tables: abs(tables['rec.vtx.mlvertex.fuzzyk.png.cvnpart']['protonid']))

limit = 100
filelist = '/pnfs/nova/persistent/users/karlwarb/HDF5-Training-19-02-26/ND-ProngCVN-RHC/*.h5'
tablesProng = pa.loader(filelist,limit=limit, index=KLP)
tablesSlc = pa.loader(filelist, limit=limit)
# associate loads the same data for tablesProng and tablesSlc just once
a = pa.associate([tablesProng, tablesSlc])

# get dataframe for kProngDispl first
specDummySlc = pa.spectrum(tablesSlc, kNDRockFilter & kNumuContainND, kProngDispl)
# get dataframe for prong level var
specProng = pa.spectrum(tablesProng, kProngCuts, kProtonID)

# go, go, go
a.Go()

prongdispldf = specDummySlc.df()
vardf = specProng.df()
print(prongdispldf.head())
print(vardf.head())
# this accomplishes two things at once
# prongdispldf has all prong displacements passing slice level cuts and displacement > 20
# doing pd.concat([prongdispldf, vardf], axis=1, join='inner') therefore only selects rows of vardf that exist in prongdispldf when we use join='inner' 
finaldf = pd.concat([prongdispldf, vardf], axis=1, join='inner')[vardf.name]

# create a spectrum object out of finaldf
finalspec = pa.filledSpectrum(finaldf, specProng.POT())

# make the prong level var plot
n, bins = finalspec.histogram(bins=25, range=(0,1))

print('Selected ' + str(n.sum()) + ' events from ' + str(finalspec.POT()) + ' POT.')

plt.hist(bins[:-1], bins=bins, weights=n, histtype='step', color='blue', label='ND RHC')
plt.xlabel('Proton ID 2018')
plt.ylabel('Prongs')

plt.legend(loc='best')

plt.show()
