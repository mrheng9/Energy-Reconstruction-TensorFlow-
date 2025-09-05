import pandas as pd
import numpy as np
from PandAna.core.core import KL, Var

#KLP = KL + ['rec.vtx.mlvertex.fuzzyk.png_idx']

def kSliceInter(tables):
    return tables['rec.training.trainingdata']['interaction']
kSliceInter = Var(kSliceInter)

def kSliceMap(tables):
    return tables['rec.training.cvnmaps']['cvnmap']
kSliceMap = Var(kSliceMap)

def kSliceNuEnergy(tables):
    return tables['rec.training.trainingdata']['nuenergy']
kSliceNuEnergy = Var(kSliceNuEnergy)

def kNuERegCVN(tables):
    return tables['rec.energy.nue']['regcvnEvtE']
kNuERegCVN = Var(kNuERegCVN)

def kRecoVtx(tables):
    return tables['rec.vtx.mlvertex'][['vtx.x', 'vtx.y', 'vtx.z']]
kRecoVtx = Var(kRecoVtx)

def kProngMap(tables):
    return tables['rec.vtx.mlvertex.fuzzyk.png.cvnmaps']['cvnmap']
kProngMap = Var(kProngMap)

def kProngEnergy(tables):
    return tables['rec.vtx.mlvertex.fuzzyk.png.truth']['p.E']
kProngEnergy = Var(kProngEnergy)

def kProngRecE(tables):
    return tables['rec.vtx.mlvertex.fuzzyk.png.prongtrainingdata']['rece']
kProngRecE = Var(kProngRecE)

def kElectronRegCVN(tables):
    return tables['rec.vtx.mlvertex.fuzzyk.png.regcvn']['prongE']
kElectronRegCVN = Var(kElectronRegCVN)

def kProng3Momentum(tables):
    return tables['rec.vtx.mlvertex.fuzzyk.png.truth'][['p.px','p.py','p.pz']]
kProng3Momentum = Var(kProng3Momentum)

def kProngPrimary(tables):
    return tables['rec.vtx.mlvertex.fuzzyk.png.prongtrainingdata']['isprimary']
kProngPrimary = Var(kProngPrimary)

def kProngLabel(tables):
    return tables['rec.vtx.mlvertex.fuzzyk.png.prongtrainingdata']['label3d']
kProngLabel = Var(kProngLabel)

"""
def kElectronMap(tables):
    df = tables['rec.vtx.mlvertex.fuzzyk.png.cvnmaps']['cvnmap']
    primarydf = tables['rec.vtx.mlvertex.fuzzyk.png.prongtrainingdata'][['isprimary', 'label3d', 'rece']]

    df = df.where((primarydf['label3d'] == 0) & (primarydf['isprimary'] == 1))
    
    ret = pd.concat([df, primarydf['rece']], axis=1)
    return ret.groupby(level=KL, group_keys=False).apply(lambda x: x.loc[x['rece'] == x['rece'].max(), ['cvnmap']])
kElectronMap = Var(kElectronMap)

def kElectronEnergy(tables):
    dfEnergy = tables['rec.vtx.mlvertex.fuzzyk.png.truth']['p.E']
    primarydf = tables['rec.vtx.mlvertex.fuzzyk.png.prongtrainingdata'][['isprimary', 'label3d', 'rece']]

    dfEnergy = dfEnergy.where((primarydf['label3d'] == 0) & (primarydf['isprimary'] == 1))
    
    ret = pd.concat([dfEnergy, primarydf['rece']], axis=1)
    return ret.groupby(level=KL, group_keys=False).apply(lambda x: x.loc[x['rece'] == x['rece'].max(), ['p.E']])
kElectronEnergy = Var(kElectronEnergy)

def kElectron3Momentum(tables):
    dfEnergy = tables['rec.vtx.mlvertex.fuzzyk.png.truth'][['p.px','p.py','p.pz']]
    primarydf = tables['rec.vtx.mlvertex.fuzzyk.png.prongtrainingdata'][['isprimary', 'label3d', 'rece']]

    dfEnergy = dfEnergy.where((primarydf['label3d'] == 0) & (primarydf['isprimary'] == 1))
    
    ret = pd.concat([dfEnergy, primarydf['rece']], axis=1)
    return ret.groupby(level=KL, group_keys=False).apply(lambda x: x.loc[x['rece'] == x['rece'].max(), ['p.px', 'p.py', 'p.pz']])
kElectron3Momentum = Var(kElectron3Momentum)

def kElectronRecE(tables):
    primarydf = tables['rec.vtx.mlvertex.fuzzyk.png.prongtrainingdata'][['isprimary', 'label3d', 'rece']]
   
    dfEnergy = primarydf['rece'][(primarydf['label3d'] == 0) & (primarydf['isprimary'] == 1)].groupby(level=KL, group_keys=False).agg(np.max)
    return dfEnergy
kElectronRecE = Var(kElectronRecE)
"""
