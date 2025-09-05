import pandas as pd
import numpy as np
from PandAna.core.core import KL, KLN, Var

#KLP = KL + ['rec.vtx.mlvertex.fuzzyk.png_idx']


def kHitPos(tables):
    return tables['hits'][['x','y','z']]
kHitX = Var(kHitPos)

def kHitTime(tables):
    return tables['hits']['t']
kHitTime = Var(kHitTime)

def kHitCharge(tables):
    return tables['hits']['pe']
kHitCharge = Var(kHitCharge)

def kHitCorrCharge(tables):
    return tables['hits']['pe_corr']
kHitCorrCharge = Var(kHitCorrCharge)

def kHitPlane(tables):
    return tables['hits']['plane']
kHitPlane = Var(kHitPlane)

def kHitView(tables):
    return tables['hits']['view']
kHitView = Var(kHitView)

def kHitCell(tables):
    return tables['hits']['cell']
kHitCell = Var(kHitCell)

def kHitID(tables):
    return tables['hits']['hit_id']
kHitID = Var(kHitID)

def kEdepHitID(tables):
    return tables['edeps']['hit_id']
kEdepHitID = Var(kEdepHitID)

def kEdepE(tables):
    return tables['edeps']['energy']
kEdepE = Var(kEdepE)

def kEdepG4ID(tables):
    return tables['edeps']['g4_id']
kEdepG4ID = Var(kEdepG4ID)

def kPartG4ID(tables):
    return tables['particles']['g4_id']
kPartG4ID = Var(kPartG4ID)

def kPartType(tables):
    return tables['particles']['type']
kPartType = Var(kPartType)
