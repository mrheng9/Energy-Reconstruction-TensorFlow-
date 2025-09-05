from PandAna import *
from PandAna.var.analysis_vars import *
from PandAna.cut.analysis_cuts import *
from PandAna.utils.enums import *

import os
import sys
import numpy as np
import pandas as pd
import argparse
import re

# your definition to run over
definition = 'def_snapshot karlwarb-MLTute2019-HDF5_R19-11-18-prod5reco.n_fluxswap' 

# define PID vars as two column dataframes for two CVNs
var = ['rec.sel.cvnloosepreselptp', 'rec.sel.cvnoldpresel']
def kNewCVNe(tables):
  df = []
  for v in var:
    df.append(tables[v]['nueid'])
  ret = pd.concat(df, axis=1)
  ret.columns = [re.sub(r'rec.sel.', '', v) for v in var]
  return ret
kNewCVNe = Var(kNewCVNe)

def kNewCVNm(tables):
  df = []
  for v in var:
    df.append(tables[v]['numuid'])
  ret = pd.concat(df, axis=1)
  ret.columns = [re.sub(r'rec.sel.', '', v) for v in var]
  return ret
kNewCVNm = Var(kNewCVNm)

def kNewCVNnc(tables):
  df = []
  for v in var:
    df.append(tables[v]['ncid'])
  ret = pd.concat(df, axis=1)
  ret.columns = [re.sub(r'rec.sel.', '', v) for v in var]
  return ret
kNewCVNnc = Var(kNewCVNnc)

cvn_vars = {}
cvn_vars['nueid'] = kNewCVNe
cvn_vars['numuid'] = kNewCVNm
cvn_vars['ncid'] = kNewCVNnc

# define a few more truth vars
kInelasticity = Var(lambda tables: tables['rec.mc.nu']['y'])
kTrueE = Var(lambda tables: tables['rec.mc.nu']['E'])

# dumb oscillation weights to apply to spectra
def kWoscDumb(tables, weight):
  sel = (tables['rec.mc']['nnu'] == 1)
  weight[sel] *= tables['rec.mc.nu']['woscdumb']
  return weight

# now define some truth cuts
kPDG = Var(lambda tables: tables['rec.mc.nu']['pdg'])
kPDGAbs = Var(lambda tables: abs(tables['rec.mc.nu']['pdg']))
kMode = Var(lambda tables: tables['rec.mc.nu']['mode'])
kIsCC = Cut(lambda tables: tables['rec.mc.nu']['iscc'] == 1)
kIsNC = Cut(lambda tables: tables['rec.mc.nu']['iscc'] == 0)

# function to return signal and background cuts depending on which analysis PID we're looking at
def SigCuts(idx='nueid'):
  cut = Cut(lambda tables: tables['rec.mc']['nnu'] == 1)
  if 'nue' in idx:
    cut = cut & (kPDGAbs == 12) & kIsCC
  if 'numu' in idx:
    cut = cut & (kPDGAbs == 14) & kIsCC
  if 'nc' in idx:
    cut = cut & (~kIsCC)
  return cut

def BkgCuts(idx='nueid'):
  cut = Cut(lambda tables: tables['rec.mc']['nnu'] == 1)
  if 'nue' in idx:
    cut = cut & ((kPDGAbs != 12) | ~kIsCC)
  if 'numu' in idx:
    cut = cut & ((kPDGAbs != 14) | ~kIsCC)
  if 'nc' in idx:
    cut = cut & (~kIsNC) 
  return cut

# some basic preselections for each analysis
kNueFDSel = kNueProngContainment & kVeto

kNumuFDSel = kNumuBasicQuality & kNumuContainFD

kNusFDSel = kNusFDContain & kVeto

ids = ['nueid', 'numuid', 'ncid']

sel_cuts = {}
sel_cuts['nueid'] = kNueFDSel
sel_cuts['numuid'] = kNumuFDSel
sel_cuts['ncid'] = kNusFDSel

truth_cuts = {}
for idx in ids:
  truth_cuts[idx] = {}
  truth_cuts[idx]['sig'] = SigCuts(idx)
  truth_cuts[idx]['bkg'] = BkgCuts(idx)

# also split sample by mode
def ModeCuts(mode=-1):
  cut = Cut(lambda tables: tables['rec.mc']['nnu'] == 1)
  if mode >= 0:
    cut = cut & (kMode == mode)
  return cut

mode_cuts = {}
mode_cuts['kAll'] = ModeCuts(-1)
mode_cuts['kQE']  = ModeCuts(mode.kQE)
mode_cuts['kRes'] = ModeCuts(mode.kRes)
mode_cuts['kDIS'] = ModeCuts(mode.kDIS)

# some helpful arguments for definition
parser = argparse.ArgumentParser(
    description='Save CVN distributions for various truth cuts')
parser.add_argument(
    '--limit', help=('Limit of files to run: '),
     type=int, default=None)
parser.add_argument(
    '--stride', help=('stride of files to run: '),
     type=int, default=1)
parser.add_argument(
    '--offset', help=('offset of files to run: '),
     type=int, default=0)
parser.add_argument(
    '--plot', help=('Run plotting part of the code '),
    action='store_true', default=False)

args = parser.parse_args()

filename = 'cvn_truth_dist.h5'
# initialize loader
tables = loader(definition, limit=args.limit, stride=args.stride, offset=args.offset)

# create PID, trueE and trueY spectra for each cut above and apply dumb oscillations
specs = {}
for selid, sel_cut in sel_cuts.items():
  for ptype, pcut in truth_cuts[selid].items():
    for mtype, mcut in mode_cuts.items():
      cut = sel_cut & pcut & mcut
      specid = 'cvn_%s_%s_%s' % (selid, ptype, mtype)
      specs[specid] = spectrum(tables, cut, cvn_vars[selid], weight=kWoscDumb)
      trueEid = 'trueE_%s_%s_%s' % (selid, ptype, mtype)
      specs[trueEid] = spectrum(tables, cut, kTrueE, weight=kWoscDumb)
      trueYid = 'trueY_%s_%s_%s' % (selid, ptype, mtype)
      specs[trueYid] = spectrum(tables, cut, kInelasticity, weight=kWoscDumb)

if not args.plot:
  # Run!
  tables.Go()
  # save spectra to HDF5 output
  save_tree(filename, specs.values(), specs.keys())

# plot
else:
  from tute_plots import *
  # load from file
  specs = load_tree(filename, specs.keys())
  combined = {}
  for selid, sel_cut in sel_cuts.items():
    for ptype, pcut in truth_cuts[selid].items():
      for mtype, mcut in mode_cuts.items():
        cid = '%s_%s_%s' % (selid, ptype, mtype)
        cvnid = 'cvn_%s_%s_%s' % (selid, ptype, mtype)
        trueEid = 'trueE_%s_%s_%s' % (selid, ptype, mtype)
        trueYid = 'trueY_%s_%s_%s' % (selid, ptype, mtype)
        
        # combine dataframes
        combined[cid] = specs[cvnid] & specs[trueEid] & specs[trueYid]

  # Make some plots
  MakeCVNDistPlot(combined['nueid_sig_kAll'], combined['nueid_bkg_kAll'], title='nueid', folder='plots')
  MakeDeltaCVNPlots(combined['nueid_sig_kAll'], title='nueid', folder='plots')
  MakeDeltaCVNPlots(combined['nueid_sig_kAll'], title='nueid', slc={'E':[0, 1, 2, 3, 4, 5]},folder='plots')
  MakeDeltaCVNPlots(combined['nueid_sig_kAll'], title='nueid', slc={'y':[0., 0.2, 0.4, 0.6, 0.8]},folder='plots')
