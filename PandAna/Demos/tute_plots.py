from PandAna import *

import os
import sys
import numpy as np
import pandas as pd
import argparse
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# function to plot the difference in PID scores for the two CVNs
# has the ability to plot for multiple cuts in another variable such as true energy, true y etc
def MakeDeltaCVNPlots(sig, title='test', color='blue', slc={}, folder='.'):
    var = 'cvnloosepreselptp'
    xspan = (-1, 1)
    nbins = 40
    pot = 1.2e21
    
    sig_df = sig.df()
    wgt = sig._weight
    
    key = ''
    val = [None]
    if len(slc):
        val = slc.values()[0]
        key = slc.keys()[0]
    fig, ax = plt.subplots()
    colors = [plt.cm.tab10(i) for i in range(max(len(val)-1,1))]
    for i in range(max(len(val)-1, 1)):
        if val[i] == None:
            df = sig_df[var] - sig_df['cvnoldpresel']
        else:
            df = sig_df[var][(sig_df[key] >= val[i]) & (sig_df[key] >= val[i+1])] - \
                 sig_df['cvnoldpresel'][(sig_df[key] >= val[i]) & (sig_df[key] >= val[i+1])]
            
            wgt = sig._weight[(sig_df[key] >= val[i]) & (sig_df[key] >= val[i+1])]
        n, bins = np.histogram(df, nbins, xspan, weights=wgt)
        n *= pot/sig.POT()
        n = n/sum(n)
        label = 'No cut'
        if val[i] != None:
            label = '%s >= %.1f & %s < %.1f' % (key, val[i], key, val[i+1])
        ax.hist(bins[:-1], bins, weights=n, histtype='step', color=colors[i],label=label)
        ax.set_xlabel('cvnloosepreselptp - cvnoldpresel')
        ax.set_ylabel('Events')
        
    fig.suptitle(title)
    plt.legend(loc='upper left')
    plt.show()
    if not os.path.isdir(folder):
        os.mkdir(folder)
    if len(slc):
      fig.savefig('%s/%s_delta_with_%s.pdf'%(folder, title, key))
    else:
      fig.savefig('%s/%s_delta_all.pdf'%(folder, title))

# calculate efficiency and background contaminations for a range of cvn cuts
def CalculateEffPur(sig, bkg, bins):
    sig_tot = sum(sig)
    bkg_tot = sum(bkg)
    width = bins[1]-bins[0]
    nbins = len(bins)-1
    sig_eff = np.array([])
    bkg_rej = np.array([])
    for i in range(nbins):
        sig_sel = sum(sig[i:nbins])
        bkg_sel = sum(bkg[i:nbins])
        sig_eff = np.append(sig_eff, sig_sel/sig_tot)
        bkg_rej = np.append(bkg_rej, bkg_sel/(sig_sel+bkg_sel))
    return sig_eff, bkg_rej

# calculate efficiency and background contamination for the loosepresel optimized prod5 cvn cut
def PrevCVNCut(df_sig, wgt_sig, df_bkg, wgt_bkg, index):
    sig = pd.concat([df_sig, wgt_sig], axis=1)
    bkg = pd.concat([df_bkg, wgt_bkg], axis=1)
    val = 0.84  # prod5 tuned cut
    sig = sig[sig['cvnloosepreselptp'] >= val]
    bkg = bkg[bkg['cvnloosepreselptp'] >= val]
    eff = sig['weight'].sum()/wgt_sig.sum()
    bkg_frac = bkg['weight'].sum()/(sig['weight'].sum()+bkg['weight'].sum())
    return eff, bkg_frac

# function to plot the CVN distributions along with the ROC curves
def MakeCVNDistPlot(sig, bkg, title = 'test', folder='.'):
    var = ['cvnoldpresel', 'cvnloosepreselptp']
    colors = [plt.cm.tab10(i+2) for i in range(len(var))]
    
    sig_weight = sig._weight
    bkg_weight = bkg._weight
    
    sigy, bkgy = [], []
    sig_effs, bkg_rejs = [], []
    x = None
    xspan = (0, 1)
    nbins = 20
    pot = 1.2e21
    numbers = "Signal : %0.2f, Bkg: %0.2f" % (sig.integral(pot), bkg.integral(pot))
    # check effeciency and background contamination (1-signal purity)
    prev_eff, prev_bkgfrac = PrevCVNCut(sig.df(), sig_weight, bkg.df(), bkg_weight, title)
    for v in var:
        nsig, sig_bins = np.histogram(sig.df()[v], nbins, xspan, weights=sig_weight)
        nbkg, bkg_bins = np.histogram(bkg.df()[v], nbins, xspan, weights=bkg_weight)
        nsig *= pot/sig.POT() 
        nbkg *= pot/bkg.POT()
        sig_eff, bkg_rej = CalculateEffPur(nsig, nbkg, sig_bins)
        sig_effs.append(sig_eff)
        bkg_rejs.append(bkg_rej)
        sigy.append(nsig)
        bkgy.append(nbkg)
        x = sig_bins
    fig, ax = plt.subplots()
    for i in range(len(sigy)):
        ax.hist(x[:-1], x, weights=sigy[i], histtype='step', color=colors[i], label=var[i])
        ax.hist(x[:-1], x, weights=bkgy[i], histtype='step', linestyle='dashed', color=colors[i])
    ax.set_title(title)
    ax.set_xlabel('CVN')
    ax.set_ylabel('Events')
    ax.set_yscale('log')
    plt.legend(loc='upper center')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fig.savefig('%s/%s_dist.pdf'%(folder,title))
   
    fig, ax = plt.subplots()
    for i in range(len(sig_effs)):
        ax.plot(bkg_rejs[i], sig_effs[i], color=colors[i], label=var[i])
    ax.axhline(y=prev_eff, linestyle='dashed', color='black')
    ax.axvline(x=prev_bkgfrac, linestyle='dashed', color='black')
    ax.set_title(numbers)
    ax.set_ylabel('Signal Efficiency')
    ax.set_xlabel('Background Contamination')
    plt.legend(loc='best')
    fig.savefig('%s/%s_roc.pdf'%(folder,title))
