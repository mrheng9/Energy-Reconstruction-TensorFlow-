#!/usr/bin/env python
import h5py
import os
import sys
sys.path.append('../..')

import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore warnings about nan and divide by zero
import pandas as pd

from PandAna.core.core import *
from PandAna.cut.analysis_cuts import *
from PandAna.reco_validation.prod5_pid_validation import *

import matplotlib 
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12,
                            'patch.linewidth': 1})

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

datasets = ['ND_Data_FHC',
            'ND_Data_RHC',
            'ND_Mont_FHC',
            'ND_Mont_RHC']    
data_mc_pairs = [{'data': 'ND_Data_FHC',
                  'mc'  : 'ND_Mont_FHC',
                  'horn':  'FHC'},
                 {'data': 'ND_Data_RHC',
                  'mc'  : 'ND_Mont_RHC',
                  'horn': 'RHC'}]
KLP = KL + ['rec.vtx.mlvertex.fuzzyk.png_idx']

# currently only some of the files have been evaluated so read from the list of 
# such files
def get_files_nd(dataset_name):
    dataset_name = dataset_name.replace('_', '-')
    d = '/lfstev/nnet/R19-02-23-miniprod5/' + dataset_name
    if 'Data' in dataset_name:
        print('Retrieving {} files from {}/{}'.format(dataset_name, d, 'evaled.txt'))
        return [line.rstrip('\n') for line in open(d + '/evaled.txt')]
    else:
        return [os.path.join(d, f) for f in os.listdir(d) if 'h5caf.h5' in f]

# only want to look at the networks that correspond to the beam mode of the files
# not the best way to do this but it works
def which_prod5pid(pid, dataset_name):
    ret = pid
    if 'RHC' in dataset_name:
        ret = ret + '_prod5rhc'
    elif 'FHC' in dataset_name:
        ret = ret + '_prod5fhc'
    return ret

# main function
def prod5_pid_data_mc(limit, stride, offset, spectra_file, output):        
    testing_datasets = [datasets[3]]
    if spectra_file is None:
        loaders = {}
        slc_tables = {}  # loader for slice level cuts only
        prong_tables = {} # loader for prong level cuts only 
        for dataset_name in datasets:
            slc_tables[dataset_name] = loader(get_files_nd(dataset_name), 
                                              stride=stride,
                                              limit=limit,
                                              offset=offset)
            prong_tables[dataset_name] = loader(get_files_nd(dataset_name), 
                                                stride=stride,
                                                limit=limit,
                                                offset=offset, index=KLP)
            loaders[dataset_name] = associate([slc_tables[dataset_name], 
                                               prong_tables[dataset_name]]) # read same data only once for both loaders

    
        specs = {}
        slc_specs = {}
        save_specs = []
        save_labels = []
        for cut_name, cut in cut_levels.items():
            slc_specs[cut_name] = {}
            specs[cut_name] = {}
            for dataset_name in datasets:
                slc_specs[cut_name][dataset_name] \
                    = spectrum(slc_tables[dataset_name], cut, kCaloE)  # use dummy var for slice spectra with slice cuts
                specs[cut_name][dataset_name] = {}
                for pid in pids:
                    for var_name in [pid+'_prod4', which_prod5pid(pid, dataset_name)]:
                        specs[cut_name][dataset_name][var_name] = \
                            spectrum(prong_tables[dataset_name], kProngCuts, pid_scores[var_name]) # apply only particle truth cuts to prong level var


        for loader_name, load in loaders.items(): 
            load.Go() # go, go, go 

        filename = output + '/prod5_pid_data_mc_spectra'
        if stride:
            filename += '_s{}'.format(stride)
        if limit:
            filename += '_l{}'.format(limit)
        if offset:
            filename += '_o{}'.format(offset)
        filename += '.hdf5'
        for cut_name, cut in cut_levels.items():
            for dataset_name in datasets:
                for pid in pids:
                    for var_name in [pid+'_prod4', which_prod5pid(pid, dataset_name)]:

                        # get prong dataframe
                        df_prong = specs[cut_name][dataset_name][var_name].df()
                        df_weight = specs[cut_name][dataset_name][var_name]._weight
                        # apply slice cuts to prong dataframe
                        df_prong = ApplySlcCutsPngDF(df_prong, slc_specs[cut_name][dataset_name].df())
                        df_weight = ApplySlcCutsPngDF(df_weight, slc_specs[cut_name][dataset_name].df())
                        # reset prong spectrum with new dataframe
                        specs[cut_name][dataset_name][var_name]._df = df_prong
                        specs[cut_name][dataset_name][var_name]._weight = df_weight
                        # save
                        save_specs.append(specs[cut_name][dataset_name][var_name])
                        save_labels.append('{}_{}_{}'.format(cut_name, var_name, dataset_name))


        save_spectra(filename,
                     save_specs,
                     save_labels)

    else:
        print('Loading spectra from {}'.format(spectra_file))

        specs = {}
        pid_score_names = list(pid_scores.keys())
        for cut_name, cut in cut_levels.items():
            specs[cut_name] = {}
            for dataset_name in datasets:
                specs[cut_name][dataset_name] = {}
                for pid in pids:
                    for var_name in [pid+'_prod4', which_prod5pid(pid, dataset_name)]:
                        spec_name = '{}_{}_{}'.format(cut_name, var_name, dataset_name)
                        specs[cut_name][dataset_name][var_name] = load_spectra(spectra_file, 
                                                                               spec_name)
                                
    if spectra_file:    
        for cut_name, cut in cut_levels.items():
            for data_mc in data_mc_pairs:
                for pid in pids:
                    for var_name in [pid+'_prod4', which_prod5pid(pid, data_mc['data'])]:
                        #var_name = which_prod5pid(pid, data_mc['data'])          
                        plot_data_mc(specs[cut_name][data_mc['data']][var_name],
                                     specs[cut_name][data_mc['mc']][var_name],
                                     30, pid, 'Prongs', 
                                     '{} {} {}'.format(cut_name, data_mc['horn'], var_name),
                                     'data_mc_plots_technote/data_mc_ND_{}_{}_{}'.format(data_mc['horn'],
                                                                                            cut_name,
                                                                                            var_name))
                        
                    
def ratio(data, mc, nbins, pot, binrange=(0, 1)):
    h1, bins1 = data.histogram(bins=nbins, range=binrange, POT=pot)
    h2, bins2 = mc.histogram(bins=nbins, range=binrange, POT=pot)
    
    # calculate statistical error on the ratio
    bin_centers = (bins1[:-1] + bins1[1:])/2
    err = np.sqrt(1/h2 + 1/h1) * h1 / h2
    return h1 / h2, bins1, err, bin_centers


def plot_data_mc(data, mc, nbins, xlabel, ylabel, title, name, logy=True):
    pot = data.POT()
    fig, ax = plt.subplots(2, sharex=True, gridspec_kw = {'hspace': 0, 'height_ratios': [3, 1]})
    
    for spec, color, label in zip([data, mc], ['k', 'r'], ['data', 'mc']):
        n, bins = spec.histogram(nbins, range=(0,1), POT=pot)
        ax[0].hist(bins[:-1], bins, weights=n, histtype='step', color=color, label=label)

    one_x = np.linspace(0, 1)
    one_y = [1 for _ in range(len(one_x))]
    ax[1].plot(one_x, one_y, color='k', linestyle='dashed')
    nratio, binsratio, err, bin_centers = ratio(data, mc, nbins, pot)
    ax[1].hist(binsratio[:-1], binsratio, weights=nratio, histtype='step', color='k')
    ax[1].errorbar(bin_centers, nratio, yerr=err, ecolor='k', fmt='k.')
    
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Data / MC')
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0.5, 1.5])
    ax[0].set_xlim([0, 1])
    ax[0].set_ylabel(ylabel)
    ax[0].set_yscale('log')

    ax[0].legend(loc='best')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    fig.suptitle(title)
    fig.savefig(name + '.png')
    print('Created {}.png'.format(name))
    plt.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Plot prod5 and prod4 PID distributions')
    parser.add_argument('--limit', '-l', default=None, type=int,
                        help='Limit the number of files to process')
    parser.add_argument('--stride', '-s', default=1, type=int,
                        help='Set stride')
    parser.add_argument('--offset', '-o', default=0, type=int,
                        help='Set file offset')
    parser.add_argument('--spectra_file', '-f', default=None,
                        help='Only plot spectra previously filled by this application')
    parser.add_argument('--output', default='.', type=str,
                        help='Output directory')
    args = parser.parse_args()

    prod5_pid_data_mc(args.limit, args.stride, args.offset, 
                      args.spectra_file, args.output)
    
