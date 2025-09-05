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

import matplotlib 
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12,
                            'patch.linewidth': 1})

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

KLP = KL + ['rec.vtx.mlvertex.fuzzyk.png_idx']

def get_files(det, swap_str, horn_current):
    d = '/lfstev/nnet/R19-02-23-miniprod5/'
    if 'FD' in det:
        d = d + 'FD-' + swap_str + "-" + horn_current + '-Eval/'
    else:
        d = d + 'ND-Mont-' + horn_current + '/'
    return [os.path.join(d, f) for f in os.listdir(d) if 'h5caf.h5' in f]

# workaround for applying slice cuts such as preselections to prong level spectra
def ApplySlcCutsPngDF(png_df, slc_df):
    png_name = png_df.name
    slc_name = slc_df.name

    # reset to slice level index
    df = png_df.reset_index().set_index(KL)
   
    val = df[png_name]  # get var column
    val = val.groupby(level=KL).agg(list)  # each row contains list of prong values

    # slc_df has indices that pass slice cuts. concat with prong column. join='inner' drops rows in prong column that don't pass slice cuts
    # drop slice var column
    val = pd.concat([val, slc_df], axis=1, join='inner').drop(slc_name, axis=1)
    
    # if no prongs pass prong and slice cuts, return an empty df
    if val.empty:
        val = pd.DataFrame([], columns=[png_name])

    else:
        # unlist each row
        val = val[png_name].apply(pd.Series).stack()
        val = val.reset_index()
        val.columns = KL + ['idx', png_name]
        # set new index 
        val = val.set_index(KL + ['idx'])

    # contains prong dataframe that has slice cuts applied
    return val[png_name]

# selection cuts
## FD check
def kIsFarDet(tables):
    query = tables._files.query
    if not type(query) is list: query = [query]
    return 'fardet' in query[0]

############### nue cuts ######################
def kNueContain(tables):
    if kIsFarDet(tables):
        return kNueProngContainment(tables) & kNueBasicPart(tables)
    else:
        return kNueNDContain(tables)

kNueContain = Cut(kNueContain)

def kNuePresel(tables):
    if kIsFarDet(tables):
        return kNueCorePresel(tables)
    else:
        return kNueNDPresel(tables)

kNuePresel = Cut(kNuePresel)

############### numu cuts #####################
# kCCE isn't working yet
kNumuNoPIDNoCCEFD = kNumuBasicQuality & kNumuContainFD
kNumuNoPIDNoCCEND = kNumuBasicQuality & kNumuContainND
def kNumuContain(tables):
    if kIsFarDet(tables):
        return kNumuContainFD(tables)
    else:
        return kNumuContainND(tables)
        
kNumuContain = Cut(kNumuContain)

def kNumuPresel(tables):
    if kIsFarDet(tables):
        return kNumuNoPIDNoCCEFD(tables)
    else:
        return kNumuNoPIDNoCCEND(tables)

kNumuPresel = Cut(kNumuPresel)

############### nus cuts ######################
def kNusContain(tables):
    if kIsFarDet(tables):
        return kNusFDContain(tables)
    else:
        return kNusNDContain(tables)
kNusContain = Cut(kNusContain)

def kNusPresel(tables):
    if kIsFarDet(tables):
        return kNusFDPresel(tables)
    else:
        return kNusNDPresel(tables)
kNusPresel = Cut(kNusPresel)

############### ORd cuts #####################
kCosVeto        = kVeto
kOrContainment  = kNumuContain | kNusContain | kNueContain
kOrPreselection = kNumuPresel  | kNusPresel  | kNuePresel

# prong quality cuts
kHas2020Score = Cut(lambda tables: tables['rec.vtx.mlvertex.fuzzyk.png.cvnpart2020fhc']['muonid'] > 0)
kHas2018Score = Cut(lambda tables: tables['rec.vtx.mlvertex.fuzzyk.png.cvnpart']['muonid'] > 0)
kProngLength = Cut(lambda tables: tables['rec.vtx.mlvertex.fuzzyk.png']['len'] < 500)

kProngCuts = kProngLength & kHas2020Score & kHas2018Score

# particle truth cuts
kIsChargedPion = Cut(lambda tables: abs(tables['rec.vtx.mlvertex.fuzzyk.png.truth']['pdg']) == 211)
kIsPhoton = Cut(lambda tables: tables['rec.vtx.mlvertex.fuzzyk.png.truth']['pdg'] == 22)
kIsNeutron = Cut(lambda tables: abs(tables['rec.vtx.mlvertex.fuzzyk.png.truth']['pdg']) == 2112)
kIsProton = Cut(lambda tables: abs(tables['rec.vtx.mlvertex.fuzzyk.png.truth']['pdg']) == 2212)
kIsElectron = Cut(lambda tables: abs(tables['rec.vtx.mlvertex.fuzzyk.png.truth']['pdg']) == 11)
kIsMuon = Cut(lambda tables: abs(tables['rec.vtx.mlvertex.fuzzyk.png.truth']['pdg']) == 13)

# PID vars prod5
def kProtonPIDProd5FHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020fhc')]
    return cvn_png_df['protonid']
def kProtonPIDProd5RHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020rhc')]
    return cvn_png_df['protonid']

def kPhotonPIDProd5FHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020fhc')]
    return cvn_png_df['photonid']
def kPhotonPIDProd5RHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020rhc')]
    return cvn_png_df['photonid']

def kPionPIDProd5FHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020fhc')]
    return cvn_png_df['pionid']
def kPionPIDProd5RHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020rhc')]
    return cvn_png_df['pionid']

def kElectronPIDProd5FHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020fhc')]
    return cvn_png_df['electronid']
def kElectronPIDProd5RHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020rhc')]
    return cvn_png_df['electronid']

def kMuonPIDProd5FHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020fhc')]
    return cvn_png_df['muonid']
def kMuonPIDProd5RHC(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart2020rhc')]
    return cvn_png_df['muonid']

def kEMPIDProd5FHC(tables):
    emid_df = kElectronPIDProd5FHC(tables) + kPhotonPIDProd5FHC(tables)
    emid_df.name = 'emid' # name the series, we'll need it later
    return emid_df
def kEMPIDProd5RHC(tables):
    emid_df = kElectronPIDProd5RHC(tables) + kPhotonPIDProd5RHC(tables)
    emid_df.name = 'emid' # name the series, we'll need it later
    return emid_df


# PID vars prod4
def kProtonPID(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart')]
    return cvn_png_df['protonid']

def kPhotonPID(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart')]
    return cvn_png_df['photonid']

def kPionPID(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart')]
    return cvn_png_df['pionid']

def kElectronPID(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart')]
    return cvn_png_df['electronid']

def kMuonPID(tables):
    cvn_png_df = tables[('rec.vtx.mlvertex.fuzzyk.png.cvnpart')]
    return cvn_png_df['muonid']

def kEMPID(tables):
    emid_df = kElectronPID(tables) + kPhotonPID(tables)
    emid_df.name = 'emid' # name the series, we'll need it later
    return emid_df

# fhc trained networks
kPhotonPIDProd5FHC   = Var(kPhotonPIDProd5FHC)
kProtonPIDProd5FHC   = Var(kProtonPIDProd5FHC)
kPionPIDProd5FHC     = Var(kPionPIDProd5FHC)
kElectronPIDProd5FHC = Var(kElectronPIDProd5FHC)
kMuonPIDProd5FHC     = Var(kMuonPIDProd5FHC)
kEMPIDProd5FHC       = Var(kEMPIDProd5FHC)
# rhc trained networks
kPhotonPIDProd5RHC   = Var(kPhotonPIDProd5RHC)
kProtonPIDProd5RHC   = Var(kProtonPIDProd5RHC)
kPionPIDProd5RHC     = Var(kPionPIDProd5RHC)
kElectronPIDProd5RHC = Var(kElectronPIDProd5RHC)
kMuonPIDProd5RHC     = Var(kMuonPIDProd5RHC)
kEMPIDProd5RHC       = Var(kEMPIDProd5RHC)

# prod4
kPhotonPID   = Var(kPhotonPID)
kProtonPID   = Var(kProtonPID)
kPionPID     = Var(kPionPID)
kElectronPID = Var(kElectronPID)
kMuonPID     = Var(kMuonPID)
kEMPID       = Var(kEMPID)


cut_levels = {'Veto'         : kCosVeto,
              'Containment'  : kCosVeto & kOrContainment,
              'Preselection' : kCosVeto & kOrContainment & kOrPreselection}

datasets   = {'FD_Nonswap_FHC'  : 'prod_h5_R19-02-23-miniprod5.n_fd_genie_default_nonswap_fhc_nova_v08_full_v1_eval',
              'FD_Fluxswap_FHC' : 'prod_h5_R19-02-23-miniprod5.n_fd_genie_default_fluxswap_fhc_nova_v08_full_v1_eval',
              'ND_Nonswap_FHC'  : 'junk',
              'ND_Nonswap_RHC'  : 'junk'}

pids = ['photonid', 'protonid', 'pionid', 'electronid', 'muonid', 'emid']


pid_scores_fhc = {'photonid_prod5fhc'     : kPhotonPIDProd5FHC,
                  'protonid_prod5fhc'     : kProtonPIDProd5FHC,
                  'pionid_prod5fhc'       : kPionPIDProd5FHC,
                  'electronid_prod5fhc'   : kElectronPIDProd5FHC,
                  'muonid_prod5fhc'       : kMuonPIDProd5FHC}
pid_scores_rhc = {'photonid_prod5rhc'     : kPhotonPIDProd5RHC,
                  'protonid_prod5rhc'     : kProtonPIDProd5RHC,
                  'pionid_prod5rhc'       : kPionPIDProd5RHC,
                  'electronid_prod5rhc'   : kElectronPIDProd5RHC,
                  'muonid_prod5rhc'       : kMuonPIDProd5RHC}

pid_scores = {'photonid_prod4'     : kPhotonPID,
              'protonid_prod4'     : kProtonPID,
              'pionid_prod4'       : kPionPID,
              'electronid_prod4'   : kElectronPID,
              'muonid_prod4'       : kMuonPID,
              'emid_prod4'         : kEMPID,
              'photonid_prod5fhc'     : kPhotonPIDProd5FHC,
              'protonid_prod5fhc'     : kProtonPIDProd5FHC,
              'pionid_prod5fhc'       : kPionPIDProd5FHC,
              'electronid_prod5fhc'   : kElectronPIDProd5FHC,
              'muonid_prod5fhc'       : kMuonPIDProd5FHC,
              'emid_prod5fhc'         : kEMPIDProd5FHC,
              'photonid_prod5rhc'     : kPhotonPIDProd5RHC,
              'protonid_prod5rhc'     : kProtonPIDProd5RHC,
              'pionid_prod5rhc'       : kPionPIDProd5RHC,
              'electronid_prod5rhc'   : kElectronPIDProd5RHC,
              'muonid_prod5rhc'       : kMuonPIDProd5RHC,
              'emid_prod5rhc'         : kEMPIDProd5RHC}

particle_cuts = {'true_pion'      : kIsChargedPion,
                 'true_photon'    : kIsPhoton,
                 'true_proton'    : kIsProton,
                 'true_electron'  : kIsElectron,
                 'true_muon'      : kIsMuon}

particle_colors = {'true_pion'      : 'darkgreen',
                   'true_pizero'    : 'chartreuse',
                   'true_photon'    : 'crimson',
                   'true_neutron'   : 'indigo',
                   'true_proton'    : 'orangered',
                   'true_electron'  : 'blueviolet',
                   'true_muon'      : 'cyan',
                   'true_em'        : 'green'}

def main(wc, limit, stride, offset, spectra_file, make_plots, output, caf):        

    if spectra_file is None:
        loaders = {}
        slc_tables = {}  # loader for slice level cuts only
        prong_tables = {} # loader for prong level cuts only 
        for dataset_name, data in datasets.items():
            if wc:
                det, swap, horn = dataset_name.split('_')
                slc_tables[dataset_name] = loader(get_files(det, swap, horn), 
                                                  stride=stride,
                                                  limit=limit,
                                                  offset=offset)
                prong_tables[dataset_name] = loader(get_files(det, swap, horn), 
                                                    stride=stride,
                                                    limit=limit,
                                                    offset=offset, index=KLP)
                loaders[dataset_name] = associate([slc_tables[dataset_name], 
                                                   prong_tables[dataset_name]]) # read same data only once for both loaders
            else:                
                query = 'defname: {}'.format(data)
                if limit is not None or stride is not None:
                    query += ' with'
                if limit is not None:
                    query += ' limit {}'.format(limit)
                if stride is not None:
                    query += ' stride {}'.format(stride)
                if offset is not None:
                    query += ' offset {}'.format(offset)
                print(query)
                slc_tables[dataset_name] = loader(query)
                prong_tables[dataset_name] = loader(query, index=KLP)
                loaders[dataset_name] = associate([slc_tables[dataset_name], prong_tables[dataset_name]])

    
        specs = {}
        slc_specs = {}
        save_specs = []
        save_labels = []
        for cut_name, cut in cut_levels.items():
            slc_specs[cut_name] = {}
            specs[cut_name] = {}
            for dataset_name, data in datasets.items():
                slc_specs[cut_name][dataset_name] \
                    = spectrum(slc_tables[dataset_name], cut, kCaloE)  # use dummy var for slice spectra with slice cuts
                specs[cut_name][dataset_name] = {}
                for particle_name, particle_cut in particle_cuts.items():
                    specs[cut_name][dataset_name][particle_name] = {}
                    for var_name, var in pid_scores.items():
                        specs[cut_name][dataset_name][particle_name][var_name] \
                                = spectrum(prong_tables[dataset_name], kProngCuts & particle_cut, var) # apply only particle truth cuts to prong level var

        for loader_name, load in loaders.items(): 
            load.Go() # go, go, go 

        filename = output + '/prod5_pid_validation_spectra'
        if stride:
            filename += '_s{}'.format(stride)
        if limit:
            filename += '_l{}'.format(limit)
        if offset:
            filename += '_o{}'.format(offset)
        filename += '.hdf5'
        for cut_name, cut in cut_levels.items():
            for dataset_name, data in datasets.items():
                for particle_name, particle_cut in particle_cuts.items():
                    for var_name in pid_scores:
                        # get prong dataframe
                        df_prong = specs[cut_name][dataset_name][particle_name][var_name].df()
                        df_weight = specs[cut_name][dataset_name][particle_name][var_name]._weight
                        # apply slice cuts to prong dataframe
                        df_prong = ApplySlcCutsPngDF(df_prong, slc_specs[cut_name][dataset_name].df())
                        df_weight = ApplySlcCutsPngDF(df_weight, slc_specs[cut_name][dataset_name].df())
                        # reset prong spectrum with new dataframe
                        specs[cut_name][dataset_name][particle_name][var_name]._df = df_prong
                        specs[cut_name][dataset_name][particle_name][var_name]._weight = df_weight
                        # save
                        save_specs.append(specs[cut_name][dataset_name][particle_name][var_name])
                        save_labels.append('{}_{}_{}_{}'.format(cut_name, particle_name, var_name, dataset_name))


        save_spectra(filename,
                     save_specs,
                     save_labels)

    else:
        print('Loading spectra from {}'.format(spectra_file))
        if caf:
            print('Loading caf results from {}'.format(caf))
            caf_file = h5py.File(caf, 'r')

        specs = {}
        pid_score_names = list(pid_scores.keys())
        for cut_name, cut in cut_levels.items():
            specs[cut_name] = {}
            for particle_name, particle_cut in particle_cuts.items():
                specs[cut_name][particle_name] = {}
                for dataset_name, data in datasets.items():
                    specs[cut_name][particle_name][dataset_name] = {}
                    for var_name in pid_scores:
                        if caf and 'prod4' in var_name:
                            specs[cut_name][particle_name][dataset_name][var_name+'caf'] = {}
                        spec_name = '{}_{}_{}_{}'.format(cut_name, particle_name, var_name, dataset_name)
                        specs[cut_name][particle_name][dataset_name][var_name] = load_spectra(spectra_file, 
                                                                                              spec_name)
                        if caf and 'prod4' in var_name:
                            specs[cut_name][particle_name][dataset_name][var_name + 'caf'] \
                                = caf_spectra(caf_file,
                                              '{}_{}_{}_{}'.format(cut_name,
                                                                   dataset_name,
                                                                   var_name.split('_')[0],
                                                                   particle_name))
        if caf: caf_file.close()
                                
                            
    if make_plots or spectra_file:
        particle_id = ['photonid',
                       'pionid',
                       'protonid',  
                       'electronid',
                       'muonid',
                       'emid']

        network_linestyles = {'prod4': '-',
                              'prod5fhc': '--',
                              'prod5rhc': '-.',
                              'prod4caf': '-'}
    
        for cut_name, cut in cut_levels.items():
            for dataset_name, data in datasets.items():
                for pid in particle_id:
                    prod4pid = pid + '_prod4'
                    prod5pidfhc = pid + '_prod5fhc'
                    prod5pidrhc = pid + '_prod5rhc'
                    prod4pid_caf = pid + '_prod4caf'
                    networks = [prod4pid, prod5pidfhc, prod5pidrhc]
                    if caf:
                        networks = networks + [prod4pid_caf]
                    for particle_name, particle_cut in particle_cuts.items():
                        plot_spectra = []
                        plot_spectra_styles = []
                        for network in networks:
                            plot_spectra_styles.append({})
                            if 'caf' in network: plot_spectra.append(specs[cut_name][particle_name][dataset_name][network])
                            else: plot_spectra.append(copy_spectrum(specs[cut_name][particle_name][dataset_name][network]))
                            
                            plot_spectra_styles[-1]['color'] = particle_colors[particle_name]
                            plot_spectra_styles[-1]['label'] = network.split('_')[-1]
                            plot_spectra_styles[-1]['linestyle'] = network_linestyles[network.split('_')[-1]]
                            if 'caf' in network:
                                plot_spectra_styles[-1]['color'] = 'springgreen'
                                plot_spectra_styles[-1]['yerr'] = True
                            else:
                                # only plot error bars for the caf hist
                                # try overriding with None
                                plot_spectra_styles[-1]['yerr'] = False
                                
                        plot_ratio(plot_spectra,
                                   30,
                                   pid,
                                   'Prongs',
                                   'pid_plots_technote/{}_{}_{}_{}'.format(cut_name, dataset_name, pid, particle_name),
                                   logy=True,
                                   spec_styles=plot_spectra_styles)

class caf_spectra:
    def __init__(self, input_file, group):
        self.contents = input_file[group + '/contents'][()]
        self.edges = input_file[group + '/edges'][()]
        self.pot = input_file[group + '/pot'][()]

    def histogram(self, bins=None, range=None, POT=None):
        if not POT: POT = self.pot
        return (self.contents * POT / self.pot, self.edges)

    def integral(self, POT=None):
        if not POT: POT = self.pot
        return self.contents.sum() * POT / self.pot

def plot(spec, nbins, xlabel, ylabel, name):
    fig, ax = plt.subplots()
    n, bins = spec.histogram(bins=nbins, range=(0, 1))
    ax.hist(bins[:-1], bins, weights=n, histtype='step')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(loc='upper right')
    fig.savefig(name + '.png')
    plt.close()

def event_count_report(label, specs, networks):
    integrals = [spec.integral() for spec in specs]
    nominal = specs[0].integral()
    print('{} | '.format(label), end='')
    for spec, label, integral in zip(specs, networks, integrals):
        perdiff = (nominal - integral) / nominal
        print('  {}: {} ({:.2f})  |'.format(label, integral, perdiff), end='')
    print('')

def plot_ratio(specs, nbins, xlabel, ylabel, plot_name, 
               logy = True, spec_styles = None, additional_handles=None):
    # assume the first entry is the one to take ratios wrt to
    if not type(specs) is list: specs = [specs]
    if spec_styles is not None:
        if not type(spec_styles) is list: spec_styles = [spec_styles]
    else:
        spec_styles = [{} for _ in range(len(specs))]
    assert len(specs) == len(spec_styles), 'Each spectrum must have a style if any are provided'
    
    pot = specs[0].POT()
    fig, ax = plt.subplots(2, sharex=True, gridspec_kw = {'hspace': 0, 'height_ratios': [3, 1]})

    one_x = np.linspace(0, 1)
    one_y = [1 for _ in range(len(one_x))]
    ax[1].plot(one_x, one_y, color='k', linestyle='dashed')
    for spec, spec_style in zip(specs, spec_styles):
        plot_ratio_err = spec_style['yerr']
        spec_style.pop('yerr')
        n, bins = spec.histogram(bins=nbins, range=(0, 1), POT=pot)
        nratio, binsratio = ratio(spec, specs[0], nbins, pot)        
            
        try:
            # matplotlib doesn't like plotting histograms with all weights = 0
            ax[0].hist(bins[:-1], bins, weights=n, histtype='step',
                       **spec_style)
            ax[1].hist(binsratio[:-1], binsratio, weights=nratio, histtype='step',
                       **spec_style)
        except:
            print('{} is empty. skipping...'.format(plot_name))

        if plot_ratio_err:
            bin_centers = (bins[:-1] + bins[1:])/2
            nominal, nominal_bins = specs[0].histogram(bins=nbins, range=(0, 1))
            err = np.sqrt(1/n + 1/nominal) * nominal / n
            ax[1].errorbar(bin_centers, nratio, yerr=err, ecolor='k', fmt='k.')



    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Ratio to prod4')
    ax[1].set_xlim([0, 1])
    ax[0].set_xlim([0, 1])
    ax[0].set_ylabel(ylabel)
    ax[0].set_yscale('log')

    # configure the legend on the main figure
    handles, labels = ax[0].get_legend_handles_labels()
    line_handles = [Line2D([], [], c=h.get_edgecolor(), ls=h.get_linestyle()) for h in handles]
    if additional_handles is not None:
        line_handles = line_handles + additional_handles['handles']
        labels = labels + additional_handles['labels']
    ax[0].legend(handles=line_handles, labels=labels, loc='upper center')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    fig.suptitle(plot_name.split('/')[-1].replace('_', ' '))
    fig.savefig(plot_name + '.png')
    print('Created {}.png'.format(plot_name))
    plt.close()
    

def ratio(spec1, spec2, nbins, pot, binrange=(0, 1)):
    h1, bins1 = spec1.histogram(bins=nbins, range=binrange, POT=pot)
    h2, bins2 = spec2.histogram(bins=nbins, range=binrange, POT=pot)
    ratio = np.nan_to_num(h1) / np.nan_to_num(h2)
    return np.nan_to_num(ratio), bins1

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
    parser.add_argument('--wc', action='store_true',
                        help='Flag for running on WC')
    parser.add_argument('--make_plots', action='store_true',
                        help='Make plots. Always true if providing a spectra file')
    parser.add_argument('--output', default='.',
                        help='Output directory')
    parser.add_argument('--caf', default=None,
                        help='Results of the Prod4 network from analysis cafs')
    args = parser.parse_args()

    main(args.wc, args.limit, args.stride, args.offset, 
         args.spectra_file, args.make_plots, args.output, 
         args.caf)
    
