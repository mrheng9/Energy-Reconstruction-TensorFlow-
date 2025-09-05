"""Plotting data/mc distributions of different PID variable.

For reco validation with PandAna.
"""
import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import PandAna.core as pa
from PandAna.core.core import KL, KLN, KLS
from PandAna.var.analysis_vars import (
    cvnProd3Train_nueid, cvnProd3Train_numuid,
    cvnProd3Train_ncid, cvnProd3Train_nutauid)
from PandAna.cut.analysis_cuts import (
    kVeto, kNueProngContainment, kNumuContainFD, kNusFDContain,
    kNueCorePresel, kNumuNoPIDFD, kNusFDPresel)


# CVN 2020
# Veto
cvn2020veto_nueid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020veto']['nueid'])
cvn2020veto_numuid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020veto']['numuid'])
cvn2020veto_ncid = pa.Var(lambda tables: tables['rec.sel.cvn2020veto']['ncid'])
cvn2020veto_nutauid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020veto']['nutauid'])
cvn2020veto_cosmicid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020veto']['cosmicid'])

# Tau cut
cvn2020taucut_nueid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020taucut']['nueid'])
cvn2020taucut_numuid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020taucut']['numuid'])
cvn2020taucut_ncid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020taucut']['ncid'])
cvn2020taucut_nutauid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020taucut']['nutauid'])
cvn2020taucut_cosmicid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020taucut']['cosmicid'])

# PtP cut
cvn2020ptpcut_nueid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020ptpcut']['nueid'])
cvn2020ptpcut_numuid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020ptpcut']['numuid'])
cvn2020ptpcut_ncid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020ptpcut']['ncid'])
cvn2020ptpcut_nutauid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020ptpcut']['nutauid'])
cvn2020ptpcut_cosmicid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020ptpcut']['cosmicid'])

# All cut
cvn2020allcut_nueid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020allcut']['nueid'])
cvn2020allcut_numuid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020allcut']['numuid'])
cvn2020allcut_ncid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020allcut']['ncid'])
cvn2020allcut_nutauid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020allcut']['nutauid'])
cvn2020allcut_cosmicid = pa.Var(
    lambda tables: tables['rec.sel.cvn2020allcut']['cosmicid'])


def get_sam_definition(prod_str, tag, det, data_mc_str,
                       swap_str, horn_current, period, version, suffix):
    """Get a single samweb definition.

    Parameters
    ----------
    prod_str : string
        Describes the production file-type e.g. `prod_h5`.
    tag : string
        Release for this production tag.
    det : string
        Which detector: far (`fd`) or near (`nd`)?
    data_mc_str : string
        Describes whether this is data or MC and which generator e.g
        `genie_default`.
    swap_str : string
        Either `nonswap`, `fluxswap` or `tauswap`.
    horn_current : string
        Beam focusing current: either forward (`fhc`) or reverse (`rhc`).
    period : string
        Specify a specific data-taking period or `full` for all periods.
    version : string
        Definition version e.g. `v1`.
    suffix : string
        Provides extra information on networks and training.

    Returns
    -------
    string
        Correctly formatted, full definition name.

    """
    definition_string = 'Defname: '
    definition_string += (prod_str + '_' + tag + '_' + det + '_' +
                          data_mc_str + '_' + swap_str + '_' + horn_current +
                          '_nova_v08_' + period + '_' + version + '_' + suffix)
    return definition_string


def get_fd_miniprod_defs(horn_current, prod_str='prod_h5',
                         tag='R19-02-23-miniprod5.n',
                         det='fd', data_mc_str='genie_default',
                         period='full', version='v1', suffix=''):
    """Get a list of miniprod5 definitions for a given horn current.

    Parameters
    ----------
    horn_current : string
        Beam focusing current: either forward (`fhc`) or reverse (`rhc`).
    prod_str : string
        Describes the production file-type (default is `prod_h5`).
    tag : string
        Release for this production tag (default is
        `R19-02-23-miniprod5.n`)
    det : string
        Which detector: far (`fd`, default) or near (`nd`)?
    data_mc_str : string
        Describes whether this is data or MC and which generator
        (default is `genie_default`).
    period : string
        Specify a specific data-taking period or `full` (default) for
        all periods.
    version : string
        Definition version (default is `v1`).
    suffix : string
        Provides extra information on networks and training.

    Returns
    -------
    string
        Contains samweb definition for all permutations of swap string.

    """
    swap_str = 'allswap'
    return get_sam_definition(prod_str, tag, det, data_mc_str, swap_str,
                              horn_current, period, version, suffix)


def get_files(swap_str, horn_current):
    d = ('/lfstev/nnet/R19-02-23-miniprod5/FD-' +
         swap_str + '-' + horn_current + '-Eval/')
    var = ['rec.sel.cvn2020veto', 'rec.sel.cvn2020ptpcut',
           'rec.sel.cvn2020taucut', 'rec.sel.cvn2020allcut']
    return [os.path.join(d, f) for f in os.listdir(d) if 'h5caf.h5' in f]


def get_all_files(horn_current):
    swap_strs = ['Nonswap', 'Fluxswap', 'Tau']
    all_files = []
    for swap_str in swap_strs:
        all_files = all_files + get_files(swap_str, horn_current)
    print('Generated a list of {} files'.format(len(all_files)))
    return all_files


def main(horn_current, on_wc, stride=None, limit=None):
    """Plot PID distributions using miniprod5 Monte-Carlo.

    Parameters
    ----------
    horn_current : string
        Beam focusing current: either forward (`fhc`) or reverse (`rhc`).
    on_wc: bool
        Flag if running on Wilson Cluster.

    Returns
    -------
    type
        Description of returned object.

    Raises
    -------
    ExceptionName
        Why the exception is raised.

    """
    # Set up loaders
    files = []
    if on_wc:
        if horn_current == 'fhc':
            files = get_all_files('FHC')
        else:
            files = get_all_files('RHC')
    else:
        # Samweb definitions
        suffix = 'eval'
        files = get_fd_miniprod_defs(horn_current, suffix=suffix)
        print(files)

    # Get tables
    tables = pa.loader(files, stride=stride, limit=limit)

    # What permutations do we want
    # Cuts
    # No Numu containment
    # Warning! No data read for rec.trk.kalman.tracks
    any_fd_containment = kNueProngContainment | kNusFDContain
    any_fd_presel = kNueCorePresel | kNusFDPresel
    cuts = {
        'no_cut': kVeto,
        'any_fd_containment': any_fd_containment,
        'any_fd_presel': any_fd_presel,
        }

    # Vars
    vars = {
        'nueid': {
            'cvn2020veto': cvn2020veto_nueid,
            'cvn2020taucut': cvn2020taucut_nueid,
            'cvn2020ptpcut': cvn2020ptpcut_nueid,
            'cvn2020allcut': cvn2020allcut_nueid,
            'cvnProd3Train': cvnProd3Train_nueid,
            },
        'numuid': {
            'cvn2020veto': cvn2020veto_numuid,
            'cvn2020taucut': cvn2020taucut_numuid,
            'cvn2020ptpcut': cvn2020ptpcut_numuid,
            'cvn2020allcut': cvn2020allcut_numuid,
            'cvnProd3Train': cvnProd3Train_numuid,
            },
        'ncid': {
            'cvn2020veto': cvn2020veto_ncid,
            'cvn2020taucut': cvn2020taucut_ncid,
            'cvn2020ptpcut': cvn2020ptpcut_ncid,
            'cvn2020allcut': cvn2020allcut_ncid,
            'cvnProd3Train': cvnProd3Train_ncid,
            },
        # 'nutauid': {
        #     'cvn2020veto': cvn2020veto_nutauid,
        #     'cvn2020taucut': cvn2020taucut_nutauid,
        #     'cvn2020ptpcut': cvn2020ptpcut_nutauid,
        #     'cvn2020allcut': cvn2020allcut_nutauid,
        #     'cvnProd3Train': cvnProd3Train_nutauid,
        #     },
        'cosmicid': {
            'cvn2020veto': cvn2020veto_cosmicid,
            'cvn2020taucut': cvn2020taucut_cosmicid,
            'cvn2020ptpcut': cvn2020ptpcut_cosmicid,
            'cvn2020allcut': cvn2020allcut_cosmicid,
            },
        }

    # Create a spectrum
    print('Creating spectra')
    spectra = {}
    spectra_list = []
    spectra_names = []
    for cut_name, cut in cuts.items():
        spectra[cut_name] = {}
        for pid_name, networks in vars.items():
            spectra[cut_name][pid_name] = {}
            for network_name, var in networks.items():
                spectrum = pa.spectrum(tables, cut, var)
                spectrum_name = cut_name + '_' + pid_name + '_' + network_name
                spectra[cut_name][pid_name][network_name] = spectrum
                spectra_list.append(spectrum)
                spectra_names.append(spectrum_name)

    # Let's do it!
    print('Loaders Go!')
    tables.Go()

    # Save them
    print('Saving spectra')
    filename = 'saved_spectra'
    if stride:
        filename += '_s{}'.format(stride)
    if limit:
        filename += '_l{}'.format(limit)
    filename += '.hdf5'
    pa.save_spectra(filename, spectra_list, spectra_names)

    # Make a histogram
    print('Making histograms')
    for cut_name, cut in cuts.items():
        for pid_name, networks in vars.items():
            fig, ax = plt.subplots()
            for network_name, var in networks.items():
                n, bins = spectra[cut_name][pid_name][network_name].histogram(
                    bins=20, range=(0, 1))

                ax.hist(bins[:-1], bins, weights=n,
                        histtype='step', label=network_name)
            ax.set_xlabel('PID score')
            ax.set_ylabel('Events')

            plt.legend(loc='upper center')

            # Save it
            title = horn_current + '_' + cut_name + '_' + pid_name
            # fig.savefig(title + '.png')
            fig.suptitle(title)

            # Logscale y
            ax.set_yscale('log')
            title += '_log'

            if stride:
                title += '_s{}'.format(stride)
            if limit:
                title += '_l{}'.format(limit)
            fig.savefig(title + '.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot PID distributions using miniprod5 Monte-Carlo.')
    parser.add_argument(
        'horn_current', help=('Beam focusing current: '
                              'either forward `fhc` or reverse(`rhc`)'),
        type=str, default='fhc')
    parser.add_argument('--on_wc', '--wc',
                        help='Flag if running on WC', action='store_true')
    parser.add_argument('--limit', '-l', type=int, default=1,
                        help='Limit number of files')
    parser.add_argument('--stride', '-s', type=int, default=None,
                        help='Stride length over file list')
    args = parser.parse_args()

    main(args.horn_current, args.on_wc, args.stride, args.limit)
