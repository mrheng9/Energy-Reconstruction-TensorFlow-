"""Plotting data/mc distributions of different PID variable.

For reco validation with PandAna.
"""
import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import PandAna.core as pa
from PandAna.core.core import KL, KLN, KLS
from PandAna.var.analysis_vars import kCVNe, kCVNm, kCVNnc, kRHC
from PandAna.cut.analysis_cuts import kNueCorePresel


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


def kCVNCut(tables):
    cvn_min = 0.
    df = kCVNe(tables)
    dfRHC = df[kRHC(tables)==1] >= cvn_min
    dfFHC = df[kRHC(tables)!=1] >= cvn_min

    return pd.concat([dfRHC, dfFHC])
cvn_cut = pa.Cut(kCVNCut)


def main(horn_current):
    """Plot PID distributions using miniprod5 Monte-Carlo.

    Parameters
    ----------
    horn_current : string
        Beam focusing current: either forward (`fhc`) or reverse (`rhc`).

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
    # Samweb definitions
    suffix = 'eval'
    definition = get_fd_miniprod_defs(horn_current, suffix=suffix)
    #definition = 'Defname: prod_h5_R19-02-23-miniprod5.n_fd_genie_default_allswap_fhc_nova_v08_full_v1_eval'
    print(definition)

    # Get tables
    limit = 30
    stride = 30
    tables = pa.loader(definition, stride=stride, limit=limit)

    # What permutations do we want
    cuts = {'no_cut': cvn_cut, 'nue_core_presel': kNueCorePresel}
    vars = {'cvn_e': kCVNe,
            'cvn_mu': kCVNm,
            'cvn_nc': kCVNnc}

    # Create a spectrum
    spectra = {}
    for cut_name, cut in cuts.iteritems():
        spectra[cut_name] = {}
        for var_name, var in vars.iteritems():
            spectra[cut_name][var_name] = pa.spectrum(tables, cut, var)

    # Let's do it!
    tables.Go()

    # Make a histogram
    for cut_name, cut in cuts.iteritems():
        fig, ax = plt.subplots()
        for var_name, var in vars.iteritems():
            n, bins = spectra[cut_name][var_name].histogram(
                bins=20, range=(0, 1))

            ax.hist(bins[:-1], bins, weights=n, histtype='step', label=var_name)
        ax.set_xlabel('CVN 2017 score')
        ax.set_ylabel('Events')

        plt.legend(loc='upper right')

        title = 'cvn2017_' + horn_current + '_' + cut_name
        #plt.show()

        fig.savefig(title + '.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot PID distributions using miniprod5 Monte-Carlo.')
    parser.add_argument(
        'horn_current', help=('Beam focusing current: '
                              'either forward `fhc` or reverse(`rhc`)'),
        type=str, default='fhc')
    args = parser.parse_args()

    main(args.horn_current)
