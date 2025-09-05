import glob
import h5py
import sys
import os
sys.path.append('../..')
from PandAna.core import spectrum, save_tree, load_tree

def main(dest, source):
    if not type(source) is list: source = [source]
    print('Saving spectra from {} files to {}'.format(len(source), dest))
    # assume all of the files have the same groups in them
    first = h5py.File(source[0], 'r')
    groups = list(first.keys())
    first.close()

    print('Found {} groups'.format(len(groups)))
    # this will work if all files have the exact same spectra
    source_spectra = {}
    dest_spectra = {}

    # create dictionaries of empty lists. We'll put spectra into these later
    for g in groups:
        source_spectra[g] = []
        dest_spectra[g] = None

    # for each source file, load all groups
    for s in source:
        print('Loading spectra from {}'.format(s))
        spectra = load_tree(s, groups)
        
        # add each spectra to the dictionary of source_spectra according to group
        for group, spec in spectra:
            source_spectra[group].append(spec)
    
    # get list of source_spectra corresponding to each group
    for g in groups:
        for spec in source_spectra[g]:
            # if first spectra make assignment
            if dest_spectra[g] is None:
                dest_spectra[g] = spec
            # else add it to the spectrum that already exists for that group
            else:
                # make sure all spectra being summed have the same name
                assert str(dest_spectra[g]._df.name) == str(spec._df.name), \
                    'Spectra must have the same name to add them together ({} =/= {})'.format(dest_spectra[g]._df.name, spec._df.name)
                dest_spectra[g] = dest_spectra[g] + spec
    
    save_tree(dest, list(dest_spectra.values()), list(dest_spectra.keys()))
    
            
    

if __name__ == '__main__':
    dest = sys.argv[1]
    if os.path.isfile(dest):
        print('{} already exists. Will not overwrite'.format(dest))
        sys.exit()
    source = []
    for arg in sys.argv[1:]:
        these = glob.glob(arg)
        if type(these) is list: source = source + these
        else: source.append(these)
    print(dest, source)
    main(dest, source)
