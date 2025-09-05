from ROOT import TH1D, TFile
import h5py

import numpy as np


def main(spectra_file, save_to):
    if save_to is None:
        save_to = spectra_file.replace('.root', '.hdf5')

    input_file = TFile(spectra_file, 'read')
    spectra = []
    for key in input_file.GetListOfKeys():
        spectra.append({})
        spectra[-1]['name'] = key.GetName()
        spectra[-1]['hist'] = input_file.Get('{}/hist'.format(key.GetName()))
        spectra[-1]['pot']  = input_file.Get('{}/pot'.format(key.GetName())).GetBinContent(1)
    
    save_spectra(save_to, spectra)
        
def save_spectra(output_name, specs):
    if not type(specs) is list: specs = [specs]
    fobj = h5py.File(output_name, 'w')
    for spec in specs:
        edges   = bin_edges(spec['hist'])
        contents = bin_contents(spec['hist'])
        pot     = spec['pot']
        group   = spec['name']
        fobj.create_group(group)
        print('Saving ' + group)
        fobj.create_dataset(group + '/edges', data=edges)
        fobj.create_dataset(group + '/contents', data=contents)
        fobj.create_dataset(group + '/pot', data=pot)
    fobj.close()
    
def bin_edges(hist):
    nbins = hist.GetNbinsX()
    edges = np.zeros(nbins+1)
    for i in range(nbins+1):
        edges[i] = hist.GetBinLowEdge(i+1)
    return edges

def bin_contents(hist):
    nbins = hist.GetNbinsX()
    contents = np.zeros(nbins)
    for i in range(nbins):
        contents[i] = hist.GetBinContent(i+1)
    return contents

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Convert ROOT spectra into numpy arrays and save to hdf5')
    
    parser.add_argument('spectra_file', 
                        help='File containing spectra to be converted')
    parser.add_argument('--save_to', default=None,
                        help='Name of output file')
    
    args = parser.parse_args()

    main(args.spectra_file, args.save_to)
 
