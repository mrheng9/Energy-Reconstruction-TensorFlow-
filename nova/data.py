from __future__ import division
from __future__ import absolute_import
import numpy as np
import os
from .config import MAX_EXAMPLES, PIXEL_SCALE, VERTEX_SCALE, CUT_THRESHOLD, DATA_FORMAT

class PixelmapGenerator:
    """
    Pixelmap generator class for Keras.
    """

    def __init__(self, pixel_map_type, pixel_map_dim, check_criterion, formula, preprocess):
        """
        Args 
        pixel_map_type: 'prong' or 'event'
        pixel_map_dim: tuple (pixel_map_dim1, pixel_map_dim2)
        check_criterion: boolean function for event/prong inclusion criterion
        formula: function for adding features
        preprocess: function for preprocessing
        """
        self.pixel_map_type = pixel_map_type
        self.pixel_map_dim = pixel_map_dim
        self.check_criterion = check_criterion
        self.formula = formula
        self.preprocess = preprocess
 
        
    @staticmethod
    def quick_build(data_type, pixel_map_dim=(141, 141)):
        """
        Return a default generator.
        
        Args 
        data_type: 'nue', 'numu', 'electron', or 'muon'
        """
        width, height = pixel_map_dim
        if width == 151 or width == 281:
            # dummy function
            formula = not_divide_plane
        else:
            formula = divide_plane
        
        if data_type == 'nue':
            return(PixelmapGenerator('event', pixel_map_dim, check_nue, formula, constant_scale))
        elif data_type == 'numu':
            return(PixelmapGenerator('event', pixel_map_dim, check_numu, formula, constant_scale))
        elif data_type == 'electron':
            return(PixelmapGenerator('prong', pixel_map_dim, check_electron, formula, constant_scale))
        elif data_type == 'muon':
            return(PixelmapGenerator('prong', pixel_map_dim, check_muon, formula, constant_scale))


    def flow(self, txt_files, batch_size, reweigh):
        """
        Return pixelmaps from txt files indefinitely.

        Args
        txt_files: list of file names
        batch_num: number of batches from each file
        batch_size: size of each batch
        """
        while 1:
            for i, f in enumerate(txt_files):
                if self.pixel_map_type == 'prong':
                    X, V, y = prong_iter(f, self.pixel_map_dim, self.check_criterion, self.formula, self.preprocess)
                elif self.pixel_map_type == 'event':
                    X, V, y = event_iter(f, self.pixel_map_dim, self.check_criterion, self.formula, self.preprocess)
                n = X.shape[0]
                pixel_map_dim1 = X.shape[2]
                pixel_map_dim2 = X.shape[3]
                batch_num = n//batch_size
                for j in range(batch_num):
                    select = np.random.randint(low=0, high=n, size=batch_size)
                    batchX = X[select, :, :, :]
                    
                    channel_axis = 1 if DATA_FORMAT == 'channels_first' else -1
                    batchX_1 = np.expand_dims(batchX[:, 0, :, :], axis=channel_axis)
                    batchX_2 = np.expand_dims(batchX[:, 1, :, :], axis=channel_axis)
                    
                    batchV = V[select, :]
                    batchY = y[select, :]
                    
                    if reweigh:
                        w = np.zeros((batch_size,))
                        for i in range(batch_size):
                            trueE = batchY[i]
                            if 0 <= trueE < 0.5:
                                w[i] = 42.5586924
                            elif 0.5 <= trueE < 5:
                                w[i] = 83.0923-97.7887*trueE+35.1566*trueE*trueE-3.42726*trueE*trueE*trueE
                            elif 5 <= trueE < 15:
                                w[i] = -137.471+68.8724*trueE-7.69719*trueE*trueE+0.26117*trueE*trueE*trueE
                            elif 15 <= trueE:
                                w[i] = 45.196
                            else:
                                raise("Negative NuE Energy!")
                        yield [batchX_1, batchX_2, batchV], batchY, w
                    else:
                        yield [batchX_1, batchX_2, batchV], batchY


def list_files(dir, filter=None):
    """
    Produces a list of files in dir.

    # Arguments:
        dir (str): Directory to list files from.
        filter (str): String to be contained in each file name.

    # Returns:
        (list) of filenames
    """
    return [os.path.join(dir, file_name) for file_name in os.listdir(dir) if filter in file_name]


def parse_text(line):
    """
    Create a data dictionary based on a single line of text.
    
    Args
    line (str): one line of raw data from the CSV

    Returns
    (dict) with keys/values of extracted variables
    """
    info = {}
    data = line.split(' ')
    info['run'] = data[0]
    info['subrun'] = data[1]
    info['event'] = data[2]
    info['subevent'] = data[3]
    info['slice'] = data[4]
    info['sheId'] = data[5]
    # event and prong identifiers
    info['event_tag'] = int(info['run'] + info['subrun'] + info['event'] + info['subevent'] + info['slice'])
    if info['sheId'] != '-1':
        info['prong_tag'] = int(info['run'] + info['subrun'] + info['event'] + info['subevent'] + info['slice'] + info['sheId'])
    else:
        info['prong_tag'] = None
    # current reconstruction
    info['sheE'] = float(data[6])
    info['sheNuE'] = float(data[7])
    offset = 0
    if len(data) == 39:
        offset = 3
        info['length'] = float(data[17])
        info['numuCVN'] = float(data[18])
        info['nueCVN'] = float(data[19])
    # location
    info['view'] = int(data[17 + offset])
    info['planeLocal'] = int(data[18 + offset])
    info['cellLocal'] = int(data[19 + offset])
    info['plane'] = int(data[20 + offset])
    info['cell'] = int(data[21 + offset])
    info['energy'] = float(data[22 + offset])
    # true prong
    info['trueProng'] = int(data[23 + offset])
    info['truePx'] = float(data[24 + offset])
    info['truePy'] = float(data[25 + offset])
    info['truePz'] = float(data[26 + offset])
    info['trueE'] = float(data[27 + offset])
    # true nu
    info['trueNuPdg'] = int(data[31 + offset])
    info['trueCC'] = int(data[32 + offset])
    info['trueNuMode'] = int(data[33 + offset])
    info['trueNuE'] = float(data[34 + offset])
    return(info)
    

def check_electron(deposit):
    """
    Check whether a deposit belongs to a electron prong.
    """
    return deposit['trueProng'] == 11 or deposit['trueProng'] == -11


def check_nue(deposit):
    """
    Check whether a deposit belongs to a nue event.
    """
    return deposit['trueNuPdg'] == 12 or deposit['trueNuPdg'] == -12


def check_muon(deposit):
    """
    Check whether a deposit belongs to a muon prong.
    """
    return deposit['trueProng'] == 13 or deposit['trueProng'] == -13


def check_numu(deposit):
    """
    Check whether a deposit belongs to a numu event.
    """
    return deposit['trueNuPdg'] == 14 or deposit['trueNuPdg'] == -14


def add_event_energy(targets, i, deposit):
    """
    Add neutrino energy to regression targets.
    """
    targets[i, 0] = deposit['trueNuE']


def add_prong_energy(targets, i, deposit):
    """
    Add lepton energy and momentum to regression targets.
    """
    targets[i, 0] = deposit['trueE']
    #targets[i, 1] = deposit['truePx']
    #targets[i, 2] = deposit['truePy']
    #targets[i, 3] = deposit['truePz']


def add_vertex(vertices, i, deposit):
    """
    Add vertex X or Y.
    """
    if deposit['view'] == 0:
        vertices[i, 0] = deposit['cell']
    else:
        vertices[i, 1] = deposit['cell']


def divide_plane(features, i, deposit):
    """
    Divide plane number by 2 to reduce dimension by half.
    """
    features[i, deposit['view'], (deposit['planeLocal'] + 30) // 2, deposit[
        'cellLocal'] + 70] = deposit['energy']
    
def not_divide_plane(features, i, deposit):
    """
    Divide plane number by 2 to reduce dimension by half.
    """
    features[i, deposit['view'], (deposit['planeLocal'] + 30), deposit[
        'cellLocal'] + 70] = deposit['energy']


def constant_scale(features, vertices, targets):
    """
    Preprocess data by constant scaling.
    """
    features = features * PIXEL_SCALE
    vertices = vertices * VERTEX_SCALE
    return(features, vertices, targets)
    

def energy_ratio_cut(features, vertices, targets):
    """
    Remove examples where the sum of energy deposits is too low.
    """
    sum_energy = np.sum(np.sum(np.sum(features, axis=1), axis=1), axis=1)
    ratio = targets[:, 0] / sum_energy
    select = ratio < CUT_THRESHOLD
    features = features[select, :, :, :] * PIXEL_SCALE
    vertices = vertices[select, :] * VERTEX_SCALE
    targets = targets[select, :]
    return(features, vertices, targets)
    

def event_iter(file_path, pixel_map_dim, check_criterion, formula, preprocess):
    """
    Generic function to iterate events from a txt file. 
    
    Args
        file_path: string of txt file path
        pixel_map_dim: tuple of pixel map dimensions
        check_criterion: boolean function for event/prong inclusion criterion
        formula: function for adding features
        preprocess: function for preprocessing
        
    Returns
        features: 4d np array of pixel maps
        vertices: 2d np array of vertices
        targets: 2d np array of regression targets
    """
    features = np.zeros((MAX_EXAMPLES, 2, pixel_map_dim[0], pixel_map_dim[1]))
    vertices = np.zeros((MAX_EXAMPLES, 2))
    targets = np.zeros((MAX_EXAMPLES, 1))
    i = -1
    current_event_tag = -1
    match = False
    with open(file_path, 'r') as f:
        for line in f:
            deposit = parse_text(line)
            if current_event_tag != deposit['event_tag']:
                current_event_tag = deposit['event_tag']
                match = False
                if check_criterion(deposit):
                    match = True
                    i += 1
                    if i == MAX_EXAMPLES:
                        break
                    add_event_energy(targets, i, deposit)
            if match:
                formula(features, i, deposit)
                if deposit['prong_tag'] is not None:
                    add_vertex(vertices, i, deposit)
    return(preprocess(features[0:(i + 1), :, :, :], vertices[0:(i + 1), :], targets[0:(i + 1), :]))


def prong_iter(file_path, pixel_map_dim, check_criterion, formula, preprocess):
    """
    Generic function to iterate prongs from a txt file. 
    
    Args
        file_path: string of txt file path
        pixel_map_dim: tuple of pixel map dimensions
        check_criterion: boolean function for event/prong inclusion criterion
        formula: function for adding features
        preprocess: function for preprocessing
        
    Returns
        features: 4d np array of pixel maps
        vertices: 2d np array of vertices
        targets: 2d np array of regression targets
    """
    features = np.zeros((MAX_EXAMPLES, 2, pixel_map_dim[0], pixel_map_dim[1])) 
    vertices = np.zeros((MAX_EXAMPLES, 2))
    targets = np.zeros((MAX_EXAMPLES, 1))
    i = -1
    current_prong_tag = -1
    match = False
    with open(file_path, 'r') as f:
        for line in f:
            deposit = parse_text(line)
            if deposit['prong_tag'] is None:
                continue
            if current_prong_tag != deposit['prong_tag']:
                current_prong_tag = deposit['prong_tag']
                match = False
                if check_criterion(deposit):
                    match = True
                    i += 1
                    if i == MAX_EXAMPLES:
                        break
                    add_prong_energy(targets, i, deposit)
            if match:
                formula(features, i, deposit)
                add_vertex(vertices, i, deposit)
    return(preprocess(features[0:(i + 1), :, :, :], vertices[0:(i + 1), :], targets[0:(i + 1), :]))