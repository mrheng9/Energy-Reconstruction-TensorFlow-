from __future__ import print_function
import os
import numpy as np
import pandas as pd
import h5py
import lmdb

TXT_DIR = "/baldig/physicsprojects/nova/data/raw/flat/"
DATASET_DIR = "/baldig/physicsprojects/nova/data/regular/"
COLNAMES = np.asarray(['Run', 'SubRun', 'Event', 'SubEvent', 'Slice',
                       'SheId', 'SheEnergy', 'SheNueEnergy', 'SheDirX', 'SheDirY', 'SheDirZ', 'SheStartX', 'SheStartY',
                       'SheStartZ', 'SheStopX', 'SheStopY', 'SheStopZ',
                       'View', 'LocalPlaneId', 'LocalCellId',
                       'VertexGlobalPlaneID', 'VertexGlobalCellID', 'cellEnergy', 'TrueProngPdg', 'TruePx', 'TruePy',
                       'TruePz', 'TrueE', 'TrueNuVtxX', 'TrueNuVtxY', 'TrueNuVtxZ', 'TrueNuPdg', 'TrueNuCCNC', 'TrueNuMode', 'TrueNuEnergy'])
# COLNAMES = ['Run',
#              'SubRun',
#              'Event',
#              'SubEvent',
#              'Slice',
#              'SheId',
#              'SheEnergy',
#              'SheNueEnergy',
#              'SheDir[0]',
#              'SheDir[1]',
#              'SheDir[2]',
#              'SheStart[0]',
#              'SheStart[1]',
#              'SheStart[2]',
#              'SheStop[0]',
#              'SheStop[1]',
#              'SheStop[2]',
#              'SheLength',
#              'EvtCVN[0]',
#              'EvtCVN[1]',
#              'View',
#              'LocalPlaneId',
#              'LocalCellId',
#              'VtxPlane',
#              'VtxCell',
#              'cellEnergy',
#              'TruePdg',
#              'TrueP4[0]',
#              'TrueP4[1]',
#              'TrueP4[2]',
#              'TrueP4[3]',
#              'TrueNuVtx[0]',
#              'TrueNuVtx[1]',
#              'TrueNuVtx[2]',
#              'TrueNuPdg',
#              'TrueNuCCNC',
#              'TrueNuMode',
#              'TrueNuEnergy']
TAG = "fdflatnuesg"
FILE_EXTENSION = '.skim.txt'

def get_num_events(the_df):
    return the_df.index.unique().shape[0]
            
def get_dataframe(number, dir=TXT_DIR, tag=TAG, prong_level=False):
    filename = tag + str(number) + FILE_EXTENSION
    filepath = os.path.join(dir, filename)
    print("Reading CSV {}".format(filepath))
    df = pd.read_csv(filepath, delim_whitespace=True, header=None, names=COLNAMES)
    # Create column with ID
    df['ID'] = ''
    for col in ['Run', 'SubRun', 'Event', 'SubEvent', 'Slice']:
        df['ID'] += df[col].map(str)

    # Set ID column as row indexing
    df = df.set_index(['ID'])
    if prong_level:
        return set_ids_to_prong_level(df)
    else:
        return df
        
def set_ids_to_prong_level(df):
    df['prong_level_id'] = ''
    df['prong_level_id'] = df.index.map(str) + ['_']*len(df.index.map(str)) + df['SheId'].map(str)
    return df.set_index(['prong_level_id'])

def get_dir(dataset_dir, dataset_name):
        new_dir = os.path.join(dataset_dir, dataset_name)
        try:
            os.mkdir(new_dir)
            print("Created directory {}".format(new_dir))
        except OSError:
            pass
        return new_dir
    
def store_data(name, data_object, tag=TAG, raw_dir=TXT_DIR, fr_=0, to_=10, verbose=0, skip=[], hdf=False, target_dir=DATASET_DIR, assert_no_nan=True):
    saving_dir = get_dir(target_dir, data_object.name)
    save_path = os.path.join(saving_dir)
    
    print("Saving to {}".format(save_path))
    
    if os.path.isdir(save_path):
#         print("Removing all files in {}".format(save_path))
        print("Path {} exists".format(save_path))
        # for f in os.listdir(save_path):
        #     os.remove(os.path.join(save_path, f))
    else:
        os.mkdir(save_path)
    
    num_events_processed = 0
    for counter, i in enumerate(range(fr_, to_)):
        
        print("Processing {}{}".format(tag, i))
        
        try:
            if not hdf:
                df = get_dataframe(dir=raw_dir, tag=tag, number=i, prong_level=data_object.prong_level)
            else:
                df = pd.read_hdf(os.path.join(raw_dir, tag + str(i) + '.h5'))
        except:
            continue
            
        if data_object.prong_level:
            df = set_ids_to_prong_level(df)
        
        filepath = os.path.join(save_path, str(i) + ".h5")
        
        if os.path.isfile(filepath):
                print("Removing {}".format(filepath))
#                 os.remove(filepath)
        print("Saving to {}".format(filepath))
        
        f = h5py.File(filepath, libver='latest')
        
        num_events = get_num_events(df)
        if num_events == 0:
            continue
        
        data_array = np.zeros((num_events,) + data_object.dim)
        id_array = np.chararray((num_events, 1), itemsize=30)
        event_idxs = []  # need to keep track of when we're actually adding events
        
        for event_counter, (event_id, event_data) in enumerate(df.groupby(df.index)):
            
            # Skip if we are looking at prongs and the prong-cluster has ID -1
            
            if data_object.prong_level and -1 in event_data['SheId'].values:
                continue
                
            # Skip if we are looking at prongs and the majority of deposits in the prong cluster
            # is not by the particle of interest
            
            
            if data_object.prong_level:
                pid_with_most_counts = event_data['TruePdg'].value_counts().idxmax()
                if pid_with_most_counts not in data_object.target_particles:
                    continue
                
            data_array[event_counter] = data_object.get_data_as_array(event_data)
            id_array[event_counter] = str(i)+ '_' + str(event_id)
            event_idxs.append(event_counter)

            
        # filter out those events which we actually added
        
        data_array = data_array[event_idxs]
        id_array = id_array[event_idxs]
        
        if assert_no_nan:
            assert np.isnan(data_array).sum() == 0
        
        num_events_processed += len(id_array)
        
        print("Number of events: {}".format(num_events))
        print("Number of processed events: {}".format(num_events_processed))
        


        dataset = f.create_dataset(name='data', shape=data_array.shape,
                         dtype='float32', data=data_array)
        idset = f.create_dataset(name='id', shape=id_array.shape,
                         dtype='S30', data=id_array)
        print(dataset.shape)

        f.close()
    print("Processed and saved to {}".format(save_path))
    
    
def store_lmdb(data_objects, fr_=0, to_=10, verbose=0):
    saving_dir = DATASET_DIR
    save_path = os.path.join(saving_dir, 'lmdb')
    
    print("Saving to {}".format(save_path))
    max_map_size = sum(os.path.getsize(os.path.join(TXT_DIR, f)) for f in os.listdir(TXT_DIR) if os.path.isfile(os.path.join(TXT_DIR, f)))
    print("Map size set to {}".format(max_map_size))
    
    with lmdb.open(save_path, map_size=max_map_size) as env:
        with env.begin(write=True) as txn:
            num_events_processed = 0
            for i in range(fr_, to_):

                print("Processing {}".format(i))

                df = get_dataframe(dir=TXT_DIR, tag=TAG, number=i, prong_level=False)

                num_events = get_num_events(df)
                if num_events == 0:
                    continue

                for event_id, event_data in df.groupby(df.index):

                    for data_object in data_objects:
                        array = data_object.get_data_as_array(event_data)
                        key = data_object.name + "_" + str(num_events_processed)
                        assert np.isnan(array).sum() == 0                                              
                        txn.put(key.encode('ascii'), array.tostring())
                        txn.put((key + "_shape").encode('ascii'), str(array.shape).encode())

                    num_events_processed += 1
                    if num_events_processed % 1000 == 0:
                        print("\n\tNumber of processed events: {}\n".format(num_events_processed))
    print("-"*100 + "\nTotal processed events: {}\n".format(num_events_processed) + "-"*100)
    
    
class EventImages(object):
    def __init__(self):
        self.dim = (2, 76, 141)
        self.name = 'small_event_images'
        self.prong_level = False
        
    def get_data_as_array(self, the_event_data):
        new_array = np.zeros(self.dim)

        for current_view in [0, 1]:
            current_view_features_all = self._get_data_for_this_view(the_event_data, current_view)
            current_view_features = current_view_features_all[current_view_features_all['SheId']==-1]
            
            if self.dim[1] == 76:
                plane = np.ceil((current_view_features['LocalPlaneId'] + 30)/2.)
            else:
                plane = current_view_features['LocalPlaneId'] + 30
            cell = current_view_features['LocalCellId'] + 70
            energy = current_view_features['cellEnergy']
            
            new_array[current_view, plane.astype('int64'), cell.astype('int64')] = energy
        assert not np.all(new_array==0.)
        return new_array*100.
        
    @staticmethod
    def _get_data_for_this_view(event_data, current_view):
        return event_data[event_data['View'] == current_view]
    
class EventEnergies(object):
    def __init__(self):
        self.dim = (1,)
        self.name = 'event_energy'
        self.prong_level = False
        
    def get_data_as_array(self, the_event_data):
        event_energy = the_event_data['TrueNuEnergy'].unique()
        if len(event_energy)!=1:
            warnings.warn("Expected one event energy, found {}. Taking first.".format(event_energy), UserWarning)
            event_energy = event_energy[0]
        return event_energy

class EventVertices(object):
    def __init__(self):
        self.dim = (2,) # CellView0, CellView1
        self.name = 'event_vertices'
        self.prong_level = False
        
    def get_data_as_array(self, the_event_data):
        new_array = self.get_vertices(the_event_data)
        return new_array/100.
    
    @staticmethod
    def get_vertices(the_event_data):
        assert the_event_data is not np.nan

        # check if there is a data point, otherwise put the middle of the range
        if the_event_data[the_event_data['View']==0].loc[:, ['VtxCell']].shape[0] != 0:
            cell_view_0 = the_event_data[the_event_data['View']==0].loc[:, ['VtxCell']].iloc[0]
        else:
            cell_view_0 = np.array([192]) # middle

        if the_event_data[the_event_data['View']==1].loc[:, ['VtxCell']].shape[0] != 0:
            cell_view_1 = the_event_data[the_event_data['View']==1].loc[:, ['VtxCell']].iloc[0]
        else:
            cell_view_1 = np.array([192]) # middle

        return np.concatenate([cell_view_0, cell_view_1])
    
    
class EventVertices3D(object):
    def __init__(self):
        self.dim = (3,) # CellView0, CellView1
        self.name = 'event_vertices_3d'
        self.prong_level = False
        
    def get_data_as_array(self, the_event_data):
        new_array = self.get_vertices(the_event_data)
        return new_array
    
    @staticmethod
    def get_vertices(the_event_data):
        assert the_event_data is not np.nan

        mask = (the_event_data['View']==0)
        # check if there is a data point, otherwise put the middle of the range
        if mask.sum() != 0:
            reco_vtx_cell_view_0 = float(the_event_data.loc[mask, ['VtxCell']].iloc[0])
            reco_vtx_plane_view_0 = float(the_event_data.loc[mask, ['VtxPlane']].iloc[0])
        else:
            reco_vtx_cell_view_0 = 0.
            reco_vtx_plane_view_0 = 0.

        mask = (the_event_data['View']==1)
        if mask.sum() != 0:
            reco_vtx_cell_view_1 = float(the_event_data.loc[mask, ['VtxCell']].iloc[0])
            reco_vtx_plane_view_1 = float(the_event_data.loc[mask, ['VtxPlane']].iloc[0])
        else:
            reco_vtx_cell_view_1 = 0.
            reco_vtx_plane_view_1 = 0.
        
        reco_vtx = (reco_vtx_cell_view_0, reco_vtx_cell_view_1, reco_vtx_plane_view_0 or reco_vtx_plane_view_1)

        try:
            return np.array(reco_vtx)
        except ValueError:
            print(reco_vtx)
            assert False
    
# class LocalTrueVertices(object):
#     def __init__(self):
#         self.dim = (3,) # CellView0, CellView1
#         self.name = 'event_true_vertices'
#         self.prong_level = False
        
#     def get_data_as_array(self, the_event_data):
#         new_array = self.get_vertices(the_event_data)
#         return new_array
    
#     @staticmethod
#     def get_vertices(the_event_data):
#         assert the_event_data is not np.nan

#         mask = (the_event_data['View']==0)
#         # check if there is a data point, otherwise put the middle of the range
#         if mask.sum() != 0:
#             reco_vtx_cell_view_0 = the_event_data.loc[mask, ['VtxCell']].iloc[0]
#             reco_vtx_plane_view_0 = the_event_data.loc[mask, ['VtxPlane']].iloc[0]
#         else:
#             reco_vtx_cell_view_0 = np.nan
#             reco_vtx_plane_view_0 = np.nan

#         mask = (the_event_data['View']==1)
#         if mask.sum() != 0:
#             reco_vtx_cell_view_1 = the_event_data.loc[mask, ['VtxCell']].iloc[0]
#         else:
#             reco_vtx_cell_view_1 = np.nan
            
#         true_vtx = the_event_data.iloc[0][['TrueNuVtx[0]', 'TrueNuVtx[1]', 'TrueNuVtx[2]']]
# #         print(true_vtx)
# #         print((true_vtx[0], true_vtx[1], true_vtx[2]))
#         reco_vtx = (reco_vtx_cell_view_0, reco_vtx_cell_view_1, reco_vtx_plane_view_0)
            
#         true_local_vtx = [float((true_vtx[0]+758.)/3.93-reco_vtx[0]), float((true_vtx[1]+749.)/3.93-reco_vtx[1]), float(true_vtx[2]/6.61-reco_vtx[2])]

#         try:
#             return np.array(true_local_vtx)
#         except ValueError:
#             print(true_local_vtx)
#             assert False
    
    
class TrueVertices(object):
    def __init__(self):
        self.dim = (3,) # CellView0, CellView1
        self.name = 'event_true_global_vertices'
        self.prong_level = False
        
    def get_data_as_array(self, the_event_data):
        new_array = self.get_vertices(the_event_data)
        return new_array
    
    @staticmethod
    def get_vertices(the_event_data):
        assert the_event_data is not np.nan
            
        true_vtx = the_event_data.iloc[0][['TrueNuVtx[0]', 'TrueNuVtx[1]', 'TrueNuVtx[2]']]

        try:
            return np.array([float(true_vtx[0]), float(true_vtx[1]), float(true_vtx[2])])
        except ValueError:
            print(true_vtx)
            assert False
    
class ElectronImages(object):
    def __init__(self):
        self.dim = (2, 151, 141)
        self.name = 'electron_images'
        self.prong_level = True
        self.target_particles = [-11, 11]
        
    def get_data_as_array(self, the_event_data):
        new_array = np.zeros(self.dim)

        for current_view in [0, 1]:
            current_view_features = self._get_data_for_this_view(the_event_data, current_view)
            
            if self.dim[1] == 76:
                plane = np.ceil((current_view_features['LocalPlaneId'] + 30)/2.)
            else:
                plane = current_view_features['LocalPlaneId'] + 30
            cell = current_view_features['LocalCellId'] + 70
            energy = current_view_features['cellEnergy']
            
            new_array[current_view, plane.astype('int64'), cell.astype('int64')] = energy
        assert not np.all(new_array==0.)
        return new_array*100.
        
    @staticmethod
    def _get_data_for_this_view(event_data, current_view):
        return event_data[event_data['View'] == current_view]
    
    
class SmallElectronImages(object):
    def __init__(self):
        self.dim = (2, 76, 141)
        self.name = 'small_electron_images'
        self.prong_level = True
        self.target_particles = [-11, 11]
        
    def get_data_as_array(self, the_event_data):
        new_array = np.zeros(self.dim)

        for current_view in [0, 1]:
            current_view_features = self._get_data_for_this_view(the_event_data, current_view)
            
            plane = np.ceil((current_view_features['LocalPlaneId'] + 30)/2.)
            cell = current_view_features['LocalCellId'] + 70
            energy = current_view_features['cellEnergy']
            
            new_array[current_view, plane.astype('int64'), cell.astype('int64')] = energy
        assert not np.all(new_array==0.)
        return new_array*100.
        
    @staticmethod
    def _get_data_for_this_view(event_data, current_view):
        return event_data[event_data['View'] == current_view]
    
class ElectronEnergies(object):
    def __init__(self):
        self.dim = (1,)
        self.name = 'electron_energy'
        self.prong_level = True
        self.target_particles = [-11, 11]
        
    def get_data_as_array(self, the_event_data):
        energy_with_most_counts = the_event_data['TrueP4[3]'].value_counts().idxmax()
        cells_with_most_counts = the_event_data[the_event_data['TrueP4[3]']==energy_with_most_counts]
        if not (cells_with_most_counts.ix[0]['TruePdg'] in self.target_particles):
            warnings.warn("For the majority of cells in this cluster expected {} found {}".format(self.target_particles, cells_with_most_counts.ix[0]['TruePdg']))
            
        px = cells_with_most_counts.ix[0]['TrueP4[0]']
        py = cells_with_most_counts.ix[0]['TrueP4[1]']
        pz = cells_with_most_counts.ix[0]['TrueP4[2]']
        prong_energy = np.sqrt(px**2 + py**2 + pz**2)
        return prong_energy

class ElectronVertices(object):
    def __init__(self):
        self.dim = (2,) # CellView0, CellView1
        self.name = 'electron_vertices'
        self.prong_level = True
        self.target_particles = [-11, 11]
        
    def get_data_as_array(self, the_event_data):
        new_array = self.get_vertices(the_event_data)
        return new_array/100.
    
    @staticmethod
    def get_vertices(the_event_data):
        assert the_event_data is not np.nan

        # check if there is a data point, otherwise put the middle of the range
        if the_event_data[the_event_data['View']==0].loc[:, ['VtxCell']].shape[0] != 0:
            cell_view_0 = the_event_data[the_event_data['View']==0].loc[:, ['VtxCell']].iloc[0]
        else:
            cell_view_0 = np.array([192]) # middle

        if the_event_data[the_event_data['View']==1].loc[:, ['VtxCell']].shape[0] != 0:
            cell_view_1 = the_event_data[the_event_data['View']==1].loc[:, ['VtxCell']].iloc[0]
        else:
            cell_view_1 = np.array([192]) # middle

        return np.concatenate([cell_view_0, cell_view_1])
    
class ElectronDirection(object):
    def __init__(self):
        self.dim = (3,)
        self.name = 'electron_direction'
        self.prong_level = True
        self.target_particles = [-11, 11]
        
    def get_data_as_array(self, the_event_data):
        prong = the_event_data
        true_dir = np.array([prong.head(1)['TrueP4[2]']*3.8,
                             prong.head(1)['TrueP4[0]']*5.9,
                             prong.head(1)['TrueP4[1]']*5.9]).squeeze()
        return true_dir
    
class ElectronRecoDirection(object):
    def __init__(self):
        self.dim = (3,)
        self.name = 'electron_reco_direction'
        self.prong_level = True
        self.target_particles = [-11, 11]
        
    def get_data_as_array(self, the_event_data):
        prong = the_event_data
        reco_dir = np.array([prong.head(1)['SheDir[2]']*3.8,
                             prong.head(1)['SheDir[0]']*5.9,
                             prong.head(1)['SheDir[1]']*5.9,]).squeeze()
        return reco_dir

class EventDirection(object):
    def __init__(self):
        self.dim = (3,)
        self.name = 'event_direction'
        self.prong_level = False
        
    def get_data_as_array(self, the_event_data):
#         idxs = (the_event_data.TruePdg == 11) | (the_event_data.TruePdg == -11)
        idxs = (the_event_data.TruePdg == 11)
        direction = the_event_data.loc[idxs][['TrueP4[2]', 'TrueP4[0]', 'TrueP4[1]']].drop_duplicates()
        if len(direction) == 0:
            direction = [np.nan, np.nan, np.nan]
#             print("No electron found. Only have {}".format(the_event_data.TruePdg.unique()))
        elif len(direction) > 1:
            print("More than one electron. Taking first.")
            direction = direction.iloc[0]
        direction_array = np.array(direction).squeeze() * np.array([3.8, 5.9, 5.9])
        try:
            direction_normed = direction_array/direction_array[2]
        except IndexError:
            print(direction_array)
            raise IndexError
        
        return direction_normed
    
class EventRecoDirection(object):
    def __init__(self):
        self.dim = (3,)
        self.name = 'event_reco_direction'
        self.prong_level = False
        
    def get_data_as_array(self, the_event_data):
        idxs = (the_event_data.TruePdg == 11) | (the_event_data.TruePdg == -11)
        direction = the_event_data.loc[idxs][['SheDir[2]', 'SheDir[0]', 'SheDir[1]']].drop_duplicates()
        if len(direction) == 0:
            direction = [np.nan, np.nan, np.nan]
        elif len(direction) > 1:
            print("More than one electron. Taking first.")
            direction = direction.iloc[0]
        direction_array = np.array(direction).squeeze() * np.array([3.8, 5.9, 5.9])
        try:
            direction_normed = direction_array/direction_array[2]
        except IndexError:
            print(direction_array)
            raise IndexError
        
        return direction_normed


