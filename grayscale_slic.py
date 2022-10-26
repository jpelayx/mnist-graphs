import torch
import torchvision.datasets as datasets 
import torchvision.transforms as T
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import skimage as ski
import networkx as nx
import time

from compute_features import grayscale_features

class GrayscaleSLIC(InMemoryDataset):
    # base class for grayscale datasets 
    # children must implement 
    # get_ds_name: returns a string with the path to the dataset
    # load_data: ...
    # optionally ds_name for pretty printing 
    std_features = ['avg_color',
                    'std_deviation_color',
                    'centroid',
                    'std_deviation_centroid']
    implemented_features = ['avg_color',
                            'std_deviation_color',
                            'centroid',
                            'std_deviation_centroid',
                            'num_pixels',
                            'avg_color_distance',
                            'std_deviation_color_distance']
    ds_name = 'GrayscaleSLIC'
    def __init__(self, 
                 root=None, 
                 n_segments= 75,  
                 compactness = 0.1, 
                 features = None, # possible features are avg_color, centroid, std_deviation_color 
                 train = True):
        self.train = train
        self.n_segments = n_segments
        self.compactness = compactness
        self.features = self.std_features if features is None else features
        self.root = self.get_ds_name() if root is None else root

        self.is_pre_loaded = True
        super().__init__(self.root, None, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        self.get_stats()
        print(self.ds_name + " Loaded.")
        print(f"Average number of nodes: {self.avg_num_nodes} with standard deviation {self.std_deviation_num_nodes}")
        print(f"Average number of edges: {self.avg_num_edges} with standard deviation {self.std_deviation_num_edges}")

    def get_ds_name(self):
        raise NotImplementedError
    
    def get_labels(self):
        raise NotImplementedError
    
    def get_targets(self):
        return torch.cat([d.y for d in self])
    
    def select_features(self):
        # AVG_COLOR 0
        # STD_DEV_COLOR 1
        # CENTROID_I 2
        # CENTROID_J 3
        # STD_DEV_CENTROID_I 4
        # STD_DEV_CENTROID_J 5
        # NUM_PIXELS 6
        self.feature_mask = []
        self.feature_mask.append('avg_color' in self.features)
        if self.feature_mask[-1]:
            print('\t+ avg_color')
        self.feature_mask.append('std_deviation_color' in self.features)
        if self.feature_mask[-1]:
            print('\t+ std_deviation_color')
        self.feature_mask.append('centroid' in self.features)
        self.feature_mask.append('centroid' in self.features)
        if self.feature_mask[-1]:
            print('\t+ centroid')
        self.feature_mask.append('std_deviation_centroid' in self.features)
        self.feature_mask.append('std_deviation_centroid' in self.features)
        if self.feature_mask[-1]:
            print('\t+ std_deviation_centroid')
        self.feature_mask.append('num_pixels' in self.features)
        if self.feature_mask[-1]:
            print('\t+ num_pixels')

    def load(self):
        self.is_pre_loaded = False
        data = self.load_data()
        img_total = len(data)
        print(f'Loading {img_total} images with n_segments = {self.n_segments} ...')
        print(f'Computing features: ')
        self.select_features()

        t = time.time()
        data_list = [self.create_data_obj(d) for d in data]
        t = time.time() - t
        self.loading_time = t
        print(f'Done in {t}s')
        self.save_stats(data_list)
        return self.collate(data_list)

    def load_data(self):
        raise NotImplementedError

    def create_data_obj(self, d):
            img, y = d
            _, dim0, dim1 = img.shape
            img_np = img.view(dim0, dim1).numpy()
            features, edge_index = grayscale_features(img_np, self.n_segments, self.compactness)
            pos = features[:, 2:4]
            features = features[:,self.feature_mask]
            return Data(x=torch.from_numpy(features).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float), y=y)

    def save_stats(self, data):
        nodes = [d.num_nodes for d in data]
        edges = [d.num_edges for d in data]
        self.avg_num_nodes = np.average(nodes)
        self.std_deviation_num_nodes = np.std(nodes)
        self.avg_num_edges = np.average(edges)
        self.std_deviation_num_edges = np.std(edges)
    
    def get_stats(self):
        if self.is_pre_loaded:
            data_list = [self[i] for i in range(len(self))]
            self.save_stats(data_list)
            self.loading_time = 0

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data, slices = self.load()
        torch.save((data, slices), self.processed_paths[0])
    