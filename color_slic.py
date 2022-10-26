import torch
import torchvision.datasets as datasets 
import torchvision.transforms as T
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import rgb2hsv
import skimage as ski
import time

from compute_features import color_features

class ColorSLIC(InMemoryDataset):
    std_features = ['avg_color',
                    'std_deviation_color',
                    'centroid',
                    'std_deviation_centroid']
    implemented_features = ['avg_color',
                            'std_deviation_color',
                            'avg_color_hsv',
                            'std_deviation_color_hsv',
                            'avg_lightness',
                            'std_deviation_lightness',
                            'centroid',
                            'std_deviation_centroid',
                            'num_pixels']
    ds_name = 'ColorSLIC'
    def __init__(self, 
                 root=None, 
                 n_segments= 75,  
                 compactness = 0.1, 
                 features = None, 
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
    
    def load_data(self):
        raise NotImplementedError

    def get_targets(self):
        return torch.cat([d.y for d in self])

    def select_features(self):
        self.features_mask = []
        self.features_mask.append('avg_color' in self.features)
        self.features_mask.append('avg_color' in self.features)
        self.features_mask.append('avg_color' in self.features)
        if self.features_mask[-1]:
            print('\t+ avg_color')
        self.features_mask.append('std_deviation_color' in self.features)
        self.features_mask.append('std_deviation_color' in self.features)
        self.features_mask.append('std_deviation_color' in self.features)
        if self.features_mask[-1]:
            print('\t+ std_deviation_color')
        self.features_mask.append('centroid' in self.features)
        self.features_mask.append('centroid' in self.features)
        if self.features_mask[-1]:
            print('\t+ centroid')
        self.features_mask.append('std_deviation_centroid' in self.features)
        self.features_mask.append('std_deviation_centroid' in self.features)
        if self.features_mask[-1]:
            print('\t+ std_deviation_centroid')
        self.features_mask.append('num_pixels' in self.features)
        if self.features_mask[-1]:
            print('\t+ num_pixels')
        self.features_mask.append('avg_color_hsv' in self.features)
        self.features_mask.append('avg_color_hsv' in self.features)
        self.features_mask.append('avg_color_hsv' in self.features)
        if self.features_mask[-1]:
            print('\t+ avg_color_hsv')
        self.features_mask.append('std_deviation_color_hsv' in self.features)
        self.features_mask.append('std_deviation_color_hsv' in self.features)
        self.features_mask.append('std_deviation_color_hsv' in self.features)
        if self.features_mask[-1]:
            print('\t+ std_deviation_color_hsv')

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

    def create_data_obj(self, d):
            img, y = d
            img_np = torch.stack([img[0], img[1], img[2]], dim=2).numpy()
            features, edge_index = color_features(img_np, self.n_segments, self.compactness)
            pos = features[:, 6:8]
            features = features[:,self.features_mask]
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