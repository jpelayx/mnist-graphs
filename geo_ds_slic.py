
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
import time

try:
    from compute_features import color_features
except ImportError:
    extension_availabe = False
else:
    extension_availabe = True

import torchvision.transforms as T

import os.path as osp
import csv
import re

from skimage.io import imread
from skimage.color import gray2rgb
from PIL import Image
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

class SuperPixelGraphGeo(Dataset):
    ds_name = 'GeoSLIC'
    num_classes = 45
    label_dict = {
        'reference map' : 0,
        'geology sketch' : 1,
        'photomicrograph' : 2,
        'profile' : 3,
        'plant and project sketch' : 4,
        'geological map' : 5,
        'table' : 6,
        'outcrop photograph' : 7,
        'hand sample photograph' : 8,
        'bar chart' : 9,
        'oil ship photograph' : 10,
        'fluxogram' : 11,
        'equipment sketch' : 12,
        'geological chart' : 13,
        'person portrait photograph' : 14,
        'submarine arrangement' : 15,
        'van krevelen diagram' : 16,
        'stratigraphic isoattribute map' : 17,
        'completion scheme' : 18,
        'seismic section' : 19,
        '3d block diagram' : 20,
        'geological cross section' : 21,
        'scatter plot' : 22,
        'equipment photograph' : 23,
        'well core photograph' : 24,
        'temperature map' : 25,
        'sattelite image' : 26,
        'variogram' : 27,
        'box plot' : 28,
        'oil rig photograph' : 29,
        'scanning electron microscope image' : 30,
        'chromatogram' : 31,
        'line graph' : 32,
        'stereogram' : 33,
        'geophysical map' : 34,
        'ternary diagram' : 35,
        '3d visualization' : 36,
        'radar chart' : 37,
        'structure contour map' : 38,
        'seismic cube' : 39,
        'diffractogram' : 40,
        'aerial photograph' : 41,
        'rose diagram' : 42,
        'microfossil photograph' : 43,
        'geotectonic map' : 44
    }
    std_features = ['avg_color',
                    'std_deviation_color',
                    'centroid',
                    'std_deviation_centroid']
    # graph types 
    graph_types_dict = {'RAG' : 0,
                        '1NNSpatial' : 1,
                        '2NNSpatial' : 2,
                        '4NNSpatial' : 3,
                        '8NNSpatial' : 4,
                        '16NNSpatial': 5,
                        '1NNFeature' : 6,
                        '2NNFeature' : 7,
                        '4NNFeature' : 8,
                        '8NNFeature' : 9,
                        '16NNFeature': 10 }
    # slic methods
    slic_methods_dict = {'SLIC0': 0,
                         'SLIC': 1,
                         'grid': 2 }

    def __init__(self,
                 ds_path,
                 root=None, 
                 n_segments=75, 
                 compactness=0.1, 
                 features=None, 
                 graph_type='RAG', 
                 slic_method='SLIC0', 
                 pre_select_features=False):
        self.ds_path = ds_path
        self.n_segments = n_segments
        self.compactness = compactness
        self.features = features if features is not None else self.std_features
        self.slic_method = slic_method
        self.graph_type = graph_type
        self.select_features()
        self.pre_select_features = pre_select_features

        self.root = self.get_ds_name()


        self.load_index_info()

        self.avg_num_nodes = None
        self.std_deviation_num_nodes = None
        self.avg_num_edges = None
        self.std_deviation_num_edges = None
        self.loading_time = None

        transform = None if self.pre_select_features else self.filter_features
        super().__init__(self.root, transform=transform)

        print(self.ds_name + " Loaded.")
        print(f"Average number of nodes: {self.avg_num_nodes} with standard deviation {self.std_deviation_num_nodes}")
        print(f"Average number of edges: {self.avg_num_edges} with standard deviation {self.std_deviation_num_edges}")

    def get_ds_name(self):
        
        if self.pre_select_features:
            self.features.sort()
            name = './geo_ds/n{}-c{}-{}'.format(self.n_segments, 
                                                self.graph_type,
                                                self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + str(self.compactness),
                                                '-'.join(self.features))
        
        else:
            name = './geo_ds/n{}-{}-{}'.format(self.n_segments, 
                                            self.graph_type,
                                            self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + str(self.compactness))
        print(name)
        return name

    
    def load_index_info(self):
        index_path = self.ds_path + '/labels/v14-one-tree-test.csv'
        self.targets = np.ndarray((self.len(), 1), dtype=np.int32)
        self.raw_path_list = []
        self.processed_path_list = []
        with open(index_path, newline='') as index_file:
            index = csv.reader(index_file)
            next(index)
            idx = 0
            for path, label in index:
                self.targets[idx] = self.label_dict[label]
                path = re.search('.*/images/by\-hash(.*)', path).group(1)
                self.raw_path_list.append(path)
                self.processed_path_list.append(f'data_{idx}.pt')
                idx += 1
        self.targets = torch.from_numpy(self.targets)

    def get_labels(self):
        return list(range(self.num_classes))

    def get_targets(self):
        return self.targets

    def select_features(self):
        self.features_mask = []
        self.features_dict = {}
        self.add_feature('avg_color')
        self.add_feature('std_deviation_color')
        self.add_feature('centroid')
        self.add_feature('std_deviation_centroid')
        self.add_feature('num_pixels')
        self.add_feature('avg_color_hsv')
        self.add_feature('std_deviation_color_hsv')
        self.print_features()
    
    def add_feature(self, feature):
        f = feature in self.features
        if 'color' in feature:
            self.features_mask.append(f)
            self.features_mask.append(f)
            self.features_mask.append(f)
        elif 'centroid' in feature:
            self.features_mask.append(f)
            self.features_mask.append(f)
        else:
            self.features_mask.append(f)
        self.features_dict[feature] = f
    
    def print_features(self):
        print('Selected features for ' + self.graph_type + ' graph:')
        for feature in self.features_dict:
            if self.features_dict[feature]:
                print('\t+ ' + feature)
    
    def filter_features(self, data):
        x_trans = data.x.numpy()
        x_trans = x_trans[:, self.features_mask]
        data.x = torch.from_numpy(x_trans).to(torch.float)
        return data

    def create_data_obj_ext(self, idx, path):
        y = torch.tensor([self.targets[idx]])
        img_np = self.load_img(path)
        features, edge_index, _ = color_features(img_np,
                                                    self.n_segments, 
                                                    self.graph_types_dict[self.graph_type], 
                                                    self.slic_methods_dict[self.slic_method], 
                                                    self.compactness)
        pos = features[:, 6:8]
        return Data(x=torch.from_numpy(features).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float), y=y)

    def load_img(self, path):
        full_path = self.ds_path + '/images/by-hash' + path 

        img_pil = Image.open(full_path) 
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        img_np = np.asarray(img_pil, dtype=np.float32)/255.0
        return img_np

    @property
    def raw_file_names(self):
        return self.raw_path_list

    @property
    def processed_file_names(self):
        return self.processed_path_list
    
    def len(self):
        return 25725
    
    def process(self):
        num_nodes = np.ndarray((self.len(), 1), dtype=np.int32)
        num_edges = np.ndarray((self.len(), 1), dtype=np.int32)
        idx = 0
        t = time.time()
        for raw_path in self.raw_paths:
            if not osp.exists(osp.join(self.root, 'processed', f'data_{idx}.pt')):
                data = self.create_data_obj_ext(idx, raw_path)
                torch.save(data, osp.join(self.root, 'processed', f'data_{idx}.pt'))
            else:
                data = self.get(idx)
            num_nodes[idx] = data.num_nodes
            num_edges[idx] = data.num_edges
            idx += 1
        self.loading_time = time.time() - t
        self.avg_num_nodes = np.mean(num_nodes)
        self.std_deviation_num_nodes = np.std(num_nodes)
        self.avg_num_edges = np.mean(num_edges)
        self.std_deviation_num_edges = np.std(num_edges)
        

    def get(self, idx):
        data = torch.load(self.root + '/processed' + f'/data_{idx}.pt')
        return data