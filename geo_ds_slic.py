
from color_slic import ColorSLIC

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

import csv
import re

from skimage.io import imread
from skimage.color import gray2rgb
from PIL import Image

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
                 index_path,
                 root=None, 
                 n_segments=75, 
                 compactness=0.1, 
                 features=None, 
                 graph_type='RAG', 
                 slic_method='SLIC0', 
                 pre_select_features=False):

        self.is_pre_loaded = True
        
        self.ds_path = ds_path
        self.index_path = index_path
        self.n_segments = n_segments
        self.compactness = compactness
        self.features = features if features is not None else self.std_features
        self.select_features()
        self.slic_method = self.slic_methods_dict[slic_method]
        self.graph_type = self.graph_types_dict[graph_type]
        self.pre_select_features = pre_select_features
        if root is None:
            self.root = self.get_ds_name()
        else:
            self.root = root

        self.load_index_info()

        transform = None if self.pre_select_features else self.filter_features
        super().__init__(root, transform=transform)

        self.get_stats()
        print(self.ds_name + " Loaded.")
        print(f"Average number of nodes: {self.avg_num_nodes} with standard deviation {self.std_deviation_num_nodes}")
        print(f"Average number of edges: {self.avg_num_edges} with standard deviation {self.std_deviation_num_edges}")

    def get_ds_name(self):
        if self.pre_select_features:
            self.features.sort()
            return  './geo_ds/n{}-c{}-{}'.format(self.n_segments, 
                                                self.graph_type,
                                                self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + self.compactness,
                                                '-'.join(self.features))
        else:
            return  './geo_ds/n{}-{}-{}'.format(self.n_segments, 
                                                self.graph_type,
                                                self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + self.compactness)
    def load_index_info(self):
        with open(self.index_path, newline='') as index_file:
            index = csv.reader(index_file)
            next(index)
            targets = np.ndarray(())

    def get_labels(self):
        return list(range(self.num_classes))

    def load(self):
        index_path = self.ds_path + '/labels/v14-one-tree-test.csv'
        img_dir = self.ds_path + '/images/by-hash'
        data_list = []
        t = time.time()
        with open(index_path, newline='') as index_file:
            index = csv.reader(index_file)
            next(index)
            for path, label in index:
                y = self.label_dict[label] 

                path = re.search('.*/images/by\-hash(.*)', path).group(1)
                img_np = imread(img_dir + path)
                if len(img_np.shape) < 3: 
                    img_np = gray2rgb(img_np)
                if len(img_np.shape) > 3 or (len(img_np.shape) == 3 and img_np.shape[2] != 3):
                    with Image.open(img_dir + path) as img_pil:
                        img_pil = img_pil.convert('RGB')
                        img_np = np.asarray(img_pil, dtype=np.float32)/255.0

                features, edge_index, _ = color_features(img_np,
                                                        self.n_segments, 
                                                        self.graph_types_dict[self.graph_type], 
                                                        self.slic_methods_dict[self.slic_method], 
                                                        self.compactness)

                pos = features[:, 6:8]
                data_list.append(Data(x=torch.from_numpy(features).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float), y=y))
        t = time.time() - t
        print(f'Done in {t}s')
        self.save_stats(data_list)
        return self.collate(data_list)

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

    def create_data_obj_ext(self, d):
            img, y = d
            img_np = torch.stack([img[0], img[1], img[2]], dim=2).numpy()
            features, edge_index, _ = color_features(img_np,
                                                     self.n_segments, 
                                                     self.graph_types_dict[self.graph_type], 
                                                     self.slic_methods_dict[self.slic_method], 
                                                     self.compactness)
            pos = features[:, 6:8]
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
    def raw_file_names(self):
        

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data, slices = self.load()
        torch.save((data, slices), self.processed_paths[0])