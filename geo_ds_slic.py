
from color_slic import ColorSLIC

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
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
from PIL import Image


class SuperPixelGraphGeo(ColorSLIC):
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

    def __init__(self, ds_path=None, root=None, n_segments=75, compactness=0.1, features=None, graph_type='RAG', slic_method='SLIC0', train=True, pre_select_features=False):
        if ds_path is None:
            self.ds_path = '/home/julia/Documents/ds'
        else:
            self.ds_path = ds_path
        super().__init__(root, n_segments, compactness, features, graph_type, slic_method, train, pre_select_features)

    def get_ds_name(self):
        return  './geo_ds/n{}-{}-{}'.format(self.n_segments, 
                                            self.graph_type,
                                            self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + self.compactness)
    def get_ds_name_with_features(self):
        self.features.sort()
        return  './geo_ds/n{}-c{}-{}'.format(self.n_segments, 
                                             self.graph_type,
                                             self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + self.compactness,
                                             '-'.join(self.features))
    def get_labels(self):
        return list(range(self.num_classes))

    def load(self):
        index_path = self.ds_path + '/labels/v14-one-tree.csv'
        img_dir = self.ds_path + '/images/by-hash'
        data_list = []
        t = time.time()
        with open(index_path, newline='') as index_file:
            index = csv.reader(index_file)
            next(index)
            for path, label in index:
                y = self.label_dict[label] 

                path = re.search('.*/images/by\-hash(.*)', path).group(1)
                img_pil = Image.open(img_dir + path)
                if(img_pil.mode != 'RGB'):
                    img_pil = img_pil.convert('RGB')
                img_np = np.asarray(img_pil, dtype=np.float32)/255.00

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

        