import torch
import torchvision.datasets as datasets 
import torchvision.transforms as T
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import skimage as ski
import time

class RGBSLIC(InMemoryDataset):
    std_features = ['avg_color',
                    'std_deviation_color',
                    'centroid',
                    'std_deviation_centroid']
    implemented_features = ['avg_color',
                            'std_deviation_color',
                            'centroid',
                            'std_deviation_centroid',
                            'num_pixels']
    ds_name = 'RGBSLIC'
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
        self.get_avg_color = 'avg_color' in self.features
        if self.get_avg_color:
            print('\t+ avg_color')
        self.get_std_dev_color = 'std_deviation_color' in self.features
        if self.get_std_dev_color:
            print('\t+ std_deviation_color')
        self.get_centroid = 'centroid' in self.features
        if self.get_centroid:
            print('\t+ centroid')
        self.get_std_dev_centroid = 'std_deviation_centroid' in self.features
        if self.get_std_dev_centroid:
            print('\t+ std_deviation_centroid')
        self.get_num_pixels = 'num_pixels' in self.features
        if self.get_num_pixels:
            print('\t+ num_pixels')

    def load(self):
        self.is_pre_loaded = False
        data = self.load_data()
        img_total = data.data.shape[0]
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
            _, dim0, dim1 = img.shape
            img_np = torch.stack([img[0], img[1], img[2]], dim=2).numpy()
            s = slic(img_np, self.n_segments, self.compactness, start_label=0)
            # rag_mean_color() fails when image is segmented into 1 superpixel 
            if np.any(s):
                g = ski.future.graph.rag_mean_color(img_np, s)
                n = g.number_of_nodes()
                edge_index = torch.from_numpy(np.array(g.edges).T).to(torch.long)
            else: 
                n = 1
                edge_index = torch.tensor([]).to(torch.long)
            s1 = np.zeros([n, 3])  # for mean color and std deviation
            s2 = np.zeros([n, 3])  # for std deviation
            pos1 = np.zeros([n, 2]) # for centroid
            pos2 = np.zeros([n, 2]) # for centroid std deviation
            num_pixels = np.zeros([n, 1])
            for idx in range(dim0 * dim1):
                    idx_i, idx_j = idx % dim0, int(idx / dim0)
                    node = s[idx_i][idx_j] - 1
                    s1[node][0]  += img_np[idx_i][idx_j][0]
                    s2[node][0]  += pow(img_np[idx_i][idx_j][0], 2)
                    s1[node][1]  += img_np[idx_i][idx_j][1]
                    s2[node][1]  += pow(img_np[idx_i][idx_j][1], 2)
                    s1[node][2]  += img_np[idx_i][idx_j][2]
                    s2[node][2]  += pow(img_np[idx_i][idx_j][2], 2)
                    pos1[node][0] += idx_i
                    pos1[node][1] += idx_j
                    pos2[node][0] += pow(idx_i, 2)
                    pos2[node][1] += pow(idx_j, 2)
                    num_pixels[node][0] += 1
            x = []
            if self.get_std_dev_color or self.get_avg_color:
                s1 = s1/num_pixels
                if self.get_avg_color:
                    avg_color = torch.from_numpy(s1).to(torch.float)
                    x.append(avg_color[:,0])
                    x.append(avg_color[:,1])
                    x.append(avg_color[:,2])
            if self.get_std_dev_color:
                s2 = s2/num_pixels
                std_dev = torch.from_numpy(np.sqrt(np.abs((s2 - s1*s1)))).to(torch.float)
                x.append(std_dev[:,0])
                x.append(std_dev[:,1])
                x.append(std_dev[:,2])
            pos1 = pos1/num_pixels
            pos = torch.from_numpy(pos1).to(torch.float)
            if self.get_centroid:
                x.append(pos[:,0])
                x.append(pos[:,1])
            if self.get_std_dev_centroid:
                pos2 = pos2/num_pixels
                std_dev_centroid = torch.from_numpy(np.sqrt(np.abs(pos2 - pos1*pos1))).to(torch.float)
                x.append(std_dev_centroid[:,0])
                x.append(std_dev_centroid[:,1])
            if self.get_num_pixels:
                x.append(torch.from_numpy(num_pixels.flatten()).to(torch.float))
            return Data(x=torch.stack(x, dim=1), edge_index=edge_index, pos=pos, y=y)

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