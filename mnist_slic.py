import torch
import torchvision.datasets as datasets 
import torchvision.transforms as T
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import skimage as ski

from multiprocessing import Pool
import time

def get_ds_name(n_segments, compactness, features, train):
    return  './{}-n{}-c{}-{}'.format('train' if train else 'test', 
                                     n_segments, 
                                     compactness,
                                     '-'.join(features))

class SuperPixelGraphMNIST(InMemoryDataset):
    def __init__(self, 
                 root=None, 
                 n_segments= 75,  
                 compactness = 0.1, 
                 features = None, # possible features are avg_color, centroid, std_deviation_color 
                 train = True):
        self.train = train
        self.n_segments = n_segments
        self.compactness = compactness
        if features is None:
            self.features = ['avg_color', 'centroid']
        else:
            self.features = features
        if root is None:
            self.root = get_ds_name(self.n_segments, self.compactness, self.features, self.train)
        else:
            self.root = root
        super().__init__(self.root, None, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def loadMNIST(self):
        mnist = datasets.MNIST(self.root, train=self.train, download=True, transform=T.ToTensor())
        img_total = mnist.data.shape[0]
        print(f'Loading {img_total} images with n_segments = {self.n_segments} ...')
        t = time.time()
        self.get_avg_color = 'avg_color' in self.features
        self.get_std_deviation_color = 'std_deviation_color' in self.features
        self.get_centroid = 'centroid' in self.features
        with Pool() as p:
            data_list = p.map(self.create_data_obj, mnist)
        t = time.time() - t
        print('Done in {t}s')
        return self.collate(data_list)

    def create_data_obj(self, d):
            img, y = d
            _, dim0, dim1 = img.shape
            img_np = img.view(dim0, dim1).numpy()
            s = slic(img_np, self.n_segments, self.compactness, start_label=0)
            g = ski.future.graph.rag_mean_color(img_np, s)
            n = g.number_of_nodes()
            s1 = np.zeros([n, 1])  # for mean color and std deviation
            s2 = np.zeros([n, 1])  # for std deviation
            pos = np.zeros([n, 2]) # for centroid
            num_pixels = np.zeros([n, 1])
            for idx in range(dim0 * dim1):
                    idx_i, idx_j = idx % dim0, int(idx / dim0)
                    node = s[idx_i][idx_j] - 1
                    s1[node][0]  += img_np[idx_i][idx_j]
                    s2[node][0]  += pow(img_np[idx_i][idx_j], 2)
                    pos[node][0] = (num_pixels[node] * pos[node][0] + idx_i) / (num_pixels[node] + 1)
                    pos[node][1] = (num_pixels[node] * pos[node][1] + idx_j) / (num_pixels[node] + 1)
                    num_pixels[node][0] += 1
            edge_index = torch.from_numpy(np.array(g.edges).T).to(torch.long)
            x = []
            if self.get_std_deviation_color or self.get_avg_color:
                s1 = s1/num_pixels
                if self.get_avg_color:
                    x.append(torch.from_numpy(s1.flatten()).to(torch.float))
            if self.get_std_deviation_color:
                s2 = s2/num_pixels
                std_deviation = np.sqrt(s2 - s1*s1)
                x.append(torch.from_numpy(std_deviation.flatten()).to(torch.float))
            if self.get_centroid:
                pos = torch.from_numpy(pos).to(torch.float)
                x.append(pos[:,0])
                x.append(pos[:,1])
            return Data(x=torch.torch.stack(x, dim=1), edge_index=edge_index, pos=pos, y=y)

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data, slices = self.loadMNIST()
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="load train dataset")
    parser.add_argument("--test", action="store_true",
                        help="load test dataset")
    parser.add_argument("--traindir", default=None, 
                        help="train dataset location")
    parser.add_argument("--testdir", default=None, 
                        help="test dataset location")
    parser.add_argument("--n_segments", type=int, default=75,
                        help="aproximate number of graph nodes. (default: 75)")
    parser.add_argument("--compactness", type=float, default=0.1,
                        help="compactness for SLIC algorithm. (default: 0.1)")
    parser.add_argument("--features", type=str, default=None,
                        help="space separated list of features. options are: avg_color, std_deviation_color, centroid. (default: avg_color centroid)")
    args = parser.parse_args()

    if not args.train and not args.test:
        print("there's nothing to do")
    
    if args.train:
        print('-----------------------------------')
        print('Loading train dataset')
        train_ds = SuperPixelGraphMNIST(root=args.traindir, 
                                        n_segments=args.n_segments,
                                        compactness=args.compactness,
                                        features=args.features,
                                        train=True)
        print('Saved in ', train_ds.root)

    if args.test:
        print('-----------------------------------')
        print('Loading test dataset')
        test_ds = SuperPixelGraphMNIST(root=args.traindir, 
                                       n_segments=args.n_segments,
                                       compactness=args.compactness,
                                       features=args.features,
                                       train=False)
        print('Saved in ', test_ds.root)