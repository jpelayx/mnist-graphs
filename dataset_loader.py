import mnist_slic 
import fashion_mnist_slic
import cifar10_slic
import cifar100_slic
import stl10_slic
import stanfordcars_slic
import geo_ds_slic
import torchvision.datasets as datasets
 
import numpy as np
import torch
from torch.utils.data import ConcatDataset
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold

import argparse

random_seed = 42

def load_dataset(args):
    """
    loads dataset with specified parameters and returns:

    + dataset: torch's ConcatDataset with 1 or 2 datasets
    + splits: as returned by sklearn's StratifiedKFold split
    + labels: int list with dataset's labels 
    """
    params = get_dataset_params(args)

    try:
        load_graphs = args.model != 'CNN'
    except:
        load_graphs = True
    
    if load_graphs:
        return load_graph_ds(params)
    else:
        return load_image_ds(params)

def load_graph_ds(params):
    dataset = params['dataset']
    n_splits = params['n_splits']
    ds = None
    if dataset == 'mnist':
        test_ds  = mnist_slic.SuperPixelGraphMNIST(root=None, 
                                                   n_segments=params['n_segments'],
                                                   compactness=params['compactness'],
                                                   features=params['features'],
                                                   graph_type=params['graph_type'],
                                                   slic_method=params['slic_method'],
                                                   train=False,
                                                   pre_select_features=params['pre_select_features'])
        train_ds = mnist_slic.SuperPixelGraphMNIST(root=None, 
                                                   n_segments=params['n_segments'],
                                                   compactness=params['compactness'],
                                                   features=params['features'],
                                                   graph_type=params['graph_type'],
                                                   slic_method=params['slic_method'],
                                                   train=True,
                                                   pre_select_features=params['pre_select_features'])
    elif dataset == 'fashion_mnist':
        test_ds  = fashion_mnist_slic.SuperPixelGraphFashionMNIST(root=None, 
                                                   n_segments=params['n_segments'],
                                                   compactness=params['compactness'],
                                                   features=params['features'],
                                                   graph_type=params['graph_type'],
                                                   slic_method=params['slic_method'],
                                                   train=False,
                                                   pre_select_features=params['pre_select_features'])
        train_ds = fashion_mnist_slic.SuperPixelGraphFashionMNIST(root=None, 
                                                   n_segments=params['n_segments'],
                                                   compactness=params['compactness'],
                                                   features=params['features'],
                                                   graph_type=params['graph_type'],
                                                   slic_method=params['slic_method'],
                                                   train=True,
                                                   pre_select_features=params['pre_select_features'])
    elif dataset == 'cifar10':
        test_ds  = cifar10_slic.SuperPixelGraphCIFAR10(root=None, 
                                                       n_segments=params['n_segments'],
                                                       compactness=params['compactness'],
                                                       features=params['features'],
                                                       graph_type=params['graph_type'],
                                                       slic_method=params['slic_method'],
                                                       train=False,
                                                       pre_select_features=params['pre_select_features'])
        train_ds = cifar10_slic.SuperPixelGraphCIFAR10(root=None, 
                                                       n_segments=params['n_segments'],
                                                       compactness=params['compactness'],
                                                       features=params['features'],
                                                       graph_type=params['graph_type'],
                                                       slic_method=params['slic_method'],
                                                       train=True,
                                                       pre_select_features=params['pre_select_features'])
    elif dataset == 'cifar100':
        test_ds  = cifar100_slic.SuperPixelGraphCIFAR100(root=None, 
                                                       n_segments=params['n_segments'],
                                                       compactness=params['compactness'],
                                                       features=params['features'],
                                                       graph_type=params['graph_type'],
                                                       slic_method=params['slic_method'],
                                                       train=False,
                                                       pre_select_features=params['pre_select_features'])
        train_ds = cifar100_slic.SuperPixelGraphCIFAR100(root=None, 
                                                       n_segments=params['n_segments'],
                                                       compactness=params['compactness'],
                                                       features=params['features'],
                                                       graph_type=params['graph_type'],
                                                       slic_method=params['slic_method'],
                                                       train=True,
                                                       pre_select_features=params['pre_select_features'])
    elif dataset == 'stl10':
        test_ds  = stl10_slic.SuperPixelGraphSTL10(root=None, 
                                                       n_segments=params['n_segments'],
                                                       compactness=params['compactness'],
                                                       features=params['features'],
                                                       graph_type=params['graph_type'],
                                                       slic_method=params['slic_method'],
                                                       train=False,
                                                       pre_select_features=params['pre_select_features'])
        train_ds = stl10_slic.SuperPixelGraphSTL10(root=None, 
                                                       n_segments=params['n_segments'],
                                                       compactness=params['compactness'],
                                                       features=params['features'],
                                                       graph_type=params['graph_type'],
                                                       slic_method=params['slic_method'],
                                                       train=True,
                                                       pre_select_features=params['pre_select_features'])
    elif dataset == 'stanfordcars':
        test_ds  = stanfordcars_slic.SuperPixelGraphStanfordCars(root=None, 
                                                       n_segments=params['n_segments'],
                                                       compactness=params['compactness'],
                                                       features=params['features'],
                                                       graph_type=params['graph_type'],
                                                       slic_method=params['slic_method'],
                                                       train=False,
                                                       pre_select_features=params['pre_select_features'])
        train_ds = stanfordcars_slic.SuperPixelGraphStanfordCars(root=None, 
                                                       n_segments=params['n_segments'],
                                                       compactness=params['compactness'],
                                                       features=params['features'],
                                                       graph_type=params['graph_type'],
                                                       slic_method=params['slic_method'],
                                                       train=True,
                                                       pre_select_features=params['pre_select_features'])
    elif dataset == 'geo_ds':
        ds = geo_ds_slic.SuperPixelGraphGeo('/home/julia/Documents/ds',
                                            root=None,
                                            n_segments=params['n_segments'],
                                            compactness=params['compactness'],
                                            features=params['features'],
                                            graph_type=params['graph_type'],
                                            slic_method=params['slic_method'],
                                            pre_select_features=params['pre_select_features'])
    else:
        print('No dataset called: \"' + dataset + '\" available.')
        return None
    if ds is None: 
        targets = torch.cat([train_ds.get_targets(), test_ds.get_targets()])
        ds = ConcatDataset([train_ds, test_ds])
    else:
        targets = ds.get_targets()
        ds = ConcatDataset([ds])
    splits = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True).split(np.zeros(len(targets)), targets)
    return ds, splits, targets

def load_image_ds(params):
    dataset = params['dataset']
    if dataset == 'mnist':
        train_ds = datasets.MNIST('./mnist/train', train=True, download=True, transform=T.ToTensor())
        test_ds = datasets.MNIST('./mnist/test', train=False, download=True, transform=T.ToTensor())
        targets = torch.cat((train_ds.targets, test_ds.targets))
        ds = ConcatDataset([train_ds, test_ds])
    elif dataset == 'fashion_mnist':
        train_ds = datasets.FashionMNIST('./fashion_mnist/train', train=True, download=True, transform=T.ToTensor())
        test_ds  = datasets.FashionMNIST('./fashion_mnist/test', train=False, download=True, transform=T.ToTensor())
        targets = torch.cat((train_ds.targets, test_ds.targets))
        ds = ConcatDataset([train_ds, test_ds])
    elif dataset == 'cifar10':
        train_ds = datasets.CIFAR10('./cifar10/train', train=True, download=True, transform=T.ToTensor())
        test_ds  = datasets.CIFAR10('./cifar10/test', train=False, download=True, transform=T.ToTensor())
        targets = torch.cat((torch.tensor(train_ds.targets), torch.tensor(test_ds.targets)))
        ds = ConcatDataset([train_ds, test_ds])
    elif dataset == 'cifar100':
        train_ds = datasets.CIFAR100('./cifar100/train', train=True, download=True, transform=T.ToTensor())
        test_ds  = datasets.CIFAR100('./cifar100/test', train=False, download=True, transform=T.ToTensor())
        targets = torch.cat((torch.tensor(train_ds.targets), torch.tensor(test_ds.targets)))
        ds = ConcatDataset([train_ds, test_ds])
    elif dataset == 'stl10':
        train_ds = datasets.STL10('./stl10/train', split='train', download=True, transform=T.ToTensor())
        test_ds  = datasets.STL10('./stl10/test', split='test', download=True, transform=T.ToTensor())
        targets = torch.cat((torch.from_numpy(train_ds.labels), torch.from_numpy(test_ds.labels)))
        ds = ConcatDataset([train_ds, test_ds])
    else:
        print('No dataset called: \"' + dataset + '\" available.')
        return None
    splits = StratifiedKFold(n_splits=5, random_state=random_seed, shuffle=True).split(np.zeros(len(targets)), targets)
    return ds, splits, targets

def dataset_info(args):
    info = {}
    if args.dataset == 'mnist':
        num_classes = 10
        num_channels = 1
        img_size = 28
    elif args.dataset == 'fashion_mnist':
        num_classes = 10
        num_channels = 1
        img_size = 28
    elif args.dataset == 'cifar10':
        num_classes = 10
        num_channels = 3
        img_size = 32
    elif args.dataset == 'cifar100':
        num_classes = 100
        num_channels = 3
        img_size = 32
    elif args.dataset == 'stl10':
        num_classes = 10
        num_channels = 3
        img_size = 96
    else:
        return None
    info['classes'] = num_classes
    info['channels'] = num_channels
    info['image size'] = img_size
    return info


def set_dataset_arguments(parser):
    parser.add_argument("--n_splits", type=int, default=5,
                        help="number of splits in StratifiedKFold cross validation")
    parser.add_argument("--n_segments", type=int, default=75,
                        help="aproximate number of graph nodes. (default: 75)")
    parser.add_argument("--compactness", type=float, default=0.1,
                        help="compactness for SLIC algorithm. (default: 0.1)")
    parser.add_argument("--graph_type", type=str, default='RAG',
                        help="RAG, (1 | 2 | 4 | 8 | 16)NNSpatial or (1 | 2 | 4 | 8 | 16)NNFeatures")
    parser.add_argument("--slic_method", type=str, default='SLIC0',
                        help="SLIC0, SLIC")
    parser.add_argument("--features", type=str, default=None,
                        help="space separated list of features. options are: avg_color, std_deviation_color, centroid, std_deviation_centroid, num_pixels. (default: avg_color centroid)")
    parser.add_argument("--dataset", "-ds", default='mnist',
                        help="dataset to train against")
    parser.add_argument("--pre_select_features", action='store_true',
                        help="only save selected features when loading dataset")

    return parser

def get_dataset_params(args):
    ds_params = {}
    ds_params['n_splits'] = args.n_splits
    ds_params['n_segments'] = args.n_segments
    ds_params['compactness'] = args.compactness
    ds_params['graph_type'] = args.graph_type
    ds_params['slic_method'] = args.slic_method
    ds_params['dataset'] = args.dataset
    ds_params['pre_select_features'] = args.pre_select_features
    if args.features is not None:
        args.features = args.features.split()
    ds_params['features'] = args.features
    return ds_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = set_dataset_arguments(parser)
    args = parser.parse_args()

    params = get_dataset_params(args)
    ds, splits, labels = load_graph_ds(params)
    
    print('Done.')
