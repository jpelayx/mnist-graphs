import mnist_slic 
import fashion_mnist_slic
import cifar10_slic
import cifar100_slic
import stl10_slic
import stanfordcars_slic
import geo_ds_slic
 
import torch
from torch.utils.data import ConcatDataset


def load_dataset(n_segments, compactness, features, graph_type, slic_method, dataset, pre_select_features):
    if dataset == 'mnist':
        test_ds  = mnist_slic.SuperPixelGraphMNIST(root=None, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   graph_type=graph_type,
                                                   slic_method=slic_method,
                                                   train=False,
                                                   pre_select_features=pre_select_features)
        train_ds = mnist_slic.SuperPixelGraphMNIST(root=None, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   graph_type=graph_type,
                                                   slic_method=slic_method,
                                                   train=True,
                                                   pre_select_features=pre_select_features)
    elif dataset == 'fashion_mnist':
        test_ds  = fashion_mnist_slic.SuperPixelGraphFashionMNIST(root=None, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   graph_type=graph_type,
                                                   slic_method=slic_method,
                                                   train=False,
                                                   pre_select_features=pre_select_features)
        train_ds = fashion_mnist_slic.SuperPixelGraphFashionMNIST(root=None, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   graph_type=graph_type,
                                                   slic_method=slic_method,
                                                   train=True,
                                                   pre_select_features=pre_select_features)
    elif dataset == 'cifar10':
        test_ds  = cifar10_slic.SuperPixelGraphCIFAR10(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=False,
                                                       pre_select_features=pre_select_features)
        train_ds = cifar10_slic.SuperPixelGraphCIFAR10(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=True,
                                                       pre_select_features=pre_select_features)
    elif dataset == 'cifar100':
        test_ds  = cifar100_slic.SuperPixelGraphCIFAR100(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=False,
                                                       pre_select_features=pre_select_features)
        train_ds = cifar100_slic.SuperPixelGraphCIFAR100(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=True,
                                                       pre_select_features=pre_select_features)
    elif dataset == 'stl10':
        test_ds  = stl10_slic.SuperPixelGraphSTL10(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=False,
                                                       pre_select_features=pre_select_features)
        train_ds = stl10_slic.SuperPixelGraphSTL10(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=True,
                                                       pre_select_features=pre_select_features)
    elif dataset == 'stanfordcars':
        test_ds  = stanfordcars_slic.SuperPixelGraphStanfordCars(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=False,
                                                       pre_select_features=pre_select_features)
        train_ds = stanfordcars_slic.SuperPixelGraphStanfordCars(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=True,
                                                       pre_select_features=pre_select_features)
    elif dataset == 'geo_ds':
        ds = geo_ds_slic.SuperPixelGraphGeo('/home/julia/Documents/ds',
                                            root=None,
                                            n_segments=n_segments,
                                            compactness=compactness,
                                            features=features,
                                            graph_type=graph_type,
                                            slic_method=slic_method,
                                            pre_select_features=pre_select_features)
        return ds
    else:
        print('No dataset called: \"' + dataset + '\" available.')
        return None
        
    ds = ConcatDataset([train_ds, test_ds])
    return ds

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_segments", type=int, default=75,
                        help="aproximate number of graph nodes. (default: 75)")
    parser.add_argument("--compactness", type=float, default=0.1,
                        help="compactness for SLIC algorithm. (default: 0.1)")
    parser.add_argument("--graph_type", type=str, default='RAG',
                        help="RAG, (1 | 2 | 4 | 8 | 16)NNSpatial or (1 | 2 | 4 | 8 | 16)NNFeatures")
    parser.add_argument("--slic_method", type=str, default=0.1,
                        help="SLIC0, SLIC")
    parser.add_argument("--features", type=str, default=None,
                        help="space separated list of features. options are: avg_color, std_deviation_color, centroid, std_deviation_centroid, num_pixels. (default: avg_color centroid)")
    parser.add_argument("--dataset", default='mnist',
                        help="dataset to train against")
    parser.add_argument("--pre_select_features", action='store_true',
                        help="only save selected features when loading dataset")
    args = parser.parse_args()

    if args.features is not None:
        args.features = args.features.split()

    ds = load_dataset(args.n_segments,
                      args.compactness,
                      args.features,
                      args.graph_type, 
                      args.slic_method,
                      args.dataset,
                      args.pre_select_features)
    
    print('Done.')