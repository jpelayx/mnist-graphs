from matplotlib.pyplot import hist
import torch
from torch.utils.data import ConcatDataset, SubsetRandomSampler
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as T
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

import mnist_slic 
import fashion_mnist_slic
import cifar10_slic
import cifar100_slic
import stl10_slic
import stanfordcars_slic
import geo_ds_slic


class GCN(torch.nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()
        # using architecture inspired by MNISTSuperpixels example 
        # (https://medium.com/@rtsrumi07/understanding-graph-neural-network-with-hands-on-example-part-2-139a691ebeac)
        hidden_channel_size = 64 
        self.initial_conv = GCNConv(data.num_features, hidden_channel_size)
        self.conv1 = GCNConv(hidden_channel_size, hidden_channel_size)
        self.conv2 = GCNConv(hidden_channel_size, hidden_channel_size)
        self.out = nn.Linear(hidden_channel_size*2, data.num_classes)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv1(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = torch.cat([global_mean_pool(hidden, batch_index),
                            global_max_pool(hidden, batch_index)], dim=1)
        out = self.out(hidden)
        return out 

def train(dataloader, model, loss_fn, optimizer, device):
    for _, b in enumerate(dataloader):
        b.to(device)
        pred = model(b.x, b.edge_index, b.batch)
        loss = loss_fn(pred, b.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn, device, labels):
    num_batches = len(dataloader)
    test_loss = 0
    Y, Y_pred = torch.empty(0), torch.empty(0)
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            pred = model(d.x, d.edge_index, d.batch)
            test_loss += loss_fn(pred, d.y).item()
            Y = torch.cat([Y, d.y.to('cpu')])
            Y_pred = torch.cat([Y_pred, pred.to('cpu')])
    test_loss /= num_batches
    Y_pred = torch.argmax(Y_pred, dim=1)
    accuracy = accuracy_score(Y, Y_pred)
    f1_micro = f1_score(Y, Y_pred, average='micro', labels=labels)
    f1_macro = f1_score(Y, Y_pred, average='macro', labels=labels)
    f1_weighted = f1_score(Y, Y_pred, average='weighted', labels=labels)
    return {"Accuracy": accuracy, "F-measure (micro)": f1_micro, "F-measure (macro)": f1_macro, "F-measure (weighted)": f1_weighted, "Avg loss": test_loss}


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
        targets = ds.get_targets()
        splits = StratifiedKFold(n_splits=5).split(np.zeros(len(targets)), targets)
        labels = ds.get_labels()
        return ds, splits, labels
    else:
        print('No dataset called: \"' + dataset + '\" available.')
        return None
        
    labels = test_ds.get_labels()
    ds = ConcatDataset([train_ds, test_ds])
    targets = torch.cat([train_ds.get_targets(), test_ds.get_targets()])
    splits = StratifiedKFold(n_splits=5).split(np.zeros(len(targets)), targets)

    return ds, splits, labels

if __name__ == '__main__':
    import argparse
    import csv
    import time

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
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="model's learning rate")
    parser.add_argument("--dataset", default='mnist',
                        help="dataset to train against")
    parser.add_argument("--pre_select_features", action='store_true',
                        help="only save selected features when loading dataset")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    field_names = ["Epoch", "Accuracy", "F-measure (micro)", "F-measure (macro)", "F-measure (weighted)", "Avg loss"]
    meta_field_names = ['n_segments', 
                        'compactness', 
                        'graph type', 
                        'slic method',
                        'features', 
                        'avg. num. of nodes', 
                        'std. dev. of num. of nodes', 
                        'avg. num. of edges', 
                        'std. dev. of num. of edges', 
                        'accuracy', 
                        'micro', 
                        'macro',
                        'weighted', 
                        'avg. loss', 
                        'training time',
                        'loading time']
    
    if args.features is not None:
        args.features = args.features.split()

    ds, splits, labels = load_dataset(args.n_segments,
                                      args.compactness,
                                      args.features,
                                      args.graph_type, 
                                      args.slic_method,
                                      args.dataset,
                                      args.pre_select_features)
    meta_info = {}
    try:
        train_ds, test_ds = ds.datasets[0], ds.datasets[1]
    except:
        train_ds = ds
    meta_info['loading time'] = train_ds.loading_time
    meta_info['avg. num. of nodes'] = train_ds.avg_num_nodes
    meta_info['std. dev. of num. of nodes'] = train_ds.std_deviation_num_nodes
    meta_info['avg. num. of edges'] = train_ds.avg_num_edges
    meta_info['std. dev. of num. of edges'] = train_ds.std_deviation_num_edges
    meta_info['n_segments']  = train_ds.n_segments
    meta_info['compactness'] = train_ds.compactness
    meta_info['graph type'] =  train_ds.graph_type
    meta_info['slic method'] = train_ds.slic_method
    meta_info['features'] = ' '.join(train_ds.features)
    
    out = './{}/n{}-{}-{}-{}.csv'.format(args.dataset,
                                            train_ds.n_segments,
                                            train_ds.graph_type,
                                            train_ds.slic_method if train_ds.slic_method == 'SLIC0' else train_ds.slic_method + 'c' + str(train_ds.compactness),
                                            '-'.join(train_ds.features))

    meta_out = './{}/training_info.csv'.format(args.dataset)
    with open(out, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.manual_seed(42)

    epochs = args.epochs
    quiet = args.quiet

    history = []
    training_time = []
    for train_index, test_index in splits:
        train_loader = DataLoader(ds, batch_size=64, sampler=SubsetRandomSampler(train_index))
        test_loader  = DataLoader(ds, batch_size=64, sampler=SubsetRandomSampler(test_index))

        model = GCN(train_ds).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        fold_hist = []
        print('------------------------')
        print(f'FOLD {len(history) + 1}/{5}')
        t0 = time.time()
        for t in range(epochs):
            train(train_loader, model, loss_fn, optimizer, device)
            res = test(test_loader, model, loss_fn, device, labels)
            res["Epoch"] = t
            if not quiet:
                print(f'Epoch: {res["Epoch"]}, accuracy: {res["Accuracy"]}, loss: {res["Avg loss"]}')
            fold_hist.append(res)
        tf = time.time()
        print(f"Done in {tf - t0}s. Accuracy {fold_hist[-1]['Accuracy']}")
        training_time.append(tf - t0)
        history.append(fold_hist)

    avg_res = {}
    with open(out, 'a', newline='') as csvfile:
        history = np.array(history)
        for e in range(epochs):
            for field in field_names:
                avg_res[field] = np.average([f[field] for f in history[:,e]])
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow(avg_res)

    meta_info['training time'] = np.average(training_time)
    meta_info['accuracy'] = avg_res['Accuracy']
    meta_info['micro'] = avg_res['F-measure (micro)']
    meta_info['macro'] = avg_res['F-measure (macro)']
    meta_info['weighted'] = avg_res['F-measure (weighted)']
    meta_info['avg. loss'] = avg_res['Avg loss']
    with open(meta_out, 'a', newline='') as infofile:
        writer = csv.DictWriter(infofile, fieldnames=meta_field_names)
        writer.writerow(meta_info)