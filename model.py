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
import cifar10_slic
import cifar100_slic

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
    for batch, b in enumerate(dataloader):
        b.to(device)
        pred = model(b.x, b.edge_index, b.batch)
        loss = loss_fn(pred, b.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            # print(f"loss: {loss:>7f}  [{(current*64):>5d}/{size:>5d}]")

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


def load_dataset(n_segments, compactness, features, train_dir, test_dir, dataset):
    if dataset == 'mnist':
        test_ds  = mnist_slic.SuperPixelGraphMNIST(root=test_dir, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   train=False)
        train_ds = mnist_slic.SuperPixelGraphMNIST(root=train_dir, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   train=True)
        labels = list(range(10))
    if dataset == 'cifar10':
        test_ds  = cifar10_slic.SuperPixelGraphCIFAR10(root=test_dir, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       train=False)
        train_ds = cifar10_slic.SuperPixelGraphCIFAR10(root=train_dir, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       train=True)
        labels = list(range(10))
    if dataset == 'cifar100':
        test_ds  = cifar100_slic.SuperPixelGraphCIFAR100(root=test_dir, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       train=False)
        train_ds = cifar100_slic.SuperPixelGraphCIFAR100(root=train_dir, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       train=True)
        labels = list(range(100))
    ds = ConcatDataset([train_ds, test_ds])
    targets = torch.cat([train_ds.get_targets(), test_ds.get_targets()])
    splits = StratifiedKFold(n_splits=5).split(np.zeros(len(targets)), targets)

    return ds, splits, labels

if __name__ == '__main__':
    import argparse
    import csv
    import time

    parser = argparse.ArgumentParser()
    cifar10_slic.add_ds_args(parser)
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="model's learning rate")
    parser.add_argument("--out", default=None,
                        help="output file")
    parser.add_argument("--metaout", default=None,
                        help="output file for information about training")
    parser.add_argument("--dataset", default='mnist',
                        help="dataset to train against")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    field_names = ["Epoch", "Accuracy", "F-measure (micro)", "F-measure (macro)", "F-measure (weighted)", "Avg loss"]
    meta_field_names = ['n_segments', 
                        'compactness', 
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
                                      args.traindir,
                                      args.testdir,
                                      args.dataset)
    if args.out is None:
        out = '{}-n{}-c{}-{}.csv'.format(args.dataset,
                                         ds.datasets[0].n_segments,
                                         ds.datasets[0].compactness,
                                         '-'.join(ds.datasets[0].features))
    else:
        out = args.out + '.csv'
    
    if args.metaout is None:
        meta_out = '{}-n{}-c{}-{}.meta.csv'.format(args.dataset,
                                         ds.datasets[0].n_segments,
                                         ds.datasets[0].compactness,
                                         '-'.join(ds.datasets[0].features))
    else:
        meta_out = args.metaout + '.csv'
    meta_info = {}

    with open(out, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

    train_ds, test_ds = ds.datasets[0], ds.datasets[1]
    meta_info['loading time'] = train_ds.loading_time
    meta_info['avg. num. of nodes'] = train_ds.avg_num_nodes
    meta_info['std. dev. of num. of nodes'] = train_ds.std_deviation_num_nodes
    meta_info['avg. num. of edges'] = train_ds.avg_num_edges
    meta_info['std. dev. of num. of edges'] = train_ds.std_deviation_num_edges
    meta_info['n_segments']  = train_ds.n_segments
    meta_info['compactness'] = train_ds.compactness
    meta_info['features'] = ' '.join(train_ds.features)

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
        print(f"Done in {tf - t0}s.")
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