import torch
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as T
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

import mnist_slic 
import cifar10_slic

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

def test(dataloader, model, loss_fn, device):
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
    f1_micro = f1_score(Y, Y_pred, average='micro', labels=[0,1,2,3,4,5,6,7,8,9])
    f1_macro = f1_score(Y, Y_pred, average='macro', labels=[0,1,2,3,4,5,6,7,8,9])
    f1_weighted = f1_score(Y, Y_pred, average='weighted', labels=[0,1,2,3,4,5,6,7,8,9])
    return {"Accuracy": accuracy, "F-measure (micro)": f1_micro, "F-measure (macro)": f1_macro, "F-measure (weigthed)": f1_weighted, "Avg loss": test_loss}


def load_models(n_segments, compactness, features, train_dir, test_dir, dataset):
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
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    return train_ds, train_loader, test_ds, test_loader

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
    parser.add_argument("--dataset", default='mnist',
                        help="dataset to train against")
    args = parser.parse_args()

    field_names = ["Epoch", "Accuracy", "F-measure (micro)", "F-measure (macro)", "F-measure (weigthed)", "Avg loss"]
    
    if args.features is not None:
        args.features = args.features.split()

    train_ds, train_loader, test_ds, test_loader = load_models(args.n_segments,
                                                               args.compactness,
                                                               args.features,
                                                               args.traindir,
                                                               args.testdir,
                                                               args.dataset)
    if args.out is None:
        out = '{}-n{}-c{}-{}.csv'.format(args.dataset,
                                         train_ds.n_segments,
                                         train_ds.compactness,
                                         '-'.join(train_ds.features))
    else:
        out = args.out

    with open(out, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = GCN(train_ds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = args.epochs
    t0 = time.time()
    for t in range(epochs):
        train(train_loader, model, loss_fn, optimizer, device)
        res = test(test_loader, model, loss_fn, device)
        res["Epoch"] = t
        print(f'Epoch: {res["Epoch"]}, accuracy: {res["Accuracy"]}, loss: {res["Avg loss"]}')
        with open(out, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow(res)
    tf = time.time()

    print(f"Done in {tf - t0}s.")