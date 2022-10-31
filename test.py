from matplotlib.transforms import TransformedBbox
import torch
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as T
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

from compute_features import grayscale_features, color_features

ds_gray = datasets.FashionMNIST(root = "./fashion_mnist/test", train=False, download=True, transform=T.ToTensor())
ds_color = datasets.CIFAR10(root='cifar10/test', train=False, download=True, transform=T.ToTensor())
img, y = ds_color[1]
_, dim0, dim1 = img.shape
img_np = torch.stack([img[0], img[1], img[2]], dim=2).numpy()
f, e = color_features(img_np, 20, 0.1)
print(f.shape, e.shape)