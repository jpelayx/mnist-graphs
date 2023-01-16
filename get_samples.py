from dataset_loader import load_dataset

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np


idx = 23

dataset = 'cifar10'
n_segments = 10
compactness = 0.1
graph_type = 'RAG'
slic_method = 'SLIC0'
features = ['avg_color',
            'std_deviation_color',
            'centroid',
            'std_deviation_centroid']
ds = None

def graph_name():
    return '{}-idx{}-n{}-c{}-{}-{}.png'.format(dataset,
                                               idx,
                                               n_segments, 
                                               graph_type,
                                               slic_method if slic_method == 'SLIC0' else slic_method + 'c' + str(compactness),
                                               '-'.join(features))
def draw_graph():
    g = ds[idx]
    nx_g = to_networkx(g, to_undirected=True)
    nx_color = g.x[:,0:3].numpy()
    nx_pos = dict(zip(range(g.num_nodes), g.pos.numpy()))
    nx.draw(nx_g, pos=nx_pos, node_color=nx_color)

def load_ds():
    d, _, _ = load_dataset(5, 
                            n_segments,
                            compactness,
                            features,
                            graph_type,
                            slic_method,
                            dataset,
                            False)
    return d.datasets[0]

print('--------------')
print(dataset)

ds = load_ds()

og_img = ds.get_og_img(idx)
og_img_name = dataset + '_idx' + str(idx) + '.png'
plt.imsave(og_img_name, og_img)
print('sample image: ' + og_img_name)

draw_graph()
plt.savefig(graph_name())
print(' + ' + graph_name())

n_segments = 20
ds = load_ds()
draw_graph()
plt.savefig(graph_name())
print(' + ' + graph_name())