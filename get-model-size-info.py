import torch 
import numpy as np

import csv 

if __name__ == '__main__':

    dss = [
        'mnist', 
        'fashion_mnist', 
        'cifar10',
        'cifar100',
        'stl10'
    ]

    num_heads = [1, 2, 4, 8, 16]

    out_fields = [
        'num. of heads', 
        'model size'
    ]


    for ds in dss:
        basedir = f'GAT/{ds}/'
        out_file = f'GAT/{ds}-heads-modelsizeinfo.csv'
        with open(out_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=out_fields)
            writer.writeheader()
        sizes = []
        print(ds)
        for h in num_heads:
            fold_sizes = []
            for f in range(5):
                model_path = basedir + f'l4h{h}n75-RAG-SLIC0-avg_color-centroid-std_deviation_centroid-std_deviation_color.fold{f}.pth'
                model = torch.load(model_path)
                fold_size = 0
                for layer in model:
                    layer_size = np.prod(model[layer].shape) * model[layer].element_size()
                    fold_size += layer_size
                fold_sizes.append(fold_size)
            mean_size = np.mean(fold_sizes)
            sizes.append(mean_size)
            print(h, mean_size)
            with open(out_file, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=out_fields)
                writer.writerow({'num. of heads': h, 'model size': mean_size})

        



