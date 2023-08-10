#!/bin/sh
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-node --num_nodes 10
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-node --num_nodes 20
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-node --num_nodes 50
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-node --num_nodes 100
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-node --num_nodes 200
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-node --num_nodes 400

python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-node --num_nodes 10
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-node --num_nodes 20
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-node --num_nodes 50
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-node --num_nodes 100
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-node --num_nodes 200
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-node --num_nodes 400

python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-node --num_nodes 10
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-node --num_nodes 20
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-node --num_nodes 50
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-node --num_nodes 100
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-node --num_nodes 200
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-node --num_nodes 400

python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-node --num_nodes 10
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-node --num_nodes 20
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-node --num_nodes 50
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-node --num_nodes 100
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-node --num_nodes 200
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-node --num_nodes 400

python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-node --num_nodes 10
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-node --num_nodes 20
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-node --num_nodes 50
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-node --num_nodes 100
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-node --num_nodes 200
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-node --num_nodes 400

