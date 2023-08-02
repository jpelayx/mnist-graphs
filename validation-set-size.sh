#!/bin/sh

python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-validation-size --n_heads  1 --validation_size 0.25 
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-validation-size --n_heads  2 --validation_size 0.25
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-validation-size --n_heads  4 --validation_size 0.25
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-validation-size --n_heads  8 --validation_size 0.25
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-validation-size --n_heads 16 --validation_size 0.25
