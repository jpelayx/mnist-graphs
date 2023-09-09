#!/bin/sh 

python3 train-model.py -m AlexNet --dataset mnist -f AlexNet/mnist 
python3 train-model.py -m AlexNet --dataset fashion_mnist -f AlexNet/fashion_mnist 
python3 train-model.py -m AlexNet --dataset cifar10 -f AlexNet/cifar10 
python3 train-model.py -m AlexNet --dataset cifar100 -f AlexNet/cifar100 
python3 train-model.py -m AlexNet --dataset stl10 -f AlexNet/stl10 
