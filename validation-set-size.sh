#!/bin/sh

python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-validation-size --n_layers 2 --validation_size 0.20 
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-validation-size --n_layers 2 --validation_size 0.20
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-validation-size --n_layers 3 --validation_size 0.20
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-validation-size --n_layers 4 --validation_size 0.20
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-validation-size --n_layers 5 --validation_size 0.20
