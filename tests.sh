#!/bin/sh

python model.py --features "avg_color std_deviation_color centroid" --dataset cifar10 --quiet
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --quiet
python model.py --features "avg_color std_deviation_color centroid num_pixels" --dataset cifar10 --quiet


python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --n_segments 10 --quiet
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --n_segments 20 --quiet
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --n_segments 50 --quiet
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --n_segments 400 --quiet
