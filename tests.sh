#!/bin/sh

python model.py --features "avg_color" --dataset cifar100 --quiet --metaout cifar100
python model.py --features "avg_color centroid" --dataset cifar100 --quiet --metaout cifar100
python model.py --features "avg_color std_deviation_color centroid" --dataset cifar100 --quiet --metaout cifar100
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --quiet --metaout cifar100
python model.py --features "avg_color std_deviation_color centroid num_pixels" --dataset cifar100 --quiet --metaout cifar100

python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --n_segments 10 --quiet --metaout cifar100
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --n_segments 20 --quiet --metaout cifar100
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --n_segments 50 --quiet --metaout cifar100
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --n_segments 100 --quiet --metaout cifar100
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --n_segments 200 --quiet --metaout cifar100
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --n_segments 400 --quiet --metaout cifar100
