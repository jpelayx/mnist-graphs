#!/bin/sh

python model.py --features "avg_color" --dataset fashion_mnist --quiet 
python model.py --features "avg_color centroid" --dataset fashion_mnist --quiet 
python model.py --features "avg_color std_deviation_color centroid" --dataset fashion_mnist --quiet 
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --quiet 
python model.py --features "avg_color std_deviation_color centroid num_pixels" --dataset fashion_mnist --quiet 

python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --n_segments 10 --quiet 
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --n_segments 20 --quiet 
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --n_segments 50 --quiet 
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --n_segments 100 --quiet 
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --n_segments 200 --quiet 
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --n_segments 400 --quiet 
