#!/bin/sh

python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv avg_lightness" --dataset cifar10 --quiet  
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv avg_lightness std_deviation_lightness" --dataset cifar10 --quiet  
