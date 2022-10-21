#!/bin/sh

python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv" --dataset cifar100 --quiet  
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv std_deviation_color_hsv" --dataset cifar100 --quiet  
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv" --dataset stanfordcars --quiet
python model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv std_deviation_color_hsv" --dataset stanfordcars --quiet  
