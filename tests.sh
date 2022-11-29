#!/bin/sh
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 1NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 2NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 4NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 8NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 16NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 1NNFeature --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 2NNFeature --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 4NNFeature --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 8NNFeature --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 16NNFeature --slic_method SLIC0 

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 1NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 2NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 4NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 8NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 16NNSpatial --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 1NNFeature --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 2NNFeature --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 4NNFeature --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 8NNFeature --slic_method SLIC0 
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 16NNFeature --slic_method SLIC0 
