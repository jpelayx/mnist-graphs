#!/bin/sh
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type RAG --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type RAG --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --graph_type RAG --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --graph_type RAG --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stl10 --graph_type RAG --slic_method SLIC0  --quiet
