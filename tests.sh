#!/bin/sh
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 1NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 2NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 4NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 8NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset mnist --graph_type 16NNFeature --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 1NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 2NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 4NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 8NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset fashion_mnist --graph_type 16NNFeature --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --graph_type 1NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --graph_type 2NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --graph_type 4NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --graph_type 8NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar10 --graph_type 16NNFeature --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --graph_type 1NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --graph_type 2NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --graph_type 4NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --graph_type 8NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset cifar100 --graph_type 16NNFeature --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stl10 --graph_type 1NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stl10 --graph_type 2NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stl10 --graph_type 4NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stl10 --graph_type 8NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stl10 --graph_type 16NNFeature --slic_method SLIC0  --quiet
